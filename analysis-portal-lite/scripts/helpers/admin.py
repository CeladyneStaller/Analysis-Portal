"""
Admin operations on the stored index and detail bins.

Every operation is preview-then-apply: called with ``apply=False`` it reports
what *would* change and writes nothing, which is the CLI tools' dry-run split
expressed as something a UI can render. Nothing here decides on the operator's
behalf — which name survives a merge, which stand a legacy run was on, whether
a sample should be destroyed — because none of those are recoverable from the
data.

The primitives are shared with the CLI tools rather than reimplemented:
``dupdetect`` for matching, ``record.merge_*`` for merging, ``record``'s stand
vocabulary for stands. Only the orchestration differs, because a modal cannot
take ``--purge 3``.

Writes take the locks pushes take, in the same order: the lock of every sample
they touch (in name order, so two admin writes cannot deadlock either), held
for the whole operation, then the index lock for one read-modify-write of the
index as it stands at that moment (`_patch_index`). The portal runs these in
worker threads, so without that a push finishing mid-operation, or a second
operation, could lose the other's change or have its own undone.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import inspect
import json
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

from scripts.helpers import dupdetect, jsonbin, registry
from scripts.helpers.record import (
    FORBIDDEN_NAME_CHARS, PRE_ELX_TYPES, STAND_OPTIONS, build_index_entry,
    decode_sidecars, guesses_stand, merge_detail_record, merge_index_entry,
    normalise_electrolyzer, old_electrolyzer_cells, parse_run_date,
    plot_bucket, stand_family, stand_number, touched_units, unplaced_rows,
)

MAX_PREVIEW_ROWS = 400


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _key(entry: Dict[str, Any]) -> str:
    """The identifier the portal addresses a run by, matching viewstore."""
    return str(entry.get('bin_id') or entry.get('job_id') or '')


def _entries(index: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(index.get('runs') or [])


def _snapshot(keys: List[str]) -> Dict[str, Any]:
    """The backup every apply takes before it writes anything.

    Taken here rather than left to the operator: an offered backup is only as
    good as the habit of taking it, and with restore in place an untaken one is
    the difference between an inconvenience and a loss.

    Scoped to the samples being touched, so the cost is a handful of detail
    reads rather than the whole store. The index snapshot is included whole
    because it is one small read and it is what puts entries back.

    Raises if it cannot be built. A record that cannot be backed up is one that
    cannot be restored, and the operation should not proceed without it.
    """
    try:
        return build_backup(keys, include_records=True)
    except Exception as e:
        raise RuntimeError(
            f'could not take a backup before writing ({type(e).__name__}: {e}); '
            f'nothing was changed') from e


# ─────────────────────────────────────────────────────────────────────
#  Locking
# ─────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _samples_locked(names: Iterable[str]):
    """Hold the push locks of the named samples, in name order."""
    keys = sorted({str(n or '') or '__unnamed__' for n in names})
    with contextlib.ExitStack() as stack:
        for k in keys:
            stack.enter_context(jsonbin._sample_lock(k))
        yield


def _patch_index(change: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]
                 ) -> None:
    """Rewrite the index as it stands now: change(runs) -> runs, under the lock.

    Read here rather than reusing the copy an operation started from, so
    entries written since — another sample's push, another admin change — are
    kept. The entries the operation itself changes cannot have moved on: their
    samples' locks are held.
    """
    with jsonbin._index_lock:
        index = jsonbin.fetch_index()
        index['runs'] = change(_entries(index))
        jsonbin._write_index(index)


def _names_for(index: Dict[str, Any], keys: Iterable[Any]) -> Set[str]:
    """Sample names of the entries with these keys."""
    want = {str(k) for k in keys or []}
    return {e.get('sample_name') or '' for e in _entries(index) if _key(e) in want}


def _locked_when_applied(names_of: Callable[[Dict[str, Any], Dict[str, Any]], Set[str]]):
    """Run the decorated operation, when applied, holding its samples' locks.

    `names_of(index, args)` names the samples the call touches, from an index
    read. It is asked again once the locks are held: a name changed in the
    meantime — another rename, a merge — means a lock is missing, and the
    locks are taken again with it. A preview takes no locks.
    """
    def deco(fn):
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def run(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            if not bound.arguments.get('apply'):
                return fn(*args, **kwargs)
            names = set(names_of(jsonbin.fetch_index(), bound.arguments))
            for _ in range(3):
                with _samples_locked(names):
                    now = set(names_of(jsonbin.fetch_index(), bound.arguments))
                    if now <= names:
                        return fn(*args, **kwargs)
                names |= now
            raise RuntimeError('the samples kept changing while this waited for '
                               'them; nothing was changed — try again')
        return run
    return deco


# ─────────────────────────────────────────────────────────────────────
#  Inventory
# ─────────────────────────────────────────────────────────────────────

def list_samples() -> Dict[str, Any]:
    """Every stored sample, for populating the admin panels."""
    index = jsonbin.fetch_index()
    out = []
    for e in _entries(index):
        out.append({
            'key': _key(e),
            'sample_name': e.get('sample_name') or '',
            'run_date': e.get('run_date') or str(e.get('timestamp') or '')[:10],
            'stand': e.get('stand') or '',
            'script': e.get('script') or '',
            'units': len(e.get('Data') or []),
        })
    out.sort(key=lambda r: (r['run_date'], r['sample_name']), reverse=True)
    return {'samples': out, 'total': len(out), 'stands': list(STAND_OPTIONS)}


# ─────────────────────────────────────────────────────────────────────
#  Duplicates
# ─────────────────────────────────────────────────────────────────────

def find_duplicates() -> Dict[str, Any]:
    """Groups of entries that look like the same sample stored twice.

    Two kinds of evidence, reported separately because they justify different
    confidence: identical measurements over the overlapping units, and names
    that differ only in punctuation or case. A name match also reports how many
    fields agree, since a re-analysis with different inputs will differ on some
    of them and that is worth seeing before merging.
    """
    index = jsonbin.fetch_index()
    by_key = {_key(e): e for e in _entries(index)}
    cache: Dict[str, Dict[str, Any]] = {}

    def detail(entry):
        k = _key(entry)
        if k not in cache:
            cache[k] = jsonbin.fetch_detail_bin(entry.get('bin_id'))
        return cache[k]

    pairs, seen = [], set()
    rejected, unreadable = [], []
    ext_pairs = set()

    def _consider(a, b, why):
        """Compare one candidate pair, keeping the outcome either way.

        A rejection used to be dropped on the floor, so a pair that was looked
        at and turned down was indistinguishable from one never considered —
        which is exactly the report someone gets when they can see two obvious
        duplicates and the tool says none. The reason is kept and shown.
        """
        pk = (_key(a), _key(b))
        if pk in seen:
            return None
        try:
            res = dupdetect.compare_records(detail(a), detail(b))
        except Exception as exc:
            # A bin that cannot be read is not evidence of anything, and
            # swallowing it silently hid real failures behind "no duplicates".
            unreadable.append({'pair': pk,
                               'names': [a.get('sample_name', ''), b.get('sample_name', '')],
                               'error': f'{type(exc).__name__}: {exc}'})
            return None
        if res.is_duplicate:
            seen.add(pk)
            pairs.append(pk)
            return res
        rejected.append({
            'pair': pk, 'via': why,
            'names': [a.get('sample_name', ''), b.get('sample_name', '')],
            'matched': res.matched_fields, 'reason': res.reason,
            'differing': ({'analysis': res.contradiction[0], 'step': res.contradiction[1],
                           'field': res.contradiction[2], 'a': res.contradiction[3],
                           'b': res.contradiction[4]} if res.contradiction else None),
        })
        return None

    for a, b in dupdetect.index_prefilter(index):
        _consider(a, b, 'measurements')

    # One name extending the other — the same cell recorded twice with a
    # qualifier appended. Neither of the other two prefilters can see it.
    for a, b in dupdetect.name_extension_prefilter(index):
        res = _consider(a, b, 'name extension')
        if res:
            ext_pairs.add((_key(a), _key(b)))

    name_pairs = []
    for a, b in dupdetect.name_prefilter(index):
        pk = (_key(a), _key(b))
        if pk in seen:
            continue
        try:
            agreed, differed = dupdetect.field_agreement(detail(a), detail(b))
        except Exception:
            agreed, differed = 0, []
        name_pairs.append({'pair': pk, 'agreed': agreed,
                           'differed': [{'analysis': d[0], 'step': d[1],
                                         'field': d[2], 'a': d[3], 'b': d[4]}
                                        for d in differed[:6]],
                           'differed_total': len(differed)})
        pairs.append(pk)

    groups = dupdetect.group_matches(pairs)
    name_by_pair = {p['pair']: p for p in name_pairs}

    out = []
    for n, g in enumerate(groups, start=1):
        names = sorted({by_key[k].get('sample_name', '') for k in g if k in by_key})
        evidence, agreed, differed = 'measurements', None, []
        if any(pk in ext_pairs for pk in
               [(x, y) for x in g for y in g if x != y]):
            evidence = 'name extension'
        for pk, info in name_by_pair.items():
            if pk[0] in g and pk[1] in g:
                evidence = 'name'
                agreed, differed = info['agreed'], info['differed']
                break
        out.append({
            'group': n,
            'keys': list(g),
            'names': names,
            'evidence': evidence,
            'fields_agreed': agreed,
            'fields_differed': differed,
            'members': [{'key': k, 'sample_name': by_key[k].get('sample_name', ''),
                         'run_date': by_key[k].get('run_date') or '',
                         'stand': by_key[k].get('stand') or ''}
                        for k in g if k in by_key],
        })
    return {'groups': out, 'scanned': len(_entries(index)),
            'considered': len(pairs) + len(rejected),
            'rejected': rejected[:MAX_PREVIEW_ROWS],
            'unreadable': unreadable[:MAX_PREVIEW_ROWS]}


@_locked_when_applied(lambda index, a: _names_for(index, a['keys'])
                      | {str(a['keep_name'] or '')})
def merge_duplicate_group(keys: List[str], keep_name: str,
                          apply: bool = False) -> Dict[str, Any]:
    """Merge one group into a single entry under ``keep_name``.

    One group per call. Each group is a different physical sample and takes its
    own name decision; applying one name across groups would give distinct
    cells the same identity and sample-keyed merging would then collapse them.
    """
    index = jsonbin.fetch_index()
    by_key = {_key(e): e for e in _entries(index)}
    members = [by_key[k] for k in keys if k in by_key]
    if len(members) < 2:
        raise ValueError('a merge needs at least two entries that still exist')

    names = {m.get('sample_name', '') for m in members}
    if keep_name not in names:
        raise ValueError(
            f'{keep_name!r} is not one of this group\'s names ({", ".join(sorted(names))}). '
            'Renaming to something new is almost always a mistyped command; '
            'rename deliberately afterwards instead.')

    ordered = sorted(members, key=lambda e: str(e.get('timestamp') or ''))
    survivor = ordered[-1].get('bin_id')

    merged_detail, merged_entry = None, None
    for m in ordered:
        merged_detail = merge_detail_record(
            merged_detail, jsonbin.fetch_detail_bin(m.get('bin_id')))
        merged_entry = merge_index_entry(merged_entry, m)
    merged_detail['sample_name'] = keep_name
    merged_entry['sample_name'] = keep_name
    merged_entry['bin_id'] = survivor

    superseded = [m.get('bin_id') for m in ordered if m.get('bin_id') != survivor]
    preview = {
        'keep_name': keep_name,
        'survivor_bin': survivor,
        'superseded_bins': superseded,
        'units_after': len(merged_entry.get('Data') or []),
        'entries_removed': len(members) - 1,
        'applied': False,
    }
    if not apply:
        return preview

    preview['backup'] = _snapshot([_key(m) for m in members])
    jsonbin.update_detail_bin(survivor, merged_detail)
    drop = {_key(m) for m in members}
    _patch_index(lambda runs: [e for e in runs if _key(e) not in drop]
                 + [merged_entry])
    preview['applied'] = True
    # Superseded bins are left in place rather than deleted: an orphan bin is
    # an accepted state, and destroying data during a merge would make a
    # reversible operation irreversible.
    return preview


# ─────────────────────────────────────────────────────────────────────
#  Rename
# ─────────────────────────────────────────────────────────────────────

@_locked_when_applied(lambda index, a: _names_for(index, [a['key']])
                      | {str(a['new_name'] or '').strip()})
def rename_sample(key: str, new_name: str, apply: bool = False) -> Dict[str, Any]:
    """Rename one stored sample.

    Writes the detail record as well as the index entry. ``build_index_entry``
    reads ``sample_name`` from the detail record, so the next push for this
    sample rebuilds its entry from the record — an index-only rename would
    silently revert.
    """
    new_name = str(new_name or '').strip()
    if not new_name:
        raise ValueError('a new name is required')

    bad = sorted({c for c in new_name if c in FORBIDDEN_NAME_CHARS})
    if bad:
        raise ValueError(
            f'{new_name!r} contains {" ".join(bad)}, which Windows does not '
            f'allow in a file name')
    if any(ord(c) < 32 for c in new_name):
        raise ValueError('a sample name cannot contain control characters')

    index = jsonbin.fetch_index()
    entries = _entries(index)
    target = next((e for e in entries if _key(e) == key), None)
    if target is None:
        raise ValueError(f'no stored sample with key {key!r}')
    old_name = target.get('sample_name') or ''
    if new_name == old_name:
        raise ValueError('that is already the name')

    # Sample-keyed merging identifies a sample by name, so two entries sharing
    # one are merged into a single bin on the next push. Renaming onto an
    # existing name would create exactly the corruption the duplicate tool
    # exists to clean up.
    clash = [e for e in entries
             if e.get('sample_name') == new_name and _key(e) != key]
    if clash:
        raise ValueError(
            f'{new_name!r} is already used by another stored sample. Two entries '
            f'under one name are merged into a single bin on the next push, so '
            f'this would make two different cells share an identity.')

    old_date = target.get('run_date')
    new_date = parse_run_date(new_name)
    preview = {
        'key': key, 'from': old_name, 'to': new_name,
        'run_date_from': old_date, 'run_date_to': new_date,
        'run_date_lost': bool(old_date) and not new_date,
        'applied': False,
    }
    if not apply:
        return preview

    preview['backup'] = _snapshot([key])
    bin_id = target.get('bin_id')
    detail = jsonbin.fetch_detail_bin(bin_id)
    detail['sample_name'] = new_name
    jsonbin.update_detail_bin(bin_id, detail)

    # Rebuild from the record so run_date follows the new name rather than
    # being patched in two places that could disagree.
    rebuilt = build_index_entry(detail, bin_id)
    merged = merge_index_entry(target, rebuilt)
    merged['sample_name'] = new_name
    if new_date:
        merged['run_date'] = new_date
    else:
        merged.pop('run_date', None)
    _patch_index(lambda runs: [e for e in runs if _key(e) != key] + [merged])
    preview['applied'] = True
    return preview


# ─────────────────────────────────────────────────────────────────────
#  Test stand
# ─────────────────────────────────────────────────────────────────────

@_locked_when_applied(lambda index, a: _names_for(index, a['keys']))
def set_stand(keys: List[str], stand: str, apply: bool = False,
              allow_family_change: bool = False) -> Dict[str, Any]:
    """Set the recorded test stand on the given samples.

    Writes the detail record too, for the same reason rename does.
    """
    if stand not in STAND_OPTIONS:
        raise ValueError(f'{stand!r} is not a known stand '
                         f'({", ".join(STAND_OPTIONS)})')
    index = jsonbin.fetch_index()
    by_key = {_key(e): e for e in _entries(index)}
    unknown = [k for k in keys if k not in by_key]
    if unknown:
        raise ValueError(f'no stored sample with key(s): {", ".join(unknown)}')

    changes, crossing = [], []
    for k in keys:
        e = by_key[k]
        cur = e.get('stand') or None
        if cur == stand:
            continue
        row = {'key': k, 'sample_name': e.get('sample_name', ''),
               'from': cur, 'to': stand}
        changes.append(row)
        if cur and stand_family(cur) != stand_family(stand):
            crossing.append(row)

    preview = {'changes': changes[:MAX_PREVIEW_ROWS], 'total': len(changes),
               'crossing_family': crossing[:MAX_PREVIEW_ROWS],
               'crossing_total': len(crossing), 'applied': False}
    if not apply:
        return preview
    if crossing and not allow_family_change:
        raise ValueError(
            f'{len(crossing)} of these would change family (Scribner ↔ FCTS). '
            f'The family is derived from the data format, so crossing it '
            f'usually means the wrong entries are selected.')

    preview['backup'] = _snapshot([r['key'] for r in changes])
    for row in changes:
        e = by_key[row['key']]
        bin_id = e.get('bin_id')
        rec = jsonbin.fetch_detail_bin(bin_id)
        rec['stand'] = stand
        jsonbin.update_detail_bin(bin_id, rec)
    changed = {r['key'] for r in changes}
    _patch_index(lambda runs: [dict(e, stand=stand) if _key(e) in changed else e
                               for e in runs])
    preview['applied'] = True
    return preview


def find_stand_drift(keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """Index entries whose stand is vaguer than the detail record's.

    The detail record keeps what was declared; before `better_stand` the index
    could be overwritten with the bare family derived from a file extension.
    Anything this reports was demoted rather than chosen, and the number is
    still there to put back.

    `keys` limits the check to those entries.
    """
    index = jsonbin.fetch_index()
    wanted = None if keys is None else set(keys)
    out = []
    for e in _entries(index):
        if wanted is not None and _key(e) not in wanted:
            continue
        idx_stand = str(e.get('stand') or '')
        try:
            rec = jsonbin.fetch_detail_bin(e.get('bin_id'))
        except Exception:
            continue
        rec_stand = str(rec.get('stand') or '')
        if not rec_stand or rec_stand == idx_stand:
            continue
        # Only the demotion case: same family, the record names a number and
        # the index does not. A genuine disagreement is not this tool's call.
        if (stand_family(rec_stand) == stand_family(idx_stand)
                and stand_number(rec_stand) is not None
                and stand_number(idx_stand) is None):
            out.append({'key': _key(e), 'sample_name': e.get('sample_name', ''),
                        'index': idx_stand or None, 'record': rec_stand})
    return {'drifted': out, 'scanned': len(_entries(index))}


def repair_stands(apply: bool = False) -> Dict[str, Any]:
    """Put the numbered stand back on entries that were demoted."""
    found = find_stand_drift()
    preview = {'repairs': found['drifted'], 'scanned': found['scanned'],
               'applied': False}
    if not apply or not found['drifted']:
        return preview

    # Checked again once the samples are locked: a push or a stand change
    # since the scan above may have settled some of them already.
    with _samples_locked(r['sample_name'] for r in found['drifted']):
        again = find_stand_drift([r['key'] for r in found['drifted']])
        names = {r['sample_name'] for r in found['drifted']}
        repairs = [r for r in again['drifted'] if r['sample_name'] in names]
        preview['repairs'] = repairs
        if not repairs:
            return preview
        preview['backup'] = _snapshot([r['key'] for r in repairs])
        fix = {r['key']: r['record'] for r in repairs}
        _patch_index(lambda runs: [dict(e, stand=fix[_key(e)]) if _key(e) in fix
                                   else e for e in runs])
    preview['applied'] = True
    return preview


# ─────────────────────────────────────────────────────────────────────
#  Backup, delete, restore
# ─────────────────────────────────────────────────────────────────────

def build_backup(keys: Optional[List[str]] = None,
                 include_records: bool = True) -> Dict[str, Any]:
    """A restorable snapshot of the named samples.

    Carries the full detail records, not just index entries: an entry alone
    cannot bring a sample back, because its measurements live in the bin.

    A record that cannot be read is listed under `unreadable` rather than
    silently left out, so the file says what it does not hold.
    """
    index = jsonbin.fetch_index()
    entries = _entries(index)
    wanted = None if keys is None else set(keys)
    chosen = [e for e in entries if wanted is None or _key(e) in wanted]
    payload = {
        'kind': 'analysis-portal-backup',
        'version': 1,
        'written': _now(),
        'index': index,
        'removed': chosen,
        'purged': [],
    }
    if include_records:
        for e in chosen:
            try:
                payload['purged'].append(
                    {'bin_id': e.get('bin_id'),
                     'record': jsonbin.fetch_detail_bin(e.get('bin_id'))})
            except Exception as exc:
                payload.setdefault('unreadable', []).append(
                    {'bin_id': e.get('bin_id'),
                     'sample_name': e.get('sample_name', ''),
                     'error': f'{type(exc).__name__}: {exc}'[:200]})
    return payload


@_locked_when_applied(lambda index, a: _names_for(index, a['keys']))
def delete_samples(keys: List[str], purge: bool = False,
                   apply: bool = False) -> Dict[str, Any]:
    """Remove samples from the index, and optionally destroy their data.

    Unlisting is reversible from a backup; purging is not, because JSONBin has
    no undelete. They are separate arguments so that destroying data is never
    a side effect of hiding it.
    """
    index = jsonbin.fetch_index()
    by_key = {_key(e): e for e in _entries(index)}
    unknown = [k for k in keys if k not in by_key]
    if unknown:
        # A key that matches nothing is far likelier to be a stale list than a
        # sample that has already gone, and acting on the rest would remove
        # things nobody checked.
        raise ValueError(f'no stored sample with key(s): {", ".join(unknown)}')

    chosen = [by_key[k] for k in keys]
    preview = {
        'samples': [{'key': _key(e), 'sample_name': e.get('sample_name', ''),
                     'run_date': e.get('run_date') or '',
                     'stand': e.get('stand') or '',
                     'bin_id': e.get('bin_id')} for e in chosen],
        'bins': [e.get('bin_id') for e in chosen if e.get('bin_id')],
        'purge': bool(purge), 'applied': False,
    }
    if not apply:
        return preview

    # Taken before anything is removed, purge or not: an unlist is reversible
    # only if something recorded what was there.
    preview['backup'] = _snapshot(keys)

    drop = set(keys)
    _patch_index(lambda runs: [e for e in runs if _key(e) not in drop])

    deleted, failed = 0, []
    if purge:
        for e in chosen:
            bid = e.get('bin_id')
            if not bid:
                continue
            try:
                jsonbin.delete_detail_bin(bid)
                deleted += 1
            except Exception as exc:
                failed.append({'bin_id': bid, 'error': str(exc)[:120]})
    preview['deleted_bins'] = deleted
    preview['failed'] = failed
    preview['applied'] = True
    return preview


def _bin_exists(bin_id: Any) -> bool:
    """Whether a detail bin is still stored."""
    if not bin_id:
        return False
    try:
        jsonbin.fetch_detail_bin(str(bin_id))
        return True
    except Exception:
        return False


def inspect_backup(payload: Dict[str, Any]) -> Dict[str, Any]:
    """What a backup file contains, and what restoring each sample would do.

    Rejects anything that is not a backup this portal wrote. Restore is the one
    path that takes a document from a person rather than from an analysis, so
    a malformed record would be written into the store and read back by every
    other view.
    """
    if not isinstance(payload, dict):
        raise ValueError('that file is not a backup')
    if payload.get('kind') not in (None, 'analysis-portal-backup'):
        raise ValueError('that file is not an analysis portal backup')
    removed = payload.get('removed')
    if not isinstance(removed, list) or not removed:
        raise ValueError('that backup lists no samples to restore')

    records = {str(p.get('bin_id')): p.get('record')
               for p in (payload.get('purged') or [])
               if isinstance(p, dict)}

    index = jsonbin.fetch_index()
    live_keys = {_key(e) for e in _entries(index)}
    live_names = {e.get('sample_name') for e in _entries(index)}
    live_bins = {e.get('bin_id') for e in _entries(index)}

    items = []
    for e in removed:
        if not isinstance(e, dict) or not e.get('sample_name'):
            continue
        k, name, bin_id = _key(e), e.get('sample_name'), e.get('bin_id')
        rec = records.get(str(bin_id))
        if k in live_keys or bin_id in live_bins:
            state, action = 'present', 'nothing to do'
        elif _bin_exists(bin_id):
            # The bin outliving its index entry is what an unlist leaves
            # behind. Asked rather than inferred from the backup's contents:
            # a backup always carries the records, and the bin may have been
            # destroyed by something else since it was written.
            state, action = 'unlisted', 'index it; its bin was never destroyed'
        elif rec:
            state, action = 'purged', 'rebuild the bin, then index it'
        else:
            state, action = 'lost', 'nothing to restore from — no saved record'
        # A name that exists again is the rename collision by another route.
        blocked = (state != 'present' and name in live_names)
        items.append({
            'key': k, 'sample_name': name,
            'run_date': e.get('run_date') or '',
            'stand': e.get('stand') or '',
            'bin_id': bin_id, 'state': state, 'action': action,
            'has_record': bool(rec),
            'blocked': blocked,
            'blocked_reason': (
                'a different sample is stored under that name now; restoring '
                'would make two cells share an identity' if blocked else None),
        })
    return {'written': payload.get('written'), 'items': items,
            'index_snapshot_entries': len(_entries(payload.get('index') or {}))}


@_locked_when_applied(lambda index, a: {str(n or '') for n in a['names'] or []})
def restore_backup(payload: Dict[str, Any], names: List[str],
                   apply: bool = False) -> Dict[str, Any]:
    """Put the named samples back, merging into the current index.

    Never applies the file's index snapshot as a whole: it is the index as it
    stood when the backup was written, and writing it back would revert every
    analysis pushed since, including other people's.
    """
    info = inspect_backup(payload)
    want = set(names or [])
    chosen = [i for i in info['items'] if i['sample_name'] in want]
    if not chosen:
        raise ValueError('none of those samples are in this backup')
    blocked = [i for i in chosen if i['blocked']]
    if blocked:
        raise ValueError(
            'cannot restore ' +
            ', '.join(i['sample_name'] for i in blocked[:4]) +
            ': a different sample is stored under that name now')

    lost = [i for i in chosen if i['state'] == 'lost']
    if lost:
        raise ValueError(
            'cannot restore ' + ', '.join(i['sample_name'] for i in lost[:4]) +
            ': the bin is gone and this backup holds no saved record for it')
    todo = [i for i in chosen if i['state'] not in ('present', 'lost')]
    preview = {'restoring': todo, 'skipped': [i for i in chosen
                                              if i['state'] == 'present'],
               'applied': False}
    if not apply:
        return preview

    # Restore adds rather than removes, but it still rewrites the index, and a
    # snapshot of what was there costs one read.
    preview['backup'] = _snapshot([])

    records = {str(p.get('bin_id')): p.get('record')
               for p in (payload.get('purged') or []) if isinstance(p, dict)}
    entries = {str(_key(e)): e for e in (payload.get('removed') or [])}

    written, restored = [], []
    for item in todo:
        entry = dict(entries.get(item['key']) or {})
        if item['state'] == 'purged':
            rec = records.get(str(item['bin_id']))
            if not rec:
                continue
            # JSONBin will not resurrect a deleted bin, so the record is
            # written as a new one and the entry repointed. The old bin_id in
            # the backup is a label, not an address.
            new_bin = jsonbin.create_detail_bin(
                rec, name=f"{item['sample_name']}-restored")
            entry = build_index_entry(rec, new_bin)
            item['new_bin_id'] = new_bin
        restored.append(entry)
        written.append(item)

    def change(runs):
        for entry in restored:
            runs = [e for e in runs if _key(e) != _key(entry)] + [entry]
        return runs
    _patch_index(change)
    preview['applied'] = True
    preview['restoring'] = written
    return preview


# ─────────────────────────────────────────────────────────────────────
#  Electrolyzer entries
# ─────────────────────────────────────────────────────────────────────
#
# Electrolyzer runs pushed under earlier rules are stored in shapes the views
# cannot read: between Aug 4 and the registry, one bucket per plot type with no
# cell and no values; before that, the right buckets but fit-derived values in
# the index, the EIS unit split five ways by a misread setpoint, and a stand
# guessed from .csv files. Everything needed to file them properly is in what
# each run stored — every plot keeps its type and the cell its script declared
# — so they are rebuilt from the records, not re-analysed: the record by
# record.normalise_electrolyzer, exactly as a push merging into it would, and
# the entry's electrolyzer units from the record.

ELECTROLYZER_SCRIPTS = tuple(registry.FAMILIES['elx']['scripts'])

# Runs from before the pol-curve script named its plot types elx_* (Jul 27,
# until 16:15; record.PRE_ELX_TYPES) are filed as fuel-cell pol curves and EIS,
# and nothing in them says which cell they were on or holds a summary to take
# values from. They are reported and left as stored; re-running is the fix.

_CELL_IN_UPLOAD = re.compile(r'(?:^|[\\/])1_([A-Za-z0-9]+)_[^\\/]*$')

_BACKUP_ABOUT = (
    'Taken before an electrolyzer rebuild: the index, and every entry and '
    'record the rebuild replaced, as they were. Restore puts back samples that '
    'are missing; it does not undo a rebuild, which rewrites samples in place. '
    'To undo one, write these records and entries back.')


def _elx_unit(analysis: Any) -> bool:
    """An index unit of an electrolyzer analysis, under any name it has been
    stored with — 'elx_model_fit' is the pol curve's."""
    return registry.family(plot_bucket(str(analysis or ''))) == 'elx'


def _stepped(analysis: Any) -> bool:
    """An electrolyzer analysis whose units are cells."""
    a = registry.ANALYSES.get(str(analysis or '')) or {}
    return a.get('family') == 'elx' and not a.get('stepless')


def _may_be_electrolyzer(entry: Dict[str, Any]) -> bool:
    """Whether an entry may hold electrolyzer runs, from the index alone, so a
    scan reads only those records."""
    return (entry.get('script') in ELECTROLYZER_SCRIPTS
            or any(_elx_unit(u.get('Analysis')) for u in entry.get('Data') or []))


def _canon(units: List[Dict[str, Any]]) -> List[str]:
    """Units as an order-free, comparable list."""
    return sorted(json.dumps(u, sort_keys=True) for u in units)


def _unit_view(u: Dict[str, Any]) -> Dict[str, Any]:
    return {'analysis': u.get('Analysis', ''), 'step': u.get('step', ''),
            'values': dict(u.get('key_values') or {})}


def _plan_one(entry: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    """What rebuilding one entry does, from its entry and record. Writes nothing.

    The record is normalised as a push would. The entry's electrolyzer units
    are rebuilt from the normalised record; of the rest it keeps every unit
    whose plots are still in the record, exactly as stored — even where the
    current rules would build it differently, which is not this operation's
    business — and drops the others, as a push merging into it would: the
    catalyst-layer plots the old rules put in the fuel-cell 'CLR' bucket are
    re-filed under EIS, and an older merge could leave a unit behind. Its
    stand goes if it was only ever guessed (record.guesses_stand).

    A record re-filed past the transport limit is left as it is, entry and
    all, and reported: fitting it would mean dropping plot data, which this
    operation never does.
    """
    bin_id = entry.get('bin_id')
    fixed = normalise_electrolyzer(record)
    built = build_index_entry(fixed, bin_id)
    now = touched_units(fixed)

    def gone(u):
        return (u.get('Analysis', ''), u.get('step', '')) not in now

    old_data = list(entry.get('Data') or [])
    kept = [u for u in old_data if not _elx_unit(u.get('Analysis')) and not gone(u)]
    data = kept + [u for u in built['Data'] if _elx_unit(u['Analysis'])]
    new_entry = dict(entry)
    # Unchanged but for their order is unchanged: an entry already rebuilt, or
    # written by a push under the current rules, is not rewritten to reorder it.
    new_entry['Data'] = old_data if _canon(data) == _canon(old_data) else data
    if not fixed.get('stand') and not guesses_stand(fixed):
        new_entry.pop('stand', None)

    record_changed = fixed is not record
    body_bytes = (len(json.dumps(fixed, ensure_ascii=False).encode())
                  if record_changed else None)

    # What the operator should see before applying.
    before = [u for u in old_data if _elx_unit(u.get('Analysis')) or gone(u)]
    after = [u for u in new_entry['Data'] if _elx_unit(u.get('Analysis'))]

    def names(units):
        return {(plot_bucket(u.get('Analysis', '')), n)
                for u in units for n in (u.get('key_values') or {})}
    gained = sorted(names(after) - names(before))
    lost = sorted(names(before) - names(after))

    sidecars = decode_sidecars(fixed)
    cells = old_electrolyzer_cells(record)
    no_cell = [] if len(cells) == 1 else sorted(
        n for b, plots in (fixed.get('metrics') or {}).items() if _stepped(b)
        for n, e in plots.items()
        if not (e.get('conditions') or {}).get('step') and n not in sidecars)

    # Old rows the rules could give no cell, that no unit takes values from.
    summary = fixed.get('summary')
    rows = summary if isinstance(summary, list) else ([summary] if summary else [])
    unused = sum(1 for i in unplaced_rows(fixed)
                 if _stepped(rows[i].get('Analysis')) and 'step' not in rows[i])

    pre_elx = sorted(n for n, sc in sidecars.items()
                     if isinstance(sc, dict) and sc.get('plot_type') in PRE_ELX_TYPES)

    # Under the old rules every cell's plots had the same names, so a second
    # cell pushed under the sample replaced the first's. Nothing stored says
    # which cells were run, but the uploads say which were in the folder: in
    # a record pushed more than once under those rules, a cell they name that
    # has no plots may be one that was overwritten. Only a hint — a folder can
    # hold cells nobody analysed under this sample.
    stored_cells = {u.get('step', '') for u in after if _stepped(u.get('Analysis'))}
    upload_cells = sorted({m.group(1).lower() for f in record.get('input_files') or []
                           for m in [_CELL_IN_UPLOAD.search(str(f))] if m})
    runs = max(len(record.get('jobs') or []), int(entry.get('n_jobs') or 1))
    maybe_lost = (sorted(set(upload_cells) - stored_cells)
                  if runs > 1 and cells and stored_cells - {''} else [])

    too_large = bool(body_bytes and body_bytes > jsonbin.MAX_BODY_BYTES)
    status = ('too_large' if too_large
              else 'rebuild' if record_changed or new_entry != entry
              else 'old' if pre_elx else 'current')
    report = {
        'key': _key(entry), 'sample_name': entry.get('sample_name', ''),
        'bin_id': bin_id,
        'units_before': [_unit_view(u) for u in before],
        'units_after': [_unit_view(u) for u in after],
        'gained': [{'analysis': a, 'name': n} for a, n in gained],
        'removed': [{'analysis': a, 'name': n,
                     'detail_only': not ((registry.metric(a, n) or {}).get('index', True))}
                    for a, n in lost],
        'stand_cleared': (entry.get('stand')
                          if entry.get('stand') and 'stand' not in new_entry else None),
        'rows_collapsed': (len(record['summary']) if isinstance(record.get('summary'), list)
                           else int(bool(record.get('summary')))) - len(rows),
        'rows_unattributed': unused,
        'plots_without_cell': no_cell[:MAX_PREVIEW_ROWS],
        'old_cells': sorted(cells),
        'upload_cells': upload_cells,
        'cells_missing': maybe_lost,
        'pre_elx_plots': len(pre_elx),
        'record_changed': record_changed, 'entry_changed': new_entry != entry,
        'body_bytes': body_bytes,
    }
    return {'status': status, 'entry': entry, 'record': record, 'fixed': fixed,
            'new_entry': new_entry, 'report': report}


def _plan_all(index: Dict[str, Any],
              entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """_plan_one for each of `entries`, reading its record. One that cannot be
    read or rebuilt — or whose bin another index entry also points at, which
    the index patch could not tell apart — is reported rather than stopping
    the rest."""
    shared = {k for k, n in Counter(_key(e) for e in _entries(index)).items() if n > 1}
    plans = []
    for e in entries:
        head = {'key': _key(e), 'sample_name': e.get('sample_name', ''),
                'bin_id': e.get('bin_id')}
        try:
            if not e.get('bin_id'):
                raise ValueError('the entry names no detail bin')
            if _key(e) in shared:
                raise ValueError('another index entry points at the same bin; '
                                 'merge them under Duplicates first')
            rec = jsonbin.fetch_detail_bin(e['bin_id'])
        except Exception as exc:
            plans.append({'status': 'unreadable', 'entry': e, 'report': dict(
                head, error=f'{exc}' if isinstance(exc, ValueError) else
                f'could not read the record ({type(exc).__name__}: {exc})'[:240])})
            continue
        try:
            plans.append(_plan_one(e, rec))
        except Exception as exc:
            plans.append({'status': 'unreadable', 'entry': e, 'report': dict(
                head, error=f'could not rebuild it ({type(exc).__name__}: {exc})'[:240])})
    return plans


def _plan_token(todo: List[Dict[str, Any]]) -> str:
    """A fingerprint of exactly what a rebuild would replace: each entry and
    record as read. Apply goes ahead only if it reads the same again."""
    items = sorted([_key(p['entry']), p['entry'], p['record']] for p in todo)
    return hashlib.sha256(json.dumps(items, sort_keys=True, ensure_ascii=False)
                          .encode()).hexdigest()


def _rebuild_report(plans: List[Dict[str, Any]]) -> Dict[str, Any]:
    by = lambda s: [p['report'] for p in plans if p['status'] == s]   # noqa: E731
    todo = by('rebuild')
    return {
        'entries': todo[:MAX_PREVIEW_ROWS],
        'total': len(todo),
        'current': len(by('current')),
        'left_as_stored': [{'key': r['key'], 'sample_name': r['sample_name'],
                            'plots': r['pre_elx_plots']} for r in by('old')][:MAX_PREVIEW_ROWS],
        'too_large': [{'key': r['key'], 'sample_name': r['sample_name'],
                       'body_bytes': r['body_bytes']} for r in by('too_large')][:MAX_PREVIEW_ROWS],
        'unreadable': by('unreadable')[:MAX_PREVIEW_ROWS],
        'scanned': len(plans),
        'applied': False,
    }


def rebuild_electrolyzer(apply: bool = False, prepare: bool = False,
                         keys: Optional[List[str]] = None,
                         token: str = '') -> Dict[str, Any]:
    """Rebuild every electrolyzer entry stored under earlier rules.

    Three calls, which read everything they report and differ in what comes
    back:

      preview          each entry that would change: its electrolyzer units
                       before and after, the values that reach the index and
                       the ones that leave it, the stand that was only
                       guessed, and what could not be given a cell. Writes
                       nothing.
      prepare=True     the same, plus a backup of every entry and record the
                       rebuild would replace and the index, the `keys` of
                       those entries, and a `token` for exactly that state.
                       Still writes nothing.
      apply=True       with that prepare's keys and token: holds those
                       samples' locks, reads them again, and writes only if
                       they are still exactly what the backup holds —
                       otherwise nothing, and the caller scans again.

    So the backup is in the operator's hands before anything is written,
    and what is written is precisely what it holds the originals of. Apply
    writes each changed record, then the index once; a record that fails to
    write leaves its entry as it was. Running it again changes nothing that is
    current, so a run cut short is finished by running it again.
    """
    if apply:
        return _rebuild_apply(list(keys or []), str(token or ''))
    index = jsonbin.fetch_index()
    plans = _plan_all(index, [e for e in _entries(index) if _may_be_electrolyzer(e)])
    out = _rebuild_report(plans)
    if prepare:
        todo = [p for p in plans if p['status'] == 'rebuild']
        out['keys'] = [_key(p['entry']) for p in todo]
        out['token'] = _plan_token(todo)
        out['backup'] = {
            'kind': 'analysis-portal-backup', 'version': 1, 'written': _now(),
            'about': _BACKUP_ABOUT, 'index': index,
            'removed': [p['entry'] for p in todo],
            'purged': [{'bin_id': p['entry'].get('bin_id'), 'record': p['record']}
                       for p in todo],
        }
    return out


def _rebuild_apply(keys: List[str], token: str) -> Dict[str, Any]:
    if not keys or not token:
        raise ValueError('prepare the rebuild first: its backup has to be taken '
                         'before anything is written')
    with _samples_locked(_names_for(jsonbin.fetch_index(), keys)):
        # Read again under the locks. A sample changed since the backup — a
        # push, a rename, another admin change — changes the token, and then
        # nothing is written: the backup no longer holds what would be
        # replaced. Only the prepared entries are read and written.
        index = jsonbin.fetch_index()
        want = set(keys)
        plans = _plan_all(index, [e for e in _entries(index) if _key(e) in want])
        todo = [p for p in plans if p['status'] == 'rebuild']
        if (sorted(_key(p['entry']) for p in todo) != sorted(want)
                or _plan_token(todo) != token):
            raise ValueError('The store changed after the backup was taken, so '
                             'nothing was written. Scan again.')

        out = _rebuild_report(plans)
        done, failed = [], []
        for p in todo:
            if p['fixed'] is not p['record']:
                try:
                    jsonbin.update_detail_bin(p['entry']['bin_id'], p['fixed'])
                except Exception as exc:
                    failed.append({'key': _key(p['entry']),
                                   'sample_name': p['entry'].get('sample_name', ''),
                                   'error': f'{type(exc).__name__}: {exc}'[:240]})
                    continue
            done.append(p)
        out['failed'] = failed
        out['rebuilt'] = len(done)
        replace = {_key(p['entry']): p['new_entry'] for p in done}
        if replace:
            try:
                _patch_index(lambda runs: [replace.get(_key(e), e) for e in runs])
            except Exception as exc:
                # The records are rewritten and the entries are not. Nothing
                # is lost — a rewritten record is the old one re-filed — and
                # the next run rebuilds the entries from it.
                out['error'] = (f'{len(done)} record(s) were rewritten but the '
                                f'index could not be ({type(exc).__name__}: {exc}). '
                                f'Scan and rebuild again to finish.')
                return out
        out['applied'] = bool(done)
    return out
