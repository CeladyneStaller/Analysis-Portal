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
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from scripts.helpers import dupdetect, jsonbin
from scripts.helpers.record import (
    FORBIDDEN_NAME_CHARS, STAND_OPTIONS, build_index_entry,
    merge_detail_record, merge_index_entry, parse_run_date, stand_family,
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
    index['runs'] = [e for e in _entries(index) if _key(e) not in drop] + [merged_entry]
    jsonbin._write_index(index)
    preview['applied'] = True
    # Superseded bins are left in place rather than deleted: an orphan bin is
    # an accepted state, and destroying data during a merge would make a
    # reversible operation irreversible.
    return preview


# ─────────────────────────────────────────────────────────────────────
#  Rename
# ─────────────────────────────────────────────────────────────────────

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
    index['runs'] = [e for e in entries if _key(e) != key] + [merged]
    jsonbin._write_index(index)
    preview['applied'] = True
    return preview


# ─────────────────────────────────────────────────────────────────────
#  Test stand
# ─────────────────────────────────────────────────────────────────────

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
        e['stand'] = stand
    jsonbin._write_index(index)
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
    """
    index = jsonbin.fetch_index()
    entries = _entries(index)
    chosen = [e for e in entries if keys is None or _key(e) in set(keys)]
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
            except Exception:
                pass
    return payload


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
    index['runs'] = [e for e in _entries(index) if _key(e) not in drop]
    jsonbin._write_index(index)

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

    index = jsonbin.fetch_index()
    written = []
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
        index['runs'] = [e for e in _entries(index) if _key(e) != _key(entry)]
        index['runs'].append(entry)
        written.append(item)
    jsonbin._write_index(index)
    preview['applied'] = True
    preview['restoring'] = written
    return preview