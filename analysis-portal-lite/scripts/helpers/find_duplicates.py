#!/usr/bin/env python3
"""
Find samples that are the same measurement under different names.

    python3 scripts/helpers/find_duplicates.py                 # report only
    python3 scripts/helpers/find_duplicates.py --apply --group 1 --keep-name <name>
    python3 scripts/helpers/find_duplicates.py --restore index-backup-....json --apply

One group per invocation. Each group is a different physical sample and needs
its own name decision, so there is no way to merge several at once — a single
--keep-name applied across groups would give distinct cells the same identity,
and sample-keyed merging would then collapse them into one bin.

Read-only by default. The report lists every candidate pair, what matched, and
for near-misses the one field that disagreed — seeing the near-misses is how the
rule earns trust before it is allowed to merge anything.

Matching is exact over the overlapping (analysis, step) units. It detects the
same measurement analysed twice, which covers a re-run, a partial-then-full
analysis, and the same folder processed under a mistyped name. It does not
detect the same physical cell measured twice, and it deliberately misses the
same data analysed with different parameters — see DUPLICATE_DETECTION_ANALYSIS.

`--apply` requires an explicit `--keep-name`. Names are never resolved
automatically: recency is not evidence of correctness, and if the newer name is
the typo, auto-resolution propagates the error and discards the identity that
was right.

On Railway:  railway ssh "python3 /app/scripts/helpers/find_duplicates.py"
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.helpers import dupdetect, jsonbin                    # noqa: E402
from scripts.helpers.record import (                              # noqa: E402
    merge_detail_record, merge_index_entry, parse_run_date,
)

OK, BAD, INFO = "  ok  ", " FAIL ", "      "
APPLY = '--apply' in sys.argv
KEEP_NAME = None
if '--keep-name' in sys.argv:
    i = sys.argv.index('--keep-name')
    if i + 1 < len(sys.argv):
        KEEP_NAME = sys.argv[i + 1]
RESTORE = (next((a for a in sys.argv[1:] if a.endswith('.json')), None)
           if '--restore' in sys.argv else None)
GROUP = None
if '--group' in sys.argv:
    i = sys.argv.index('--group')
    if i + 1 < len(sys.argv):
        try:
            GROUP = int(sys.argv[i + 1])
        except ValueError:
            GROUP = -1

# Samples whose mutual matching would falsify the premise this rests on:
# qualification replicates are built to be identical, so if two *different*
# replicates match exactly the assumption is wrong and nothing should merge.
REPLICATE_HINT = ('Qual', 'qual')


def normalised(name):
    """Sample name reduced to alphanumerics, lowercased.

    Used only by the replicate warning, never by matching. Two names differing
    only in punctuation — '260421_GSMA-Qual-1' and '260421_GSMA_Qual1' — are
    the same sample typed two ways, not two different replicates, and flagging
    them would make the warning fire on exactly the case being fixed.
    """
    return ''.join(c for c in str(name).lower() if c.isalnum())


def head(t):
    print(f"\n{t}\n" + "-" * len(t))


def die(msg, hint=''):
    print(f"{BAD}{msg}")
    if hint:
        print(f"{INFO}{hint}")
    sys.exit(1)


def label(entry):
    return (f"{entry.get('sample_name', '?')} "
            f"[{entry.get('run_date') or str(entry.get('timestamp', ''))[:10]}]")


# ── restore ───────────────────────────────────────────────────────
if RESTORE:
    head("Restore")
    try:
        backup = json.loads(Path(RESTORE).read_text())
    except Exception as e:
        die(f"cannot read {RESTORE}: {e}")
    if not isinstance(backup.get('runs'), list):
        die(f"{RESTORE} is not an index backup (no 'runs' list)")
    print(f"{INFO}{RESTORE} holds {len(backup['runs'])} entr(ies)")
    if not APPLY:
        print(f"{INFO}Dry run — add --apply to write it back.")
        sys.exit(0)
    try:
        jsonbin._write_index(backup)
    except Exception as e:
        die(f"restore failed: {e}")
    print(f"{OK}index restored")
    sys.exit(0)


# ── 1. read ───────────────────────────────────────────────────────
head("1. Index")

if not jsonbin.is_configured():
    die("JSONBin is not configured",
        "Run this in the same environment as the app, e.g. railway ssh.")
try:
    index = jsonbin.fetch_index()
except Exception as e:
    die(f"cannot read the index: {e}")

runs = index.get('runs', [])
print(f"{OK}{len(runs)} entr(ies), "
      f"{len({r.get('sample_name') for r in runs})} distinct sample name(s)")

candidates = dupdetect.index_prefilter(index)
print(f"{OK}{len(candidates)} candidate pair(s) share an index unit")
if not candidates:
    print(f"{INFO}No two differently-named samples report the same key values.")
    print(f"{INFO}Nothing to review.")
    sys.exit(0)


# ── 2. confirm at full precision ──────────────────────────────────
head("2. Confirmation against the detail bins")

_cache = {}


def detail_of(entry):
    bid = entry.get('bin_id')
    if bid not in _cache:
        _cache[bid] = jsonbin.fetch_detail_bin(bid)
    return _cache[bid]


matches, near = [], []
for a, b in candidates:
    try:
        ra, rb = detail_of(a), detail_of(b)
    except Exception as e:
        print(f"{BAD}{label(a)} vs {label(b)}: cannot read a bin — {e}")
        continue
    res = dupdetect.compare_records(ra, rb)
    (matches if res.is_duplicate else near).append((a, b, res))

print(f"{INFO}{len(matches)} confirmed, {len(near)} rejected on closer look")

if near:
    print(f"\n{INFO}Rejected — shown because near-misses are the evidence that")
    print(f"{INFO}the rule discriminates rather than merging anything similar:")
    for a, b, res in near:
        print(f"{INFO}  {label(a)} vs {label(b)}")
        print(f"{INFO}    {res.describe()}")

if matches:
    print(f"\n{INFO}Confirmed duplicates:")
    weak = 0
    for a, b, res in matches:
        print(f"{OK}{label(a)}  ==  {label(b)}")
        print(f"{INFO}    {res.describe()}")
        print(f"{INFO}    bins {a.get('bin_id')} / {b.get('bin_id')}")
        if getattr(res, 'source', 'summary') != 'summary':
            weak += 1
    if weak:
        print(f"\n{INFO}{weak} match(es) used plot-annotation values rather than")
        print(f"{INFO}the tier-1 summary, because one side predates it. Those")
        print(f"{INFO}values are rounded for display, so the evidence is weaker")
        print(f"{INFO}than a summary match — worth eyeballing before merging.")


# ── 3. the premise check ──────────────────────────────────────────
head("3. Replicate check")

reps = [r for r in runs
        if any(h in str(r.get('sample_name', '')) for h in REPLICATE_HINT)]
rep_names = sorted({r.get('sample_name') for r in reps})
if len(rep_names) < 2:
    print(f"{INFO}Fewer than two replicate-style sample names present; "
          f"nothing to check.")
else:
    hits = [(a, b) for a, b, _ in matches
            if a.get('sample_name') in rep_names
            and b.get('sample_name') in rep_names
            and normalised(a.get('sample_name'))
            != normalised(b.get('sample_name'))]
    print(f"{INFO}Replicate-style names: {', '.join(rep_names)}")
    variants = {}
    for n in rep_names:
        variants.setdefault(normalised(n), []).append(n)
    for group in variants.values():
        if len(group) > 1:
            print(f"{INFO}  {' / '.join(group)} — same name, different "
                  f"punctuation; treated as one replicate")
    if hits:
        print(f"{BAD}{len(hits)} pair(s) of DIFFERENT replicate samples matched "
              f"exactly.")
        print(f"{INFO}This should not happen — replicates are different cells "
              f"and cannot produce identical doubles.")
        print(f"{INFO}Either they genuinely share source data, or the premise "
              f"behind this rule is wrong. Do not merge until it is understood.")
    else:
        print(f"{OK}No two different replicate samples matched — as expected.")


# ── 4. apply ──────────────────────────────────────────────────────
head("4. Merge" + ("" if APPLY else " (dry run)"))

if not matches:
    print(f"{INFO}Nothing confirmed; nothing to merge.")
    sys.exit(0)

groups = dupdetect.group_matches(
    [(a.get('bin_id'), b.get('bin_id')) for a, b, _ in matches])
by_bin = {r.get('bin_id'): r for r in runs}

print(f"{INFO}{len(groups)} group(s) would be consolidated:")
for n, g in enumerate(groups, start=1):
    names = [by_bin[b].get('sample_name') for b in g if b in by_bin]
    print(f"{INFO}  [{n}] {' + '.join(names)}  ({len(g)} entries → 1)")
    for nm in sorted(set(names)):
        # A mistyped prefix usually fails the YYMMDD date parse, so this is a
        # useful hint — but only a hint. The choice stays with the operator.
        ok_date = parse_run_date(nm) is not None
        print(f"{INFO}        {nm:32} "
              f"{'run_date parses' if ok_date else 'no valid run_date prefix'}")

if not APPLY:
    print(f"\n{INFO}Nothing was written. To merge one group:")
    print(f"{INFO}  --apply --group <n> --keep-name \"<the correct sample name>\"")
    print(f"{INFO}")
    print(f"{INFO}One group per run. Each group is a different sample and needs")
    print(f"{INFO}its own name; one --keep-name across groups would give distinct")
    print(f"{INFO}cells the same identity.")
    print(f"{INFO}The name is not chosen for you: recency is not evidence of")
    print(f"{INFO}correctness, and merging under a typo would discard the right")
    print(f"{INFO}name along with the record that held it.")
    sys.exit(0)

if not KEEP_NAME:
    die("--apply requires --keep-name",
        'e.g. --apply --group 1 --keep-name "260421_GSMA-Qual-1"')

if GROUP is None:
    if len(groups) > 1:
        die(f"{len(groups)} groups found — --apply needs --group <n>",
            "Each group is a different sample and takes its own --keep-name. "
            "Applying one name across groups would give distinct cells the "
            "same identity, and sample-keyed merging would then collapse them.")
    GROUP = 1
if not (1 <= GROUP <= len(groups)):
    die(f"--group {GROUP} is out of range (1..{len(groups)})")

selected = groups[GROUP - 1]
sel_names = sorted({by_bin[b].get('sample_name') for b in selected if b in by_bin})
print(f"{INFO}Merging group [{GROUP}]: {' + '.join(sel_names)}")
if KEEP_NAME not in sel_names:
    print(f"{BAD}--keep-name {KEEP_NAME!r} is not one of this group's names.")
    print(f"{INFO}Names in group [{GROUP}]: {', '.join(sel_names)}")
    print(f"{INFO}Renaming to something new is possible but is almost always a")
    print(f"{INFO}typo in the command; re-run with one of the names above, or")
    print(f"{INFO}rename deliberately afterwards.")
    sys.exit(1)
groups = [selected]

stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
backup_path = Path(f'index-backup-{stamp}.json')
try:
    backup_path.write_text(json.dumps(index, ensure_ascii=False, indent=2))
    print(f"{OK}index backed up to {backup_path.resolve()}")
except Exception as e:
    print(f"{BAD}could not write a backup file: {e}")
print(f"{INFO}Backup also printed below, so it survives an ephemeral filesystem.")
print("----- BEGIN INDEX BACKUP -----")
print(json.dumps(index, ensure_ascii=False))
print("----- END INDEX BACKUP -----")

superseded = []
for g in groups:
    ordered = sorted(g, key=lambda b: str(by_bin[b].get('run_date')
                                          or by_bin[b].get('timestamp', '')))
    survivor_bin = ordered[-1]
    merged_detail, merged_entry = None, None
    for bid in ordered:
        merged_detail = merge_detail_record(merged_detail, detail_of(by_bin[bid]))
        merged_entry = merge_index_entry(merged_entry, by_bin[bid])
    merged_detail['sample_name'] = KEEP_NAME
    merged_entry['sample_name'] = KEEP_NAME
    merged_entry['bin_id'] = survivor_bin

    # Detail bin first: if the index write then fails, the bin holds merged
    # data and the index still describes the old layout, which is recoverable.
    try:
        jsonbin.update_detail_bin(survivor_bin, merged_detail)
        print(f"{OK}merged into bin {survivor_bin} as {KEEP_NAME!r}")
    except Exception as e:
        die(f"writing bin {survivor_bin} failed: {e}",
            "The index has not been touched. Nothing is lost; re-run.")

    superseded.extend(b for b in ordered if b != survivor_bin)
    kept = [r for r in index['runs'] if r.get('bin_id') not in g]
    kept.append(merged_entry)
    index['runs'] = kept

try:
    jsonbin._write_index(index)
    print(f"{OK}index rewritten — {len(index['runs'])} entr(ies)")
except Exception as e:
    die(f"index write failed: {e}",
        f"Detail bins were merged. Restore with "
        f"--restore {backup_path} --apply")

head("Done")
print(f"Merged under the name {KEEP_NAME!r}.")
if superseded:
    print("Superseded bins are still present and no longer referenced:")
    for b in superseded:
        print(f"{INFO}  {b}")
    print("\nDelete them from the JSONBin dashboard once the result looks right.")
