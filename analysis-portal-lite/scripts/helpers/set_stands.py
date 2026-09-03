#!/usr/bin/env python3
"""
Inspect and correct the test stand recorded against stored runs.

    python3 scripts/helpers/set_stands.py                     # counts per stand
    python3 scripts/helpers/set_stands.py --list              # every sample, by stand
    python3 scripts/helpers/set_stands.py --from FCTS         # samples on one stand
    python3 scripts/helpers/set_stands.py --from Scribner --to "Scribner 1" --apply
    python3 scripts/helpers/set_stands.py --sample 260421_GSMA-Qual-1 --to "FCTS 2" --apply
    python3 scripts/helpers/set_stands.py --restore index-backup-....json --apply

Runs analysed before stands were numbered carry a bare family — 'Scribner' or
'FCTS' — derived from the file extensions, because that is all an extension can
tell you. Which numbered stand a run actually came from is not in the data; only
the operator knows. This tool applies that knowledge, it does not infer it.

Both the index entry and the detail bin are updated. Correcting the index alone
would not last: the next push for that sample rebuilds its index entry from the
detail record, and a record with no declared stand falls back to deriving the
bare family again, silently undoing the correction.

Read-only by default. --apply backs the index up first, to a file and to stdout.

On Railway:  railway ssh "python3 /app/scripts/helpers/set_stands.py"
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.helpers import jsonbin                              # noqa: E402
from scripts.helpers.record import STAND_OPTIONS, stand_family   # noqa: E402

OK, BAD, INFO = "  ok  ", " FAIL ", "      "

APPLY = '--apply' in sys.argv


def _arg(flag):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


FROM = _arg('--from')
TO = _arg('--to')
SAMPLE = _arg('--sample')
RESTORE = _arg('--restore') if '--restore' in sys.argv else None

# 'none' selects entries carrying no stand at all, which cannot be written as
# an empty command-line argument without ambiguity.
UNSET = ('none', 'None', '')


def head(t):
    print(f"\n{t}\n" + "-" * len(t))


def die(msg, hint=''):
    print(f"{BAD}{msg}")
    if hint:
        print(f"{INFO}{hint}")
    sys.exit(1)


def current(entry):
    return entry.get('stand') or None


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
    print(f"{INFO}Detail bins are not reverted — re-run the correction if needed.")
    sys.exit(0)


# ── 1. read and report ────────────────────────────────────────────
head("1. Stands currently recorded")

if not jsonbin.is_configured():
    die("JSONBin is not configured",
        "Run this in the same environment as the app, e.g. railway ssh.")
try:
    index = jsonbin.fetch_index()
except Exception as e:
    die(f"cannot read the index: {e}")

runs = index.get('runs', [])
counts = {}
for r in runs:
    counts[current(r) or '(none)'] = counts.get(current(r) or '(none)', 0) + 1

print(f"{OK}{len(runs)} entr(ies)")
for stand, n in sorted(counts.items()):
    numbered = stand in STAND_OPTIONS
    note = ('' if numbered else
            '  ← bare family; no stand number recorded'
            if stand != '(none)' else '  ← no stand recorded at all')
    print(f"{INFO}  {stand:14} {n:>3}{note}")

legacy = sum(n for s, n in counts.items()
             if s != '(none)' and s not in STAND_OPTIONS)
if not legacy and '(none)' not in counts:
    print(f"\n{INFO}Every entry already names a numbered stand. Nothing to correct.")


# ── 2. select ─────────────────────────────────────────────────────
head("2. Selection")

LIST = '--list' in sys.argv

# --from without --to is a question rather than a change: which samples carry
# this stand? Answering it here is what makes the correction workflow usable —
# see the group, decide, then re-run with --to.
if LIST or (not TO and (FROM or SAMPLE)):
    head("Samples by stand")
    groups = {}
    for r in runs:
        if SAMPLE and r.get('sample_name') != SAMPLE:
            continue
        key = current(r) or '(none)'
        if FROM is not None:
            if FROM in UNSET:
                if current(r) is not None:
                    continue
            elif current(r) != FROM:
                continue
        groups.setdefault(key, []).append(r)

    if not groups:
        print(f"{INFO}No entries match.")
        sys.exit(0)
    for stand in sorted(groups):
        names = sorted({r.get('sample_name', '?') for r in groups[stand]})
        print(f"\n{INFO}{stand}  ({len(names)} sample(s))")
        for n in names:
            print(f"    {n}")
    total = sum(len({r.get('sample_name') for r in v}) for v in groups.values())
    print(f"\n{INFO}{total} sample(s) listed. Add --to \"<stand>\" to change them.")
    sys.exit(0)

if not TO and not (FROM or SAMPLE):
    print(f"{INFO}Report only. To correct entries:")
    print(f'{INFO}  --from Scribner --to "Scribner 1" --apply')
    print(f'{INFO}  --sample 260421_GSMA-Qual-1 --to "FCTS 2" --apply')
    print(f'{INFO}  --from none --to "FCTS 1" --apply      (entries with no stand)')
    print(f"{INFO}")
    print(f"{INFO}To see which samples carry a stand before changing anything:")
    print(f"{INFO}  --list                 every sample, grouped by stand")
    print(f"{INFO}  --from FCTS            just the samples tagged FCTS")
    print(f"{INFO}")
    print(f"{INFO}Which numbered stand a legacy run came from is not in the data.")
    print(f"{INFO}This applies your knowledge; it does not infer anything.")
    sys.exit(0)

if not TO:
    die("--to is required", f'one of: {", ".join(STAND_OPTIONS)}')
if TO not in STAND_OPTIONS:
    die(f"--to {TO!r} is not a known stand",
        f'expected one of: {", ".join(STAND_OPTIONS)}')

selected = []
for r in runs:
    if SAMPLE and r.get('sample_name') != SAMPLE:
        continue
    if FROM is not None:
        if FROM in UNSET:
            if current(r) is not None:
                continue
        elif current(r) != FROM:
            continue
    if current(r) == TO:
        continue                       # already correct; nothing to do
    selected.append(r)

if not selected:
    print(f"{INFO}No entries match. Nothing to do.")
    sys.exit(0)

# A family change is almost always a mistake — Scribner files do not become
# FCTS files — so it is called out rather than performed quietly.
crossing = [r for r in selected
            if current(r) and stand_family(current(r)) != stand_family(TO)]

print(f"{OK}{len(selected)} entr(ies) would become {TO!r}:")
for r in selected[:25]:
    print(f"{INFO}  {str(current(r) or '(none)'):12} → {TO:12} "
          f"{r.get('sample_name', '?')}")
if len(selected) > 25:
    print(f"{INFO}  … and {len(selected) - 25} more")

if crossing:
    print(f"\n{BAD}{len(crossing)} of these would change FAMILY "
          f"(Scribner ↔ FCTS):")
    for r in crossing[:10]:
        print(f"{INFO}  {current(r)} → {TO}   {r.get('sample_name', '?')}")
    print(f"{INFO}The family is derived from the data format, so this usually")
    print(f"{INFO}means the wrong entries were selected. Narrow with --from or")
    print(f"{INFO}--sample if that was not intended.")


# ── 3. apply ──────────────────────────────────────────────────────
head("3. Apply" + ("" if APPLY else " (dry run)"))

if not APPLY:
    print(f"{INFO}Nothing was written. Re-run with --apply to make the change.")
    sys.exit(0)

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

# Detail bins first. If the index write then fails, the bins carry the new
# stand and the index still shows the old one — which the next push would
# reconcile anyway, and which re-running this fixes. The reverse order would
# leave a corrected index that the next push silently reverts.
updated_bins, failed = 0, []
for r in selected:
    bid = r.get('bin_id')
    if not bid:
        failed.append((r.get('sample_name'), 'no bin_id'))
        continue
    try:
        rec = jsonbin.fetch_detail_bin(bid)
        rec['stand'] = TO
        jsonbin.update_detail_bin(bid, rec)
        updated_bins += 1
    except Exception as e:
        failed.append((r.get('sample_name'), str(e)[:70]))

print(f"{OK}{updated_bins} detail bin(s) updated")
for name, why in failed:
    print(f"{BAD}  {name}: {why}")
if failed:
    die("some detail bins could not be updated — index left unchanged",
        "Fix the cause and re-run; nothing has been lost.")

by_bin = {r.get('bin_id') for r in selected}
for entry in index['runs']:
    if entry.get('bin_id') in by_bin:
        entry['stand'] = TO
try:
    jsonbin._write_index(index)
    print(f"{OK}index updated — {len(selected)} entr(ies) now {TO!r}")
except Exception as e:
    die(f"index write failed: {e}",
        f"Detail bins already carry {TO!r}; re-run to finish, or restore with "
        f"--restore {backup_path} --apply")

head("Done")
print(f"{len(selected)} entr(ies) now record {TO!r}.")
print("The View tab's stand filter will show them under that stand; entries")
print("still carrying a bare family continue to appear under every stand in it.")