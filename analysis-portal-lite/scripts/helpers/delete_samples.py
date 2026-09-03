#!/usr/bin/env python3
"""
List stored samples, and remove them.

    python3 scripts/helpers/delete_samples.py                      # list every sample
    python3 scripts/helpers/delete_samples.py --stand "FCTS 1"     # list one stand
    python3 scripts/helpers/delete_samples.py --sample A --sample B          # preview
    python3 scripts/helpers/delete_samples.py --sample A --apply             # unlist
    python3 scripts/helpers/delete_samples.py --sample A --apply --purge 1   # destroy
    python3 scripts/helpers/delete_samples.py --restore index-backup-....json --apply

Removal happens in two steps, deliberately separated:

  --apply           removes the index entry. The sample disappears from the
                    portal, the detail bin survives untouched as an orphan, and
                    the backup written here is enough to put it back.

  --apply --purge N deletes the detail bins too. JSONBin has no undelete, so
                    the measurements are gone. N must equal the number of bins
                    about to be destroyed, which means the count has to have
                    been read before it can be typed.

Samples are named explicitly and matched exactly. There is no pattern matching
and no "delete everything on this stand" — the whole point of the friction is
that a deletion cannot be produced by a slightly wrong argument.

On Railway:  railway ssh "python3 /app/scripts/helpers/delete_samples.py"
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.helpers import jsonbin                       # noqa: E402
from scripts.helpers.record import stand_matches          # noqa: E402

OK, BAD, INFO = "  ok  ", " FAIL ", "      "

APPLY = '--apply' in sys.argv


def _args(flag):
    """Every value given for a repeatable flag."""
    out = []
    for i, a in enumerate(sys.argv):
        if a == flag and i + 1 < len(sys.argv):
            out.append(sys.argv[i + 1])
    return out


def _arg(flag):
    vals = _args(flag)
    return vals[0] if vals else None


SAMPLES = _args('--sample')
STAND = _arg('--stand')
RESTORE = _arg('--restore') if '--restore' in sys.argv else None
PURGE = _arg('--purge') if '--purge' in sys.argv else None


def head(t):
    print(f"\n{t}\n" + "-" * len(t))


def die(msg, hint=''):
    print(f"{BAD}{msg}")
    if hint:
        print(f"{INFO}{hint}")
    sys.exit(1)


def describe(entry):
    return (f"{entry.get('sample_name', '?')}  "
            f"[{entry.get('run_date') or str(entry.get('timestamp', ''))[:10]}]  "
            f"{entry.get('stand') or 'no stand'}  bin={entry.get('bin_id')}")


# ── restore ───────────────────────────────────────────────────────
if RESTORE:
    head("Restore")
    try:
        backup = json.loads(Path(RESTORE).read_text())
    except Exception as e:
        die(f"cannot read {RESTORE}: {e}")
    index_backup = backup.get('index', backup)
    if not isinstance(index_backup.get('runs'), list):
        die(f"{RESTORE} does not contain an index (no 'runs' list)")
    print(f"{INFO}{RESTORE} holds {len(index_backup['runs'])} index entr(ies)")
    if backup.get('purged'):
        print(f"{BAD}This backup records {len(backup['purged'])} PURGED bin(s).")
        print(f"{INFO}Restoring the index will point those entries at bins that")
        print(f"{INFO}no longer exist. Their measurements cannot be recovered;")
        print(f"{INFO}only the detail records saved in this file remain, under")
        print(f"{INFO}the 'purged' key, and re-uploading them is a manual step.")
    if not APPLY:
        print(f"{INFO}Dry run — add --apply to write it back.")
        sys.exit(0)
    try:
        jsonbin._write_index(index_backup)
    except Exception as e:
        die(f"restore failed: {e}")
    print(f"{OK}index restored")
    sys.exit(0)


# ── 1. read ───────────────────────────────────────────────────────
head("1. Stored samples")

if not jsonbin.is_configured():
    die("JSONBin is not configured",
        "Run this in the same environment as the app, e.g. railway ssh.")
try:
    index = jsonbin.fetch_index()
except Exception as e:
    die(f"cannot read the index: {e}")

runs = index.get('runs', [])
listed = [r for r in runs
          if not STAND or stand_matches(r.get('stand'), STAND)]

if not SAMPLES:
    scope = f" on {STAND}" if STAND else ""
    print(f"{OK}{len(listed)} entr(ies){scope}, "
          f"{len({r.get('sample_name') for r in listed})} distinct sample name(s)")
    for r in sorted(listed, key=lambda e: str(e.get('run_date')
                                              or e.get('timestamp', ''))):
        print(f"{INFO}  {describe(r)}")
    print(f"\n{INFO}To remove one, name it exactly:")
    print(f'{INFO}  --sample "<name>" --apply            unlist it, keep the data')
    print(f'{INFO}  --sample "<name>" --apply --purge 1  delete the data too')
    sys.exit(0)


# ── 2. select ─────────────────────────────────────────────────────
head("2. Selection")

selected, unknown = [], []
for name in SAMPLES:
    hits = [r for r in runs if r.get('sample_name') == name]
    if hits:
        selected.extend(hits)
    else:
        unknown.append(name)

if unknown:
    # A name that matches nothing is far more likely to be a typo than a
    # sample that has already gone, and acting on the rest of the list would
    # delete things the operator did not check.
    die(f"no sample named: {', '.join(repr(n) for n in unknown)}",
        "Run without --sample to list the names exactly as stored. "
        "Nothing was changed.")

print(f"{OK}{len(selected)} entr(ies) selected:")
for r in selected:
    print(f"{INFO}  {describe(r)}")

bins = [r.get('bin_id') for r in selected if r.get('bin_id')]
print(f"\n{INFO}{len(bins)} detail bin(s) behind them.")


# ── 3. apply ──────────────────────────────────────────────────────
head("3. Remove" + ("" if APPLY else " (dry run)"))

if not APPLY:
    print(f"{INFO}Nothing was written.")
    print(f"{INFO}  --apply              remove from the index; bins survive as")
    print(f"{INFO}                       orphans and the backup can restore them")
    print(f"{INFO}  --apply --purge {len(bins):<4} also delete the {len(bins)} bin(s) — "
          f"permanent")
    sys.exit(0)

purge = False
if PURGE is not None:
    try:
        want = int(PURGE)
    except ValueError:
        die(f"--purge takes the number of bins to delete, not {PURGE!r}",
            f"here that number is {len(bins)}")
    if want != len(bins):
        die(f"--purge {want} does not match the {len(bins)} bin(s) selected",
            "The count has to match so that a deletion cannot come from a "
            "half-read command. Nothing was changed.")
    purge = True

# The backup carries the full detail records when purging, because the index
# entry alone cannot bring a sample back — its measurements live in the bin.
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
backup = {'index': index, 'removed': selected, 'purged': []}

if purge:
    print(f"{INFO}Fetching detail records before deleting them…")
    for r in selected:
        bid = r.get('bin_id')
        if not bid:
            continue
        try:
            backup['purged'].append({'bin_id': bid,
                                     'record': jsonbin.fetch_detail_bin(bid)})
        except Exception as e:
            die(f"could not read bin {bid} before deleting it: {e}",
                "Nothing was changed. A bin that cannot be read cannot be "
                "backed up, and deleting it would be unrecoverable.")

backup_path = Path(f'index-backup-{stamp}.json')
payload = json.dumps(backup, ensure_ascii=False)
try:
    backup_path.write_text(json.dumps(backup, ensure_ascii=False, indent=2))
    print(f"{OK}backup written to {backup_path.resolve()} ({len(payload):,} B)")
except Exception as e:
    die(f"could not write the backup: {e}",
        "Nothing was changed. Removal without a backup is not offered.")

if purge:
    print(f"{INFO}The backup holds the full detail records. Railway's filesystem")
    print(f"{INFO}is ephemeral — copy it somewhere durable before redeploying.")
else:
    # Small enough to survive in the deploy log, which the purge backup is not.
    print("----- BEGIN INDEX BACKUP -----")
    print(json.dumps(index, ensure_ascii=False))
    print("----- END INDEX BACKUP -----")

# Index first when purging: an entry pointing at a deleted bin is a broken
# reference the portal would fail on, whereas a bin with no entry is an orphan,
# which is already an accepted state.
drop = {id(r) for r in selected}
index['runs'] = [r for r in runs if id(r) not in drop]
try:
    jsonbin._write_index(index)
    print(f"{OK}index updated — {len(index['runs'])} entr(ies) remain")
except Exception as e:
    die(f"index write failed: {e}", "No bins were deleted. Nothing was lost.")

if not purge:
    head("Done")
    print(f"{len(selected)} entr(ies) removed from the index.")
    print("Their detail bins are untouched and now orphaned. Re-run with")
    print(f"--restore {backup_path.name} --apply to put the entries back.")
    sys.exit(0)

deleted, failed = 0, []
for item in backup['purged']:
    try:
        jsonbin.delete_detail_bin(item['bin_id'])
        deleted += 1
    except Exception as e:
        failed.append((item['bin_id'], str(e)[:70]))

print(f"{OK}{deleted} detail bin(s) deleted")
for bid, why in failed:
    print(f"{BAD}  {bid}: {why} — delete it from the dashboard")

head("Done")
print(f"{len(selected)} sample entr(ies) removed and {deleted} bin(s) destroyed.")
print(f"The only remaining copy of that data is {backup_path.name}.")
