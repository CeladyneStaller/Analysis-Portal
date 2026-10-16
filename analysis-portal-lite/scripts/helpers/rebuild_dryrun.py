#!/usr/bin/env python3
"""
Rehearse the electrolyzer rebuild offline, on a backup of the store.

    python3 scripts/helpers/rebuild_dryrun.py portal-backup-manual-....json
    python3 scripts/helpers/rebuild_dryrun.py backup.json --json report.json

The backup is the file the admin modal's ⭳ Backup button saves: the index and
every detail record. This runs the admin "Rebuild electrolyzer entries"
operation against an in-memory copy of it — the preview, then the rebuild
applied to the copy, then the preview again — and prints, per sample, the
electrolyzer units before and after with their values.

Nothing here can reach JSONBin. The one function that makes HTTP calls is
replaced before anything runs, and the JSONBin settings are overwritten in
this process, so a shell that has the real key exported changes nothing.

Exit status: 0 when every electrolyzer record was checked and the second
preview finds nothing left to rebuild; 1 when it does find something (the
rebuild would not be idempotent) or a record could not be rebuilt at all; 2
when the file is unusable; 3 when the rebuild is idempotent but some records
were missing from the backup, so the check is incomplete. Entries that are
skipped for what they are — no bin, a bin shared with another entry — are
reported and leave the status alone.
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Overwritten, not defaulted: whatever is exported in this shell is ignored.
os.environ.update(JSONBIN_API_KEY='offline', JSONBIN_COLLECTION_ID='offline',
                  JSONBIN_INDEX_BIN_ID='OFFLINE-INDEX')

from scripts.helpers import admin, jsonbin                     # noqa: E402
from scripts.helpers.record import (                         # noqa: E402
    PRE_ELX_TYPES, decode_sidecars, plot_bucket,
)
from scripts.helpers import registry                           # noqa: E402

INDEX_ID = 'OFFLINE-INDEX'


class OfflineStore:
    """JSONBin in memory, seeded from a backup. Writes stay in memory."""

    def __init__(self, backup):
        self.bins = {INDEX_ID: copy.deepcopy(backup['index'])}
        for p in backup.get('purged') or []:
            if isinstance(p, dict) and p.get('bin_id') and isinstance(p.get('record'), dict):
                self.bins[str(p['bin_id'])] = copy.deepcopy(p['record'])
        self.writes = []

    def __call__(self, url, method='GET', body=None, extra_headers=None):
        base = jsonbin._JSONBIN_BASE
        if not url.startswith(base):
            raise RuntimeError(f'offline: refusing a request outside JSONBin: {url}')
        path = url[len(base):].strip('/').split('/')
        if method == 'GET' and len(path) == 2 and path[1] == 'latest':
            if path[0] not in self.bins:
                raise RuntimeError(f'offline: bin {path[0]} is not in the backup')
            return {'record': copy.deepcopy(self.bins[path[0]])}
        if method == 'PUT' and len(path) == 1:
            self.bins[path[0]] = json.loads(json.dumps(jsonbin.json_safe(body),
                                                       allow_nan=False))
            self.writes.append(path[0])
            return {}
        raise RuntimeError(f'offline: {method} {url} is not something a rebuild does')


def _fmt_units(units, indent):
    pad = ' ' * indent
    if not units:
        return pad + 'none'
    lines, bare = [], []
    for u in units:
        if not u['values']:
            bare.append(u['analysis'] + (f" · {u['step']}" if u['step'] else ''))
            continue
        vals = ', '.join(f'{k} {v:.4g}' if isinstance(v, (int, float)) else f'{k} {v}'
                         for k, v in u['values'].items())
        lines.append(f"{pad}{u['analysis']} · {u['step'] or 'no cell'}   {vals}")
    if bare:
        lines.append(f"{pad}{len(bare)} without values: {', '.join(bare)}")
    return '\n'.join(lines)


def _notes(r):
    out = []
    if r['gained']:
        out.append(f"+{len(r['gained'])} value(s): "
                   + ', '.join(f"{g['analysis']} {g['name']}" for g in r['gained']))
    detail = [x['name'] for x in r['removed'] if x['detail_only']]
    lost = [f"{x['analysis']} {x['name']}" for x in r['removed'] if not x['detail_only']]
    if detail:
        out.append('now in run detail only: ' + ', '.join(detail))
    if lost:
        out.append('NO LONGER IN THE INDEX: ' + ', '.join(lost))
    if r['stand_cleared']:
        out.append(f"guessed stand {r['stand_cleared']} dropped")
    if r['rows_collapsed']:
        out.append(f"{r['rows_collapsed']} earlier summary row(s) collapsed to the latest"
                   + (" — possibly the overwritten cell's; the backup keeps them"
                      if r['cells_missing'] else ''))
    if r['rows_unattributed']:
        out.append(f"WARNING {r['rows_unattributed']} summary row(s) name no cell and "
                   f"reach no unit (older runs' cells: {r['old_cells'] or 'none readable'})")
    if r['plots_without_cell']:
        out.append(f"WARNING {len(r['plots_without_cell'])} plot(s) stored without "
                   f"plot data could not be given a cell")
    if r['cells_missing']:
        out.append(f"check: uploads also name {', '.join(r['cells_missing'])}, which has "
                   f"no plots stored — if it was analysed here, re-run it")
    if r['pre_elx_plots']:
        out.append(f"{r['pre_elx_plots']} plot(s) from before Jul 27 stay filed as "
                   f"fuel-cell plots")
    return out


def _missed(backup, candidates):
    """Records holding electrolyzer plots whose index entry the scan skipped.

    The scan picks entries from the index alone; this reads every record in
    the backup to check that nothing it should have read was passed over.
    """
    records = {str(p.get('bin_id')): p.get('record')
               for p in backup.get('purged') or [] if isinstance(p, dict)}
    out = []
    for e in (backup.get('index') or {}).get('runs') or []:
        if admin._key(e) in candidates:
            continue
        rec = records.get(str(e.get('bin_id')))
        if not isinstance(rec, dict):
            continue
        types = {str(sc.get('plot_type') or '') for sc in decode_sidecars(rec).values()
                 if isinstance(sc, dict)}
        if (any(registry.family(plot_bucket(b)) == 'elx' for b in rec.get('metrics') or {})
                or types & PRE_ELX_TYPES):
            out.append(e.get('sample_name', ''))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('backup', help='a backup file saved from the admin modal')
    ap.add_argument('--json', metavar='PATH', help='also write the full reports here')
    args = ap.parse_args(argv)

    try:
        backup = json.loads(Path(args.backup).read_text(encoding='utf-8'))
    except Exception as e:
        print(f'could not read {args.backup}: {e}')
        return 2
    if (not isinstance(backup, dict) or backup.get('kind') != 'analysis-portal-backup'
            or not isinstance((backup.get('index') or {}).get('runs'), list)):
        print('that is not a backup the portal wrote (no index in it)')
        return 2

    store = OfflineStore(backup)
    jsonbin._request = store
    runs = backup['index']['runs']
    held = sum(1 for e in runs if str(e.get('bin_id')) in store.bins)
    print(f"Backup written {backup.get('written', '?')}: {len(runs)} index entries, "
          f"{held} with their record.")
    absent = [e.get('sample_name', '') for e in runs
              if e.get('bin_id') and str(e.get('bin_id')) not in store.bins]
    if absent:
        print(f"  {len(absent)} entr(ies) have no record in the backup and cannot be "
              f"checked: {', '.join(absent[:20])}{' …' if len(absent) > 20 else ''}")

    preview = admin.rebuild_electrolyzer(apply=False)
    candidates = {admin._key(e) for e in runs if admin._may_be_electrolyzer(e)}
    print(f"\nElectrolyzer samples: {preview['scanned']} scanned — {preview['total']} to "
          f"rebuild, {preview['current']} already current, "
          f"{len(preview['left_as_stored'])} left as stored, "
          f"{len(preview['unreadable'])} unreadable.")

    for r in preview['entries']:
        print(f"\n{r['sample_name']}   (bin {r['bin_id']})")
        print('  now:')
        print(_fmt_units(r['units_before'], 4))
        print('  after:')
        print(_fmt_units(r['units_after'], 4))
        for n in _notes(r):
            print('  - ' + n)
    for x in preview['left_as_stored']:
        print(f"\n{x['sample_name']}: left as stored — {x['plots']} plot(s) from before "
              f"Jul 27 filed as fuel-cell plots; re-run it to bring it up to date")
    for x in preview['too_large']:
        print(f"\n{x['sample_name']}: left as stored — re-filed it would be "
              f"{x['body_bytes']:,} B, over the size limit")
    for x in preview['unreadable']:
        print(f"\n{x['sample_name']}: not checked — {x['error']}")

    missed = _missed(backup, candidates)
    if missed:
        print(f"\nWARNING {len(missed)} record(s) hold electrolyzer plots but the scan, "
              f"which picks samples from the index, would pass them over: "
              f"{', '.join(missed)}")

    prepared = admin.rebuild_electrolyzer(prepare=True)
    applied = (admin.rebuild_electrolyzer(apply=True, keys=prepared['keys'],
                                          token=prepared['token'])
               if prepared['keys'] else {'rebuilt': 0, 'failed': []})
    again = admin.rebuild_electrolyzer()
    print(f"\nApplied to the in-memory copy: {applied.get('rebuilt', 0)} entr(ies) "
          f"rebuilt, {len(applied.get('failed') or [])} failed"
          + (f" ({applied['error']})" if applied.get('error') else '') + '.')
    print(f"Second scan: {again['total']} left to rebuild"
          + (' — the rebuild is idempotent.' if not again['total'] else
             ' — NOT idempotent: ' + ', '.join(r['sample_name'] for r in again['entries'])))

    if args.json:
        Path(args.json).write_text(json.dumps(
            {'preview': preview, 'applied': applied, 'second_preview': again,
             'missed': missed, 'not_in_backup': absent}, indent=1, ensure_ascii=False),
            encoding='utf-8')
        print(f'Full reports written to {args.json}')
    broken = [x for x in preview['unreadable'] if x['error'].startswith('could not rebuild')]
    if again['total'] or broken:
        return 1
    if any(x['error'].startswith('could not read') for x in preview['unreadable']):
        print('INCOMPLETE: the records above that were not checked need a fuller backup.')
        return 3
    return 0


if __name__ == '__main__':
    sys.exit(main())
