"""Rebuilding electrolyzer entries — admin.rebuild_electrolyzer on old stores.

The store it runs against (fixtures/elx_eras_store.json) is not hand-made:
each run in it was pushed by the portal code of the era it stands for, through
that code's own main._run_job and _on_job_done, into an in-memory JSONBin —

  E0  0ff614b  Jul 27 morning: plot types not yet elx_*, nothing declared
  E1  3394d7b  Jul 27 evening to Aug 4: elx buckets, the cell declared
  E2  2902d1d  Aug 4 to phase 1: one bucket per plot type, no values
  P1  b9cbe97  phase 1

— one sample per scenario (`scenarios` in the file). Sidecars are reduced to
their plot type and declared metadata, which is all the rebuild reads of one.

The gate for phase 2: a dry run reports the expected units for every era,
for two cells, for a run that named no cell; applying it writes only what it
reports, leaves fuel-cell entries alone, and a second run changes nothing.

Run:  python3 scripts/helpers/test_elx_rebuild.py
"""

import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.update(JSONBIN_API_KEY='k', JSONBIN_COLLECTION_ID='C',
                  JSONBIN_INDEX_BIN_ID='INDEX')

from scripts.helpers import admin, jsonbin                         # noqa: E402
from scripts.helpers.record import (                               # noqa: E402
    attach_sidecars, decode_sidecars, normalise_electrolyzer, stored_plot_keys,
)

passed = failed = 0


def check(name, got, want):
    global passed, failed
    if got == want:
        passed += 1
    else:
        failed += 1
        print(f'  FAIL {name}\n       got  {got!r}\n       want {want!r}')


def close(name, got, want, tol=1e-5):
    global passed, failed
    if isinstance(got, (int, float)) and isinstance(want, (int, float)) \
            and abs(got - want) <= tol * max(1.0, abs(want)):
        passed += 1
    else:
        failed += 1
        print(f'  FAIL {name}: got {got!r}, want {want!r}')


def true(name, cond, detail=''):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        print(f'  FAIL {name} {detail}')


FIX = json.loads((Path(__file__).resolve().parent / 'fixtures'
                  / 'elx_eras_store.json').read_text(encoding='utf-8'))
SCEN = FIX['scenarios']


def stored_record(bin_id):
    rec = copy.deepcopy(FIX['records'][bin_id])
    attach_sidecars(rec, rec.pop('plots'))
    return rec


def plain(rec):
    """A record with its plot data decoded. gzip stamps the time it ran, so
    one record encoded twice can differ in bytes; its content cannot."""
    out = {k: v for k, v in rec.items() if k != 'sidecars'}
    out['sidecars'] = decode_sidecars(rec)
    return out


class Store:
    """JSONBin in memory. Logs calls; `fail` makes chosen calls raise."""

    def __init__(self):
        self.bins = {'INDEX': copy.deepcopy(FIX['index'])}
        for bid in FIX['records']:
            self.bins[bid] = stored_record(bid)
        self.calls, self.fail = [], {}

    def __call__(self, url, method='GET', body=None, extra_headers=None):
        parts = url.rstrip('/').split('/')
        bid = parts[-2] if parts[-1] == 'latest' else parts[-1]
        self.calls.append((method, bid))
        if (method, bid) in self.fail:
            raise RuntimeError(self.fail[(method, bid)])
        if method == 'PUT':
            self.bins[bid] = json.loads(json.dumps(jsonbin.json_safe(body),
                                                   allow_nan=False))
            return {}
        if method != 'GET':
            raise RuntimeError(f'unexpected {method}')
        if bid not in self.bins:
            raise RuntimeError(f'404 {bid}')
        return {'record': copy.deepcopy(self.bins[bid])}

    def entry(self, sample):
        return next(e for e in self.bins['INDEX']['runs']
                    if e['sample_name'] == sample)

    def puts(self):
        return [b for m, b in self.calls if m == 'PUT']


def install():
    s = Store()
    jsonbin._request = s
    return s


def rebuild():
    """Prepare, then apply with what prepare handed back — as the pane does."""
    prep = admin.rebuild_electrolyzer(prepare=True)
    if not prep['keys']:
        return prep, None
    return prep, admin.rebuild_electrolyzer(apply=True, keys=prep['keys'],
                                            token=prep['token'])


def units(entry):
    """(analysis, step) -> key_values, electrolyzer units only."""
    return {(u['Analysis'], u['step']): u.get('key_values') or {}
            for u in entry.get('Data') or [] if admin._elx_unit(u['Analysis'])}


def rows_of(sample, analysis):
    bid = next(e['bin_id'] for e in FIX['index']['runs'] if e['sample_name'] == sample)
    return [r for r in FIX['records'][bid].get('summary') or []
            if r.get('Analysis') == analysis]


def last(sample, analysis, field):
    return rows_of(sample, analysis)[-1][field]


DUR = ['Degradation rate', 'Duration', 'HFR_final', 'HFR_initial', 'V_final', 'V_initial']
POL = ['V @ 1 A/cm²', 'V @ 2 A/cm²']

# ════════════════════════════════════════════════════════════════════
print('the scan reads the electrolyzer samples and nothing else')
store = install()
preview = admin.rebuild_electrolyzer()
check('a preview writes nothing', store.puts(), [])
check('preview says so', preview['applied'], False)
read = {b for m, b in store.calls if m == 'GET' and b != 'INDEX'}
fc_bin = store.entry(SCEN['E2 fuel cell only'])['bin_id']
true('a fuel-cell-only sample is not even read', fc_bin not in read)
check('every other sample is', len(read), len(FIX['records']) - 1)
check('scanned', preview['scanned'], len(FIX['records']) - 1)
by = {r['sample_name']: r for r in preview['entries']}

expected_rebuilt = {SCEN[k] for k in (
    'E1 single cell', 'E1 durability', 'E2 single cell + durability',
    'E2 same cell twice', 'E2 two cells', 'E2 no cell', 'E2 cell then no cell',
    'E1 then E2, same cell', 'E0 then E2, same cell', 'E2 durability only',
    'E2 fuel cell + electrolyzer')}
check('the samples that change', set(by), expected_rebuilt)
check('total', preview['total'], len(expected_rebuilt))
check('a sample phase 1 already pushed again is current', preview['current'], 1)
check('a run from before Jul 27 16:15 is left as stored',
      [x['sample_name'] for x in preview['left_as_stored']], [SCEN['E0 single cell']])
check('nothing unreadable', preview['unreadable'], [])

# ════════════════════════════════════════════════════════════════════
print('the units each era comes out with')


def after(sample):
    return {(u['analysis'], u['step']): u['values'] for u in by[sample]['units_after']}


s = SCEN['E2 single cell + durability']
check('Aug 4 era: a pol curve and an EIS unit for the cell, and the durability test',
      sorted(after(s)), [('durability', ''), ('elx_eis', 'a1'), ('elx_polcurve', 'a1')])
check('with the pol-curve values', sorted(after(s)[('elx_polcurve', 'a1')]), POL)
close('V @ 1 A/cm² is the summary value', after(s)[('elx_polcurve', 'a1')]['V @ 1 A/cm²'],
      last(s, 'elx_polcurve', 'V_at_1Acm2'))
close('HFR is the summary value', after(s)[('elx_eis', 'a1')]['HFR'],
      last(s, 'elx_eis', 'HFR'))
check('and all six durability values', sorted(after(s)[('durability', '')]), DUR)
close('degradation rate', after(s)[('durability', '')]['Degradation rate'],
      last(s, 'durability', 'Degradation rate'))
check('every value is new to the index', len(by[s]['gained']), 9)
check('none leaves it', by[s]['removed'], [])
check('the stand guessed from .csv files goes', by[s]['stand_cleared'], 'FCTS')
true('before: one unit per plot type, none with a value',
     len(by[s]['units_before']) >= 9
     and not any(u['values'] for u in by[s]['units_before']))

s = SCEN['E1 single cell']
check('Jul 27 to Aug 4: the six EIS units collapse to one',
      sorted(after(s)), [('elx_eis', 'a1'), ('elx_polcurve', 'a1')])
check('before had six, split by a misread setpoint',
      sum(1 for u in by[s]['units_before'] if u['analysis'] == 'elx_eis'), 6)
check('nothing is gained', by[s]['gained'], [])
check('the fit-derived values leave the index, to run detail',
      sorted((x['name'], x['detail_only']) for x in by[s]['removed']),
      [('ASR_total', True), ('E_rev', True), ('Fit RMSE', True),
       ('Tafel slope (anode)', True)])
close('V @ 2 A/cm² unchanged', after(s)[('elx_polcurve', 'a1')]['V @ 2 A/cm²'],
      last(s, 'elx_polcurve', 'V_at_2Acm2'))
check('the CLR unit goes with its plots, re-filed under EIS',
      any(u['analysis'] == 'CLR' for u in by[s]['units_before']), True)

s = SCEN['E1 durability']
check('a durability test from then only loses its guessed stand',
      (by[s]['gained'], by[s]['removed'], by[s]['stand_cleared']), ([], [], 'FCTS'))
check('its record is left as it was', by[s]['record_changed'], False)

s = SCEN['E2 same cell twice']
check('the same cell twice: one set of values', sorted(after(s)),
      [('elx_eis', 'a1'), ('elx_polcurve', 'a1')])
check('the earlier EIS row collapsed', by[s]['rows_collapsed'], 1)

s = SCEN['E2 two cells']
check('two cells under the old names: only the second has plots left',
      sorted(after(s)), [('elx_eis', 'a2'), ('elx_polcurve', 'a2')])
close('and it gets its own HFR, not the first cell\'s',
      after(s)[('elx_eis', 'a2')]['HFR'], last(s, 'elx_eis', 'HFR'))
true('the two HFR rows differ, so that is a real check',
     abs(rows_of(s, 'elx_eis')[0]['HFR'] - rows_of(s, 'elx_eis')[-1]['HFR']) > 1e-4)
check('the overwritten cell is pointed out', by[s]['cells_missing'], ['a1'])

s = SCEN['E2 no cell']
check('a run that named no cell: units with no step',
      sorted(after(s)), [('elx_eis', ''), ('elx_polcurve', '')])
check('with values', sorted(after(s)[('elx_polcurve', '')]), POL)
check('its catalyst-layer plots leave the fuel-cell CLR bucket too',
      any(u['analysis'] == 'CLR' for u in by[s]['units_before']), True)
check('so its stand was only guessed', by[s]['stand_cleared'], 'FCTS')
check('no cell is pointed out as missing', by[s]['cells_missing'], [])

s = SCEN['E2 cell then no cell']
check('a cell, then a run with none over it: the second run\'s values',
      sorted(after(s)), [('elx_eis', ''), ('elx_polcurve', '')])
check('earlier row collapsed', by[s]['rows_collapsed'], 1)

for k in ('E1 then E2, same cell', 'E0 then E2, same cell'):
    s = SCEN[k]
    check(f'{k}: one cell', sorted(after(s)), [('elx_eis', 'a1'), ('elx_polcurve', 'a1')])
    close(f'{k}: latest V @ 1 A/cm²', after(s)[('elx_polcurve', 'a1')]['V @ 1 A/cm²'],
          last(s, 'elx_polcurve', 'V_at_1Acm2'))

s = SCEN['E2 durability only']
check('a durability test alone gains its six values',
      sorted(after(s)[('durability', '')]), DUR)

s = SCEN['E2 fuel cell + electrolyzer']
check('a sample with both kinds: electrolyzer units rebuilt',
      sorted(after(s)), [('elx_eis', 'a1'), ('elx_polcurve', 'a1')])
check('its stand is kept: the fuel-cell files are what it came from',
      by[s]['stand_cleared'], None)

# ════════════════════════════════════════════════════════════════════
print('preparing hands over the backup and writes nothing')
store = install()
# A fuel-cell unit as some older code wrote it, unlike what today's would
# build from the same record. Not this operation's business: kept as stored.
mixed = store.entry(SCEN['E2 fuel cell + electrolyzer'])
fc_unit = next(u for u in mixed['Data'] if u['Analysis'] == 'polcurve' and u['step'])
fc_unit['key_values']['Written by older code'] = 1.0
before_index = copy.deepcopy(store.bins['INDEX'])
prep = admin.rebuild_electrolyzer(prepare=True)
check('nothing written', store.puts(), [])
check('the keys of the entries it would rebuild',
      sorted(store.entry(n)['bin_id'] for n in expected_rebuilt), sorted(prep['keys']))
bak = prep['backup']
check('the backup is in the format the Restore pane reads',
      admin.inspect_backup(bak)['index_snapshot_entries'], len(before_index['runs']))
check('it holds the index as it is', bak['index'], before_index)
check('and every entry it would replace',
      sorted(e['sample_name'] for e in bak['removed']), sorted(expected_rebuilt))
check('with every record, as stored',
      {p['bin_id']: plain(p['record']) for p in bak['purged']},
      {e['bin_id']: plain(stored_record(e['bin_id'])) for e in before_index['runs']
       if e['sample_name'] in expected_rebuilt})
check('as read from the store, byte for byte',
      [p['record'] == store.bins[p['bin_id']] for p in bak['purged']],
      [True] * len(bak['purged']))
true('and says what it is for', 'does not undo a rebuild' in bak.get('about', ''))

print('apply writes only against a prepared backup')
store.calls.clear()
try:
    admin.rebuild_electrolyzer(apply=True)
    check('apply without preparing is refused', 'not refused', 'refused')
except ValueError as e:
    true('apply without preparing is refused', 'prepare' in str(e), f'({e})')
s2 = SCEN['E2 two cells']
saved = copy.deepcopy(store.bins[store.entry(s2)['bin_id']])
store.bins[store.entry(s2)['bin_id']]['summary'][0]['HFR'] = 0.5   # pushed since
try:
    admin.rebuild_electrolyzer(apply=True, keys=prep['keys'], token=prep['token'])
    check('a store changed since the backup is not written', 'written', 'refused')
except ValueError as e:
    true('a store changed since the backup is not written', 'changed' in str(e), f'({e})')
store.bins[store.entry(s2)['bin_id']] = saved
try:
    admin.rebuild_electrolyzer(apply=True, keys=prep['keys'][1:], token=prep['token'])
    check('nor is a different set of entries', 'written', 'refused')
except ValueError:
    passed += 1
check('none of those wrote anything', store.puts(), [])

res = admin.rebuild_electrolyzer(apply=True, keys=prep['keys'], token=prep['token'])
check('applied', res['applied'], True)
check('every prepared entry rebuilt', res['rebuilt'], len(expected_rebuilt))
check('no failures', res['failed'], [])
check('no backup sent twice', 'backup' in res, False)
puts = store.puts()
check('the index written once', puts.count('INDEX'), 1)
check('and last', puts[-1], 'INDEX')
changed_records = {r['bin_id'] for r in preview['entries'] if r['record_changed']}
check('only records that changed are rewritten', set(puts) - {'INDEX'}, changed_records)
true('which is not all of them — an entry can change alone',
     len(changed_records) < len(expected_rebuilt))

after_index = store.bins['INDEX']
by_name = {e['sample_name']: e for e in after_index['runs']}
check('no entry added or lost', sorted(by_name),
      sorted(e['sample_name'] for e in before_index['runs']))
for k in ('E2 fuel cell only', 'E0 single cell', 'E2 then phase 1, same cell'):
    s = SCEN[k]
    check(f'{k}: entry untouched', by_name[s],
          next(e for e in before_index['runs'] if e['sample_name'] == s))
s = SCEN['E2 fuel cell + electrolyzer']
old_fc = [u for u in next(e for e in before_index['runs'] if e['sample_name'] == s)['Data']
          if u['Analysis'] == 'polcurve']
check('the fuel-cell units of a mixed sample are exactly as they were',
      [u for u in by_name[s]['Data'] if u['Analysis'] == 'polcurve'], old_fc)
true('down to a value today\'s code would not write',
     any('Written by older code' in (u.get('key_values') or {})
         for u in by_name[s]['Data']))
check('and its stand', by_name[s].get('stand'), 'Scribner')
check('no CLR unit is left behind by its re-filed plots',
      sorted(e['sample_name'] for e in after_index['runs']
             if any(u['Analysis'] == 'CLR' for u in e['Data'])),
      [SCEN['E0 single cell']])
for s in expected_rebuilt:
    check(f'{s}: index units are the preview\'s', units(by_name[s]),
          {(u['analysis'], u['step']): u['values'] for u in by[s]['units_after']})
true('every stand that was only guessed is gone',
     all('stand' not in by_name[s] for s in expected_rebuilt
         if by[s]['stand_cleared']))
for k in ('job_id', 'n_jobs', 'timestamp', 'script', 'run_date', 'bin_id'):
    check(f'{k} kept on every entry', [e.get(k) for e in after_index['runs']],
          [e.get(k) for e in before_index['runs']])

rec = store.bins[store.entry(SCEN['E2 two cells'])['bin_id']]
true('a rewritten record is filed under the current rules',
     set(rec['metrics']) == {'elx_polcurve', 'elx_eis'})
check('with its plots and their data kept', sorted(decode_sidecars(rec)),
      sorted(decode_sidecars(stored_record(store.entry(SCEN['E2 two cells'])['bin_id']))))
check('and its rows named by cell', sorted({r.get('step') for r in rec['summary']}), ['a2'])

print('running it again changes nothing')
store.calls.clear()
again = admin.rebuild_electrolyzer()
check('nothing left to rebuild', again['total'], 0)
check('all current but the one left as stored',
      (again['current'], len(again['left_as_stored'])), (len(expected_rebuilt) + 1, 1))
prep2, res2 = rebuild()
check('a second prepare has nothing to hand over', (prep2['keys'], res2), ([], None))
check('and nothing was written', store.puts(), [])
check('the stored records need nothing from a push either',
      [b for b in changed_records
       if normalise_electrolyzer(store.bins[b]) is not store.bins[b]], [])
# A push lists its units in whatever order it built them. Order is not
# content: an entry that differs only in that is not rewritten.
for e in store.bins['INDEX']['runs']:
    e['Data'].reverse()
check('units in another order are still current',
      admin.rebuild_electrolyzer()['total'], 0)

# ════════════════════════════════════════════════════════════════════
print('a record that fails to write leaves its entry alone')
store = install()
two = store.entry(SCEN['E2 two cells'])
store.fail[('PUT', two['bin_id'])] = 'HTTP 413'
_, res = rebuild()
check('reported', [f['sample_name'] for f in res['failed']], [SCEN['E2 two cells']])
check('its entry is as it was', store.entry(SCEN['E2 two cells']), two)
check('the rest are written', res['rebuilt'], len(expected_rebuilt) - 1)
store.fail.clear()
_, res = rebuild()
check('the next run finishes it', (res['rebuilt'], res['failed']), (1, []))
check('and then nothing is left', admin.rebuild_electrolyzer()['total'], 0)

print('an index that fails to write is finished by the next run')
store = install()
index_before = copy.deepcopy(store.bins['INDEX'])
store.fail[('PUT', 'INDEX')] = 'HTTP 502'
_, res = rebuild()
true('the response says what happened', 'index could not be' in res.get('error', ''),
     f"({res.get('error')})")
check('not applied', res['applied'], False)
check('but says how many records it rewrote', res['rebuilt'], len(expected_rebuilt))
check('the index is as it was', store.bins['INDEX'], index_before)
store.fail.clear()
store.calls.clear()
_, res = rebuild()
check('the next run rebuilds every entry', res['rebuilt'], len(expected_rebuilt))
check('writing only the index — the records are already re-filed', store.puts(), ['INDEX'])

print('two entries sharing a bin are left alone')
store = install()
twin = copy.deepcopy(store.entry(SCEN['E2 two cells']))
twin['sample_name'] = '260908_E2TWO-copy'
store.bins['INDEX']['runs'].append(twin)
pv = admin.rebuild_electrolyzer()
check('both reported, neither rebuilt',
      sorted(x['sample_name'] for x in pv['unreadable']),
      sorted([SCEN['E2 two cells'], '260908_E2TWO-copy']))
true('saying why', all('same bin' in x['error'] for x in pv['unreadable']))
check('the others still are', pv['total'], len(expected_rebuilt) - 1)

# ════════════════════════════════════════════════════════════════════
print('what cannot be given a cell is reported, not guessed')
# The two-cell sample as it would be had the cells' plots not shared names:
# some of the old plots declare one cell and some the other, so the old rows
# — which name none — cannot be told apart, and a plot stored without its
# data has no cell to take.
store = install()
two = store.entry(SCEN['E2 two cells'])
rec = store.bins[two['bin_id']]
sc = decode_sidecars(rec)
for n in sorted(sc)[::2]:
    sc[n]['metadata'] = {'conditions': {'step': 'a1'}}
fit = next(n for n in sc if sc[n]['plot_type'] == 'elx_eis_fit')
del sc[fit]
attach_sidecars(rec, sc)
r = next(x for x in admin.rebuild_electrolyzer()['entries']
         if x['sample_name'] == SCEN['E2 two cells'])
check('the old cells are both seen', r['old_cells'], ['a1', 'a2'])
check('every old row reaches no unit', r['rows_unattributed'],
      len(FIX['records'][two['bin_id']]['summary']))
check('the plot without data is filed under no cell', r['plots_without_cell'], [fit])
check('no unit takes a value from those rows — HFR, which only rows carry, is nowhere',
      [u for u in r['units_after'] if 'HFR' in u['values']], [])
# A plot's own readout is another matter: it is filed under the cell its
# sidecar declares, so what it reads out goes there and nowhere else.
pol = next(n for n, s in sc.items() if s['plot_type'] == 'elx_polcurve')
cell = sc[pol]['metadata']['conditions']['step']
got = {(u['analysis'], u['step']): u['values'] for u in r['units_after']}
close('the pol curve\'s readout stays with its cell',
      got[('elx_polcurve', cell)].get('V @ 1 A/cm²'),
      rec['metrics']['elx_polcurve'][pol]['values']['V @ 1 A/cm²']['value'])
check('and reaches no other', [k for k, v in got.items() if v and k != ('elx_polcurve', cell)],
      [])
check('rows are not collapsed when their cells are unknown', r['rows_collapsed'], 0)

# Counted row by row. With all its plot data lost, the two-cell sample's plots
# are filed under no cell, its one cell for that purpose: the one pol-curve row
# reaches that unit, the two EIS rows cannot both, and so neither does.
store = install()
rec = store.bins[two['bin_id']]
attach_sidecars(rec, {})
r = next(x for x in admin.rebuild_electrolyzer()['entries']
         if x['sample_name'] == SCEN['E2 two cells'])
check('two of the three rows reach no unit', r['rows_unattributed'], 2)
check('the pol-curve one does', sorted(next(u['values'] for u in r['units_after']
                                            if u['analysis'] == 'elx_polcurve')), POL)

# A current run is not evidence for old rows nothing else places: it declares
# its conditions, not just its cell. An old record whose plot data was lost —
# nothing says which cell its runs were on — then a current run on a1.
rec = stored_record(store.entry(SCEN['E2 same cell twice'])['bin_id'])
attach_sidecars(rec, {})
cur = {'step': 'a1', 'T_C': 80.0, 'P_value': 0.0, 'P_unit': 'barg', 'P_anode_barg': 0.0}
new_sc = {}
for bucket, pt in (('elx_polcurve', 'elx_polcurve'), ('elx_eis', 'elx_nyquist')):
    name = f'NEW-{pt}_a1'
    rec['metrics'].setdefault(bucket, {})[name] = {'conditions': dict(cur), 'values': {}}
    new_sc[name] = {'plot_type': pt, 'metadata': {'conditions': dict(cur)}}
attach_sidecars(rec, new_sc)
rec['summary'] = rec['summary'] + [
    {'Analysis': 'elx_polcurve', 'V_at_1Acm2': 1.5, 'step': 'a1'},
    {'Analysis': 'elx_eis', 'HFR': 0.03, 'step': 'a1'}]
old_rows = [x for x in rec['summary'] if 'step' not in x]
fixed = normalise_electrolyzer(rec)
check('old rows that nothing places are kept as they are, not given the new cell',
      [x for x in fixed['summary'] if 'step' not in x], old_rows)
check('the current run\'s rows untouched',
      [x for x in fixed['summary'] if 'step' in x], rec['summary'][len(old_rows):])

# ════════════════════════════════════════════════════════════════════
print('a record re-filed past the size limit is left as stored')
# Fitting it would mean dropping plot data — perhaps a fuel-cell run's, or the
# only plots saying which cell an old run was on — which the rebuild never does.
store = install()
s = SCEN['E2 single cell + durability']
bid = store.entry(s)['bin_id']
size = len(json.dumps(normalise_electrolyzer(store.bins[bid]), ensure_ascii=False).encode())
real_limit = jsonbin.MAX_BODY_BYTES
jsonbin.MAX_BODY_BYTES = size - 1
try:
    entry_before, rec_before = copy.deepcopy(store.entry(s)), copy.deepcopy(store.bins[bid])
    pv = admin.rebuild_electrolyzer()
    check('reported with its size', [(x['sample_name'], x['body_bytes']) for x in pv['too_large']],
          [(s, size)])
    true('and not among those rebuilt', s not in {x['sample_name'] for x in pv['entries']})
    rebuild()
    check('its record untouched', store.bins[bid], rec_before)
    check('and its entry', store.entry(s), entry_before)
    check('the rest rebuilt', admin.rebuild_electrolyzer()['total'], 0)
finally:
    jsonbin.MAX_BODY_BYTES = real_limit

# ════════════════════════════════════════════════════════════════════
print('a summary stored as one bare dict is one row')
rec = stored_record(store.entry(SCEN['E2 no cell'])['bin_id'])
row = next(x for x in rec['summary'] if x['Analysis'] == 'elx_polcurve')
rec['summary'] = dict(row)
fixed = normalise_electrolyzer(rec)
check('it is given its cell like any old row', fixed['summary'], [dict(row, step='')])
dur = stored_record(store.entry(SCEN['E2 durability only'])['bin_id'])
dur['summary'] = dict(dur['summary'][0])
check('and a record that needs nothing keeps it as it was',
      normalise_electrolyzer(dur) is dur, True)
store = install()
store.bins[store.entry(SCEN['E2 no cell'])['bin_id']]['summary'] = dict(row)
store.bins[store.entry(SCEN['E2 durability only'])['bin_id']]['summary'] = dur['summary']
pv = admin.rebuild_electrolyzer()
check('the scan reads both', pv['unreadable'], [])
r = next(x for x in pv['entries'] if x['sample_name'] == SCEN['E2 no cell'])
check('and finds the values in it', sorted(next(u['values'] for u in r['units_after']
                                                 if u['analysis'] == 'elx_polcurve')), POL)

# ════════════════════════════════════════════════════════════════════
print('an old catalyst-layer plot goes with its run')
# Only the electrolyzer script has ever written a CLR plot type. A run that
# named no cell declared nothing, so declarations alone left its CLR plots in
# the fuel-cell bucket; a run from before Jul 27 16:15 files every plot as a
# fuel-cell one, and its CLR plots stay with the rest.
nocell = stored_record(store.entry(SCEN['E2 no cell'])['bin_id'])
keys = stored_plot_keys(nocell)
check('a no-cell run\'s CLR plots key with its EIS',
      {keys[n] for n in nocell['metrics']['CLR']}, {('elx_eis', '')})
e0 = stored_record(store.entry(SCEN['E0 single cell'])['bin_id'])
keys = stored_plot_keys(e0)
check('a pre-Jul 27 run\'s stay where its other plots are',
      {keys[n][0] for n in e0['metrics']['CLR']}, {'CLR'})
check('and that record is left exactly as it was', normalise_electrolyzer(e0) is e0, True)

print(f'\n{passed} passed, {failed} failed')
sys.exit(1 if failed else 0)
