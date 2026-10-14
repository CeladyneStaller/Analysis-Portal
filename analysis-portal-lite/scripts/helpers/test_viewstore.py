"""
Offline harness for viewstore.py. JSONBin transport is mocked.

Run:  python3 scripts/helpers/test_viewstore.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.helpers import jsonbin, viewstore  # noqa: E402
from scripts.helpers.record import encode_sidecars  # noqa: E402

_passed = 0
_failed = 0


def check(name, got, want):
    global _passed, _failed
    if got == want:
        _passed += 1
    else:
        _failed += 1
        print(f"  FAIL {name}\n       got:  {got!r}\n       want: {want!r}")


def check_true(name, cond, detail=''):
    global _passed, _failed
    if cond:
        _passed += 1
    else:
        _failed += 1
        print(f"  FAIL {name} {detail}")


# ─────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────

def sidecar(plot_type, text=''):
    return {'plot_type': plot_type, 'data': {'axes': [{
        'texts': [{'text': text}] if text else [],
        'axhlines': [], 'axvlines': [],
        'lines': [{'x': [0, 1], 'y': [1, 0], 'label': 'L'}]}]}}


DETAILS = {
    'job-A': {
        'schema': 2, 'job_id': 'job-A', 'sample_name': '260126_FCS6',
        'script': 'FC Polarization Curve', 'timestamp': '2026-05-01T10:00:00Z',
        'metrics': {'polcurve': {
            'polcurve_b14_80C_100RH': {
                'conditions': {'step': 'b14', 'T_C': 80.0},
                'values': {'OCV': {'value': 0.95, 'unit': 'V'}}}}},
        'sidecars': encode_sidecars(
            {'polcurve_b14_80C_100RH': sidecar('polcurve', 'OCV = 0.95 V')}),
    },
    'job-B': {
        'schema': 2, 'job_id': 'job-B', 'sample_name': '260421_GSMA-Qual-1',
        'script': 'Fuel Cell Full Analysis', 'timestamp': '2026-07-22T22:26:04Z',
        'metrics': {
            'polcurve': {'polcurve_b4_80C_95RH': {
                'conditions': {'step': 'b4'}, 'values': {}}},
            # Cleaning metrics are present but its sidecar was excluded at
            # write time, so this plot must report renderable: False.
            'cleaning': {'cleaning_cycles_a4': {
                'conditions': {}, 'values': {'Cycles': 50.0}}},
        },
        'sidecars': encode_sidecars(
            {'polcurve_b4_80C_95RH': sidecar('polcurve')}),
    },
}

INDEX = {'schema': 2, 'runs': [
    {'job_id': 'job-A', 'sample_name': '260126_FCS6',
     'script': 'FC Polarization Curve', 'timestamp': '2026-05-01T10:00:00Z',
     'bin_id': 'BIN-A',
     'Data': [{'Analysis': 'polcurve', 'step': 'b14',
               'Conditions': {'T_C': 80.0}, 'key_values': {'OCV': 0.95}}]},
    {'job_id': 'job-B', 'sample_name': '260421_GSMA-Qual-1',
     'script': 'Fuel Cell Full Analysis', 'timestamp': '2026-07-22T22:26:04Z',
     'bin_id': 'BIN-B',
     'Data': [{'Analysis': 'polcurve', 'step': 'b4', 'Conditions': {}},
              {'Analysis': 'cleaning', 'step': '', 'Conditions': {}}]},
]}

BIN_TO_JOB = {'BIN-A': 'job-A', 'BIN-B': 'job-B'}
_calls = []


def mock_request(url, method='GET', body=None, extra_headers=None):
    _calls.append((method, url))
    for bin_id, job_id in BIN_TO_JOB.items():
        if f'/{bin_id}/' in url:
            return {'record': DETAILS[job_id], 'metadata': {}}
    return {'record': INDEX, 'metadata': {}}


jsonbin._request = mock_request
jsonbin._JSONBIN_BASE = 'https://api.jsonbin.io/v3/b'


def reset():
    viewstore.clear_cache()
    _calls.clear()


# ─────────────────────────────────────────────────────────────────────

print("listing and filters")
reset()
res = viewstore.list_runs()
check('all runs', res['total'], 2)
check('newest first', [r['job_id'] for r in res['runs']], ['job-B', 'job-A'])

check('sample substring', viewstore.list_runs(sample='GSMA')['total'], 1)
check('sample case-insensitive', viewstore.list_runs(sample='gsma')['total'], 1)
check('sample no match', viewstore.list_runs(sample='zzz')['total'], 0)
check('script exact', viewstore.list_runs(script='FC Polarization Curve')['total'], 1)
check('analysis present in Data',
      viewstore.list_runs(analysis='cleaning')['total'], 1)
check('analysis in both', viewstore.list_runs(analysis='polcurve')['total'], 2)
check('since filter', viewstore.list_runs(since='2026-06-01')['total'], 1)
check('until filter', viewstore.list_runs(until='2026-06-01')['total'], 1)
check('range excludes both',
      viewstore.list_runs(since='2026-06-01', until='2026-06-02')['total'], 0)
check('combined filters',
      viewstore.list_runs(sample='FCS6', analysis='polcurve')['total'], 1)

r = viewstore.list_runs(limit=1)
check('limit caps returned', r['returned'], 1)
check('limit preserves total', r['total'], 2)

print("ordering by experiment date")
# The index entries were analysed in one batch but the experiments span months,
# so ordering must follow run_date where it exists.
_orig = INDEX['runs']
INDEX['runs'] = [
    {'job_id': 'old-exp', 'sample_name': '260407_BM1', 'script': 'S',
     'timestamp': '2026-07-23T18:00:00Z', 'run_date': '2026-04-07',
     'bin_id': 'B1', 'Data': []},
    {'job_id': 'new-exp', 'sample_name': '260511_Gen2', 'script': 'S',
     'timestamp': '2026-07-22T10:00:00Z', 'run_date': '2026-05-11',
     'bin_id': 'B2', 'Data': []},
    {'job_id': 'undated', 'sample_name': 'NoPrefix', 'script': 'S',
     'timestamp': '2026-07-23T19:00:00Z', 'bin_id': 'B3', 'Data': []},
]
viewstore.clear_cache()
order = [r['job_id'] for r in viewstore.list_runs()['runs']]
check('newest experiment first, undated falls back to analysis date',
      order, ['undated', 'new-exp', 'old-exp'])
check_true('run_date beats a later analysis timestamp',
           order.index('new-exp') < order.index('old-exp'))
INDEX['runs'] = _orig
viewstore.clear_cache()

print("facets")
f = viewstore.list_runs()['facets']
check('sample facet', f['samples'], ['260126_FCS6', '260421_GSMA-Qual-1'])
check('analysis facet', f['analyses'], ['cleaning', 'polcurve'])
check('script facet', len(f['scripts']), 2)

print("index caching")
reset()
viewstore.list_runs()
n1 = len(_calls)
viewstore.list_runs()
viewstore.list_runs()
check('index fetched once within TTL', len(_calls), n1)
viewstore.list_runs(force=True)
check_true('force refetches', len(_calls) > n1)

print("detail fetch and cache")
reset()
d = viewstore.fetch_detail('job-A')
check('right record', d['job_id'], 'job-A')
n_after_first = len(_calls)
viewstore.fetch_detail('job-A')
check('second fetch served from cache', len(_calls), n_after_first)

stats = viewstore.cache_stats()
check('one cached', stats['detail_cached'], 1)
check('capacity reported', stats['detail_capacity'], viewstore.DETAIL_CACHE_SIZE)

raised = False
try:
    viewstore.fetch_detail('job-missing')
except KeyError:
    raised = True
check_true('unknown job raises KeyError', raised)

# LRU eviction: shrink the cache and confirm the oldest goes first.
orig = viewstore.DETAIL_CACHE_SIZE
viewstore.DETAIL_CACHE_SIZE = 1
reset()
viewstore.fetch_detail('job-A')
viewstore.fetch_detail('job-B')
check('evicted to capacity', viewstore.cache_stats()['detail_cached'], 1)
check_true('newest retained',
           any('BIN-B' in k for k in viewstore.cache_stats()['detail_keys']))
viewstore.DETAIL_CACHE_SIZE = orig

print("keying by bin id, sample and job")
reset()
# bin_id is unambiguous; sample_name and job_id resolve too.
check('resolve by bin id', viewstore.fetch_detail('BIN-A')['job_id'], 'job-A')
check('resolve by job id', viewstore.fetch_detail('job-A')['job_id'], 'job-A')
check('resolve by sample name',
      viewstore.fetch_detail('260126_FCS6')['job_id'], 'job-A')
raised = False
try:
    viewstore.fetch_detail('nothing-like-this')
except KeyError:
    raised = True
check_true('unknown key raises', raised)

# Merging rewrites a bin in place, so the cache must key on the entry's
# timestamp rather than treating detail records as immutable.
reset()
viewstore.fetch_detail('BIN-A')
n_before = len(_calls)
viewstore.fetch_detail('BIN-A')
check('unchanged entry is served from cache', len(_calls), n_before)
INDEX['runs'][0]['timestamp'] = '2026-08-01T00:00:00Z'   # simulate a merge
viewstore.clear_cache()
_calls.clear()
viewstore.fetch_detail('BIN-A')
first = len(_calls)
INDEX['runs'][0]['timestamp'] = '2026-08-02T00:00:00Z'   # merged again
viewstore._index_cache['at'] = 0                          # index TTL expires
viewstore.fetch_detail('BIN-A')
check_true('a merged bin is re-read rather than served stale',
           len(_calls) > first)
INDEX['runs'][0]['timestamp'] = '2026-05-01T10:00:00Z'
reset()

print("plot inventory")
reset()
plots = viewstore.run_plots('job-B')
check('both plots listed', len(plots), 2)
by_name = {p['plot']: p for p in plots}
check_true('stored sidecar is renderable',
           by_name['polcurve_b4_80C_95RH']['renderable'])
check_true('excluded cleaning sidecar is not renderable',
           not by_name['cleaning_cycles_a4']['renderable'])
check('cleaning metrics still present',
      by_name['cleaning_cycles_a4']['values'], {'Cycles': 50.0})
check('analysis tagged', by_name['cleaning_cycles_a4']['analysis'], 'cleaning')

print("materialisation")
reset()
tmp = tempfile.mkdtemp()
try:
    written = viewstore.materialize_sidecars('job-A', Path(tmp))
    check('one sidecar written', written, ['polcurve_b14_80C_100RH'])
    path = Path(tmp) / '_plot_data' / 'polcurve_b14_80C_100RH.json'
    check_true('written in analysis-output layout', path.exists())
    check('content round-trips', json.loads(path.read_text())['plot_type'],
          'polcurve')

    # find_sidecar must locate it — this is the whole point of the layout.
    from scripts.helpers.plot_compare import find_sidecar
    found = find_sidecar(tmp, 'polcurve_b14_80C_100RH.png')
    check_true('find_sidecar locates the materialised file', bool(found),
               f'(got {found!r})')

    # Requesting an unstored plot yields nothing rather than erroring.
    none_written = viewstore.materialize_sidecars(
        'job-B', Path(tmp) / 'b', plots=['cleaning_cycles_a4'])
    check('unstored plot skipped', none_written, [])
finally:
    shutil.rmtree(tmp)

print("compare staging")
reset()
tmp = tempfile.mkdtemp()
try:
    sources = viewstore.materialize_for_compare([
        {'job_id': 'job-A', 'plot': 'polcurve_b14_80C_100RH'},
        {'job_id': 'job-B', 'plot': 'polcurve_b4_80C_95RH'},
        # Not stored — must be dropped rather than producing a broken source.
        {'job_id': 'job-B', 'plot': 'cleaning_cycles_a4'},
    ], Path(tmp))
    check('unstored selection dropped', len(sources), 2)
    check('sample names attached',
          sorted(s['sample_name'] for s in sources),
          ['260126_FCS6', '260421_GSMA-Qual-1'])
    check_true('per-run directories',
              sources[0]['output_dir'] != sources[1]['output_dir'])
    check_true('filenames carry an extension',
               all(s['filename'].endswith('.png') for s in sources))

    from scripts.helpers.plot_compare import find_sidecar
    check_true('every staged source resolves',
               all(find_sidecar(s['output_dir'], s['filename'])
                   for s in sources))

    check('label defaults empty', sources[0]['label'], '')
    labelled = viewstore.materialize_for_compare(
        [{'job_id': 'job-A', 'plot': 'polcurve_b14_80C_100RH',
          'label': 'baseline'}], Path(tmp) / 'l')
    check('label passed through', labelled[0]['label'], 'baseline')
finally:
    shutil.rmtree(tmp)

print("RH cycles axis")
# FC Durability stores, with a test that cycles RH, its hold hours against RH
# cycles completed; Comparisons' Export can stage a durability plot against
# cycles off it. Here a first hold cycles RH for 2 h (50 an hour) and a second
# holds it steady for 1 h.
MAP = {'plot_type': 'fcdur_rh_cycles', 'data': {'axes': [{
    'xlabel': 'Hold hours (h)', 'ylabel': 'RH cycles', 'texts': [], 'axhlines': [],
    'axvlines': [{'x': 2.0}],
    'lines': [{'x': [0.0, 1.0, 2.0, 3.0], 'y': [0, 50, 100, 100],
               'label': 'RH cycles completed'}]}]}}


def ocv_plot():
    return {'plot_type': 'fcdur_ocv_vs_time', 'data': {'axes': [{
        'xlabel': 'Hold time (h)', 'ylabel': 'Hold OCV (V)', 'xlim': [-0.15, 3.15],
        'texts': [{'text': 'OCV loss = 1 %', 'x': 0.02, 'y': 0.9, 'in_axes_fraction': True},
                  {'text': 'death', 'x': 2.75, 'y': 0.7, 'in_axes_fraction': False}],
        'axhlines': [{'y': 0.7}], 'axvlines': [{'x': 2.0}, {'x': 2.75}],
        'lines': [{'x': [0.5, 1.5], 'y': [0.95, 0.94], 'label': '100↔0% RH cycling'},
                  {'x': [2.5], 'y': [0.93], 'label': '30% RH'}],
        'scatters': [{'x': [0.5, 1.5], 'y': [0.9, 0.9], 'label': 'bin low'}]}]}}


check('the map: hours and cycles, in order',
      viewstore.rh_cycle_map({'m': MAP, 'o': ocv_plot()}), ([0.0, 1.0, 2.0, 3.0], [0.0, 50.0, 100.0, 100.0]))
stepped = json.loads(json.dumps(MAP))
stepped['data']['axes'][0]['lines'][0].update(x=[0.0, 1.0, 0.5, 2.0, None], y=[0, 50, 25, 100, 120])
check('... a point stepping back in hours, or missing a number, left out',
      viewstore.rh_cycle_map({'m': stepped}), ([0.0, 1.0, 2.0], [0.0, 50.0, 100.0]))
short = json.loads(json.dumps(MAP))
short['data']['axes'][0]['lines'][0].update(x=[0.0], y=[0])
renamed = json.loads(json.dumps(MAP))
renamed['data']['axes'][0]['lines'][0]['label'] = 'cycles'
check('... none from one point, another line, or a test without the plot',
      [viewstore.rh_cycle_map(s) for s in ({'m': short}, {'m': renamed}, {'o': ocv_plot()}, {})],
      [None, None, None, None])
H, C = viewstore.rh_cycle_map({'m': MAP})
check('cycles at an hour: straight between points, held at the ends, a gap passed through',
      [viewstore.cycles_at(H, C, h) for h in (-1, 0, 0.5, 1.0, 1.25, 2.5, 3.0, 9.0, None)],
      [0.0, 0.0, 25.0, 50.0, 62.5, 100.0, 100.0, 100.0, None])
moved = viewstore.sidecar_on_cycles(ocv_plot(), H, C)['data']['axes'][0]
check('a plot against cycles: lines, points, vertical lines, text placed by data, range, label',
      ([ln['x'] for ln in moved['lines']], moved['scatters'][0]['x'],
       [v['x'] for v in moved['axvlines']], [t['x'] for t in moved['texts']],
       moved['xlim'], moved['xlabel'], moved['ylabel'], moved['axhlines']),
      ([[25.0, 75.0], [100.0]], [25.0, 75.0], [100.0, 100.0], [0.02, 100.0],
       [0.0, 100.0], 'RH cycles', 'Hold OCV (V)', [{'y': 0.7}]))

_orig = INDEX['runs']
INDEX['runs'] = _orig + [
    {'job_id': 'job-C', 'sample_name': '261001_CYC', 'script': 'FC Durability',
     'timestamp': '2026-10-01T10:00:00Z', 'bin_id': 'BIN-C',
     'Data': [{'Analysis': 'fc_durability', 'step': '', 'Conditions': {}}]},
    {'job_id': 'job-D', 'sample_name': '261002_STEADY', 'script': 'FC Durability',
     'timestamp': '2026-10-02T10:00:00Z', 'bin_id': 'BIN-D',
     'Data': [{'Analysis': 'fc_durability', 'step': '', 'Conditions': {}}]}]
BIN_TO_JOB.update({'BIN-C': 'job-C', 'BIN-D': 'job-D'})
long_map = json.loads(json.dumps(MAP))
long_map['data']['axes'][0]['lines'][0].update(x=[k / 100 for k in range(901)],
                                               y=[min(k, 450) for k in range(901)])
for job, sample, plots in (('job-C', '261001_CYC', {'ocv': ocv_plot(), 'cyc': MAP}),
                           ('job-D', '261002_STEADY', {'ocv': ocv_plot()}),
                           ('job-L', '261003_LONG', {'cyc': long_map})):
    DETAILS[job] = {
        'schema': 2, 'job_id': job, 'sample_name': sample, 'script': 'FC Durability',
        'metrics': {'fc_durability': {f'{sample}-FCDurability-{k}': {
            'conditions': {'step': ''}, 'values': {}} for k in plots}},
        'sidecars': encode_sidecars({f'{sample}-FCDurability-{k}': v for k, v in plots.items()})}
INDEX['runs'].append({'job_id': 'job-L', 'sample_name': '261003_LONG', 'script': 'FC Durability',
                      'timestamp': '2026-10-03T10:00:00Z', 'bin_id': 'BIN-L',
                      'Data': [{'Analysis': 'fc_durability', 'step': '', 'Conditions': {}}]})
BIN_TO_JOB['BIN-L'] = 'job-L'
reset()
tmp = tempfile.mkdtemp()
try:
    ask = {'analysis': 'fc_durability', 'step': '', 'plot_type': 'fcdur_rh_cycles', 'full': True}
    got = viewstore.plot_series_batch([dict(ask, key='BIN-L'), dict(ask, key='BIN-D')])
    line = got['series'][0]['axes'][0]['lines'][0]
    check('the map sent whole to the chart, 901 points not thinned; a test without one missing',
          (len(line['x']), line.get('decimated', False), [m['key'] for m in got['missing']]),
          (901, False, ['BIN-D']))
    OCV_C, OCV_D = '261001_CYC-FCDurability-ocv', '261002_STEADY-FCDurability-ocv'
    sources = viewstore.materialize_for_compare([
        {'key': 'BIN-C', 'plot': OCV_C, 'label': 'cycled', 'x': 'cycles'},
        {'key': 'BIN-C', 'plot': OCV_C, 'label': 'again', 'x': 'cycles'},
        {'key': 'BIN-D', 'plot': OCV_D, 'label': 'steady', 'x': 'cycles'},
        {'key': 'BIN-A', 'plot': 'polcurve_b14_80C_100RH', 'label': 'polcurve', 'x': 'cycles'},
    ], Path(tmp))
    check('staged against cycles: the cycling test (once for both), not the steady one; '
          'not a durability plot, as stored',
          [(s['label'], s['plot_type']) for s in sources],
          [('cycled', 'fcdur_ocv_vs_time'), ('again', 'fcdur_ocv_vs_time'), ('polcurve', 'polcurve')])
    from scripts.helpers.plot_compare import find_sidecar, load_sidecar
    staged = [load_sidecar(find_sidecar(s['output_dir'], s['filename']))['data']['axes'][0]
              for s in sources]
    check('... its x read off its own map, converted once',
          ([ln['x'] for ln in staged[0]['lines']], [v['x'] for v in staged[0]['axvlines']],
           staged[0]['xlabel'], staged[1]['lines'][0]['x']),
          ([[25.0, 75.0], [100.0]], [100.0, 100.0], 'RH cycles', [25.0, 75.0]))
    check('... the polcurve staged as stored', (staged[2]['lines'][0]['x'], staged[2].get('xlabel')),
          ([0, 1], None))
    plain = viewstore.materialize_for_compare([
        {'key': 'BIN-C', 'plot': OCV_C}, {'key': 'BIN-D', 'plot': OCV_D, 'x': 'hours'}], Path(tmp) / 'h')
    check('... and on hold hours, every test as stored',
          [load_sidecar(find_sidecar(s['output_dir'], s['filename']))['data']['axes'][0]['lines'][0]['x']
           for s in plain], [[0.5, 1.5], [0.5, 1.5]])
    both = viewstore.materialize_for_compare([
        {'key': 'BIN-C', 'plot': OCV_C, 'x': 'cycles'}, {'key': 'BIN-C', 'plot': OCV_C},
        {'key': 'BIN-C', 'plot': '261001_CYC-FCDurability-cyc', 'x': 'cycles'}], Path(tmp) / 'b')
    check('... the same plot asked for on both axes staged both ways; a plot not against hold '
          'hours (the map itself) as stored',
          [(lambda a: (a['lines'][0]['x'][:2], a.get('xlabel')))(
              load_sidecar(find_sidecar(s['output_dir'], s['filename']))['data']['axes'][0]) for s in both],
          [([25.0, 75.0], 'RH cycles'), ([0.5, 1.5], 'Hold time (h)'), ([0.0, 1.0], 'Hold hours (h)')])
finally:
    shutil.rmtree(tmp)
    INDEX['runs'] = _orig
    for b, j in (('BIN-C', 'job-C'), ('BIN-D', 'job-D'), ('BIN-L', 'job-L')):
        BIN_TO_JOB.pop(b)
        DETAILS.pop(j)
    reset()

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
