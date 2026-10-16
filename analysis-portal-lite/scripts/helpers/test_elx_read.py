"""The read API for electrolyzer runs.

  * viewstore.plot_series, asked for an (analysis, step), returns that
    analysis's main plot — the pol curve, the Nyquist plot, the durability
    voltage — not whichever plot the script happened to write first; asked
    for a plot type, it returns that one. Fuel-cell analyses, which list no
    main plots, keep the rule they have always had.
  * /api/view/runs (viewstore.list_runs) carries the registry, and names
    the analysis each old per-plot-type bucket in the index belongs to.
  * The run detail's plot list (viewstore.run_plots) carries each plot's
    type, which Recall orders an analysis's main plots by.

Run:  python3 scripts/helpers/test_elx_read.py
"""

import copy
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.update(JSONBIN_API_KEY='k', JSONBIN_COLLECTION_ID='C',
                  JSONBIN_INDEX_BIN_ID='INDEX')

from scripts.helpers import jsonbin, registry, viewstore            # noqa: E402
from scripts.helpers.record import attach_sidecars, build_index_entry  # noqa: E402

passed = failed = 0


def check(name, got, want):
    global passed, failed
    if got == want:
        passed += 1
    else:
        failed += 1
        print(f'  FAIL {name}\n       got  {got!r}\n       want {want!r}')


def sidecar(plot_type, cell=None):
    return {'plot_type': plot_type,
            'metadata': {'conditions': {'step': cell}} if cell else {},
            'data': {'axes': [{'title': plot_type, 'xlabel': 'x', 'ylabel': 'y',
                               'texts': [], 'axhlines': [], 'axvlines': [],
                               'lines': [{'label': plot_type, 'x': [0, 1, 2],
                                          'y': [1.0, 2.0, 3.0]}]}]}}


S = '260811_ELX'
PRE = f'{S}-ElxPolCurve-'
# Each cell's plots in the order the script writes them: the iR correction
# before the pol curve, the per-spectrum fits before the Nyquist plot.
POL = [('ir_correction', 'elx_ir_correction'), ('polcurve', 'elx_polcurve'),
       ('j_vs_cycle', 'elx_j_vs_cycle'), ('losses_vs_cycle', 'elx_j_and_losses_vs_cycle'),
       ('model_fit', 'elx_model_fit')]
EIS = [('eis_fit_V_dc1.250V', 'elx_eis_fit'), ('eis_for_ir', 'elx_eis_for_ir'),
       ('nyquist', 'elx_nyquist'), ('CLR_analysis', 'elx_clr_analysis')]
DUR = [('asr_vs_j_evolution', 'durability_asr_evolution'),
       ('hfr_vs_time', 'durability_hfr'), ('voltage_vs_time', 'durability_voltage'),
       ('degradation_rate', 'durability_degradation')]


def elx_record(drop=()):
    metrics, sidecars = {}, {}
    for cell in ('a1', 'a2'):
        for bucket, plots in (('elx_polcurve', POL), ('elx_eis', EIS)):
            for stem, pt in plots:
                name = f'{PRE}{stem}_{cell}'
                metrics.setdefault(bucket, {})[name] = {
                    'conditions': {'step': cell}, 'values': {}}
                sidecars[name] = sidecar(pt, cell)
    for stem, pt in DUR:
        name = f'{S}-ElxDurability-{stem}'
        metrics.setdefault('durability', {})[name] = {'conditions': {}, 'values': {}}
        sidecars[name] = sidecar(pt)
    rec = {'schema': 2, 'job_id': 'j1', 'sample_name': S,
           'script': 'Electrolyzer Pol Curve', 'timestamp': '2026-10-07T00:00:00Z',
           'input_files': [], 'metrics': metrics, 'conditions': None}
    attach_sidecars(rec, {n: s for n, s in sidecars.items()
                          if not any(n.endswith(d) for d in drop)})
    return rec


F = '260804_FCS'
FC_PLOT = f'{F}-PolCurve-polcurve_{F}_b14_IV_80C'
FC_OVER = f'{F}-PolCurve-polcurve_batch_overlay_b14'


def fc_record(store_first=True):
    metrics = {'polcurve': {
        FC_OVER: {'conditions': {'step': 'b14'}, 'values': {}},
        FC_PLOT: {'conditions': {'step': 'b14'}, 'values': {}}}}
    rec = {'schema': 2, 'job_id': 'j2', 'sample_name': F,
           'script': 'FC Polarization Curve', 'timestamp': '2026-10-07T00:00:00Z',
           'input_files': [], 'metrics': metrics, 'conditions': None}
    sc = {FC_PLOT: sidecar('polcurve')}
    if store_first:
        sc[FC_OVER] = sidecar('polcurve_overlay')
    attach_sidecars(rec, sc)
    return rec


class Store:
    def __init__(self, records):
        self.bins = {'INDEX': {'schema': 2, 'runs': [
            build_index_entry(r, bid) for bid, r in records.items()]}}
        self.bins.update(copy.deepcopy(records))

    def __call__(self, url, method='GET', body=None, extra_headers=None):
        assert method == 'GET', method
        bid = url.rstrip('/').split('/')[-2]
        if bid not in self.bins:
            raise RuntimeError(f'404 {bid}')
        return {'record': copy.deepcopy(self.bins[bid])}


def use(records):
    jsonbin._request = Store(records)
    viewstore.clear_cache()


def got(analysis, step, plot_type='', key='ELX', plot=''):
    sr = viewstore.plot_series(key, analysis, step, plot=plot, plot_type=plot_type)
    return (sr['plot_type'], sr['plot'][len(PRE):] if sr['plot'].startswith(PRE)
            else sr['plot']) if sr else None


# ════════════════════════════════════════════════════════════════════
print('an (analysis, step) gets its main plot')
use({'ELX': elx_record(), 'FC': fc_record()})
check('the pol curve, not the iR correction written before it',
      got('elx_polcurve', 'a1'), ('elx_polcurve', 'polcurve_a1'))
check('each cell its own', got('elx_polcurve', 'a2'), ('elx_polcurve', 'polcurve_a2'))
check('the Nyquist plot for EIS, not a per-spectrum fit',
      got('elx_eis', 'a1'), ('elx_nyquist', 'nyquist_a1'))
check('the voltage for a durability test, which has no step',
      got('durability', ''), ('durability_voltage', f'{S}-ElxDurability-voltage_vs_time'))
sr = viewstore.plot_series('ELX', 'elx_polcurve', 'a1')
check('the payload says which unit it answers', (sr['analysis'], sr['step']),
      ('elx_polcurve', 'a1'))
check('and carries the series', sr['axes'][0]['lines'][0]['y'], [1.0, 2.0, 3.0])
check('a cell that is not there gets nothing', got('elx_polcurve', 'a3'), None)
check('nor does an analysis the sample lacks', got('elx_polcurve', '', key='FC'), None)

print('a plot type asks for that plot')
check('cycling', got('elx_polcurve', 'a1', 'elx_j_vs_cycle'),
      ('elx_j_vs_cycle', 'j_vs_cycle_a1'))
check('of the right cell', got('elx_polcurve', 'a2', 'elx_model_fit'),
      ('elx_model_fit', 'model_fit_a2'))
check('a plot that is not a main one', got('elx_eis', 'a2', 'elx_eis_fit'),
      ('elx_eis_fit', 'eis_fit_V_dc1.250V_a2'))
check('a type the run did not produce gets nothing',
      got('elx_polcurve', 'a1', 'elx_v_vs_cycle'), None)
check('nor does a type from another analysis',
      got('elx_eis', 'a1', 'elx_polcurve'), None)
check('a name still wins over both', got('elx_eis', 'a1', 'elx_nyquist',
                                         plot=f'{PRE}model_fit_a1'),
      ('elx_model_fit', 'model_fit_a1'))

print('when the main plot was not stored')
use({'ELX': elx_record(drop=('-polcurve_a1', '-nyquist_a1', '-eis_for_ir_a1'))})
check('the next main plot in the registry\'s order',
      got('elx_polcurve', 'a1'), ('elx_j_vs_cycle', 'j_vs_cycle_a1'))
check('the other cell is unaffected', got('elx_polcurve', 'a2'),
      ('elx_polcurve', 'polcurve_a2'))
check('a unit with no main plot stored gets nothing, not a fit',
      got('elx_eis', 'a1'), None)
check('which is still there when asked for by type',
      got('elx_eis', 'a1', 'elx_eis_fit'), ('elx_eis_fit', 'eis_fit_V_dc1.250V_a1'))

print('fuel-cell analyses keep their rule: the first plot filed')
use({'ELX': elx_record(), 'FC': fc_record()})
check('the first plot, stored', got('polcurve', 'b14', key='FC'),
      ('polcurve_overlay', FC_OVER))
check('a type picks the other', got('polcurve', 'b14', 'polcurve', key='FC'),
      ('polcurve', FC_PLOT))
use({'FC': fc_record(store_first=False)})
check('and a first plot without its data still means nothing, as before',
      got('polcurve', 'b14', key='FC'), None)

print('a batch passes plot types through')
use({'ELX': elx_record(), 'FC': fc_record()})
out = viewstore.plot_series_batch([
    {'key': 'ELX', 'analysis': 'elx_polcurve', 'step': 'a1'},
    {'key': 'ELX', 'analysis': 'elx_polcurve', 'step': 'a1', 'plot_type': 'elx_j_vs_cycle'},
    {'key': 'ELX', 'analysis': 'elx_polcurve', 'step': 'a1', 'plot_type': 'elx_v_vs_cycle'},
    {'key': 'NOPE', 'analysis': 'elx_polcurve', 'step': 'a1'},
])
check('two found, in order', [s['plot_type'] for s in out['series']],
      ['elx_polcurve', 'elx_j_vs_cycle'])
check('the rest reported with their selection',
      [(m.get('plot_type', ''), m['key'], m['reason'][:8]) for m in out['missing']],
      [('elx_v_vs_cycle', 'ELX', 'no store'), ('', 'NOPE', 'KeyError')])
check('no answer carries a title unless asked',
      [s for s in out['series'] if 'suptitle' in s], [])


def two_corrections():
    """A cell with current-dependent EIS at two cycles: an iR correction for
    each, named by cycle, plus the other cell's single one."""
    rec = elx_record()
    sc = record_sidecars(rec)
    for n in (3, 5):
        name = f'{PRE}ir_correction_cycle{n}_a1'
        rec['metrics']['elx_polcurve'][name] = {'conditions': {'step': 'a1'}, 'values': {}}
        sc[name] = dict(sidecar('elx_ir_correction', 'a1'))
        sc[name]['data'] = dict(sc[name]['data'], suptitle=f'iR Correction — Cycle {n}')
    sc[f'{PRE}ir_correction_a2']['data']['suptitle'] = 'iR Correction — Cycle 4'
    attach_sidecars(rec, sc)
    return rec


def record_sidecars(rec):
    from scripts.helpers.record import decode_sidecars
    return decode_sidecars(rec)


print('every plot of a type, when asked for every one')
use({'ELX': two_corrections(), 'FC': fc_record()})
ir = {'key': 'ELX', 'analysis': 'elx_polcurve', 'plot_type': 'elx_ir_correction',
      'every': True, 'title': True}
out = viewstore.plot_series_batch([dict(ir, step='a1'), dict(ir, step='a2'),
                                   dict(ir, step='a3'),
                                   {'key': 'ELX', 'analysis': 'elx_polcurve', 'step': 'a1',
                                    'plot_type': 'elx_ir_correction', 'title': True}])
check('three for a1 in stored order, one for a2, then the single ask',
      [(s['step'], s['plot'][len(PRE):]) for s in out['series']],
      [('a1', 'ir_correction_a1'), ('a1', 'ir_correction_cycle3_a1'),
       ('a1', 'ir_correction_cycle5_a1'), ('a2', 'ir_correction_a2'),
       ('a1', 'ir_correction_a1')])
check('each with its title, \'\' where the figure has none',
      [s['suptitle'] for s in out['series']],
      ['', 'iR Correction — Cycle 3', 'iR Correction — Cycle 5', 'iR Correction — Cycle 4', ''])
check('a cell with none is missing, once', [(m['step'], m['reason']) for m in out['missing']],
      [('a3', 'no stored plot data')])
out = viewstore.plot_series_batch([dict(ir, key='NOPE', step='a1'),
                                   dict(ir, step='a1', plot_type='')])
check('a sample that cannot be read is missing with why; no type, the main plot',
      ([m['reason'][:8] for m in out['missing']], [s['plot_type'] for s in out['series']]),
      (['KeyError'], ['elx_polcurve']))
check('every and title leave the series as they were',
      viewstore.plot_series_batch([dict(ir, step='a2')])['series'][0]['axes'],
      viewstore.plot_series('ELX', 'elx_polcurve', 'a2', plot_type='elx_ir_correction')['axes'])

# ════════════════════════════════════════════════════════════════════
print('the run list carries the registry')
res = viewstore.list_runs()
reg = res.get('registry') or {}
check('families, analyses and metrics', sorted(reg), ['analyses', 'families', 'metrics'])
check('the electrolyzer analyses with their main plots',
      reg['analyses']['elx_eis']['main_plots'], ['elx_nyquist', 'elx_eis_for_ir'])
check('step-less durability', reg['analyses']['durability'].get('stepless'), True)
hfr = next(m for m in reg['metrics']['elx_eis'] if m['name'] == 'HFR')
check('HFR stored in Ω·cm², shown in mΩ·cm²', (hfr['unit'], hfr['show'], hfr['scale']),
      ('Ω·cm²', 'mΩ·cm²', 1000))
v1 = {a: next(m['better'] for m in reg['metrics'][a] if m['name'] == 'V @ 1 A/cm²')
      for a in ('polcurve', 'elx_polcurve')}
check('one value name, opposite directions by analysis', v1,
      {'polcurve': 1, 'elx_polcurve': -1})
check('the runs are still there', res['total'], 2)
reg['analyses']['elx_eis']['main_plots'].append('changed')
check('what is sent is a copy: changing it changes nothing',
      registry.main_plots('elx_eis'), ['elx_nyquist', 'elx_eis_for_ir'])

# ════════════════════════════════════════════════════════════════════
print('the plot list says what each plot is')
use({'ELX': elx_record(drop=('-polcurve_a1',)), 'FC': fc_record(store_first=False)})
plots = {p['plot']: p for p in viewstore.run_plots('ELX')}
check('a stored plot carries its type',
      plots[f'{PRE}nyquist_a2']['plot_type'], 'elx_nyquist')
check('one without plot data has none, and is not renderable',
      (plots[f'{PRE}polcurve_a1']['plot_type'], plots[f'{PRE}polcurve_a1']['renderable']),
      ('', False))
check('every plot is listed, typed or not', len(plots), 2 * (len(POL) + len(EIS)) + len(DUR))
check('the fields Recall reads', sorted(plots[f'{PRE}model_fit_a1']),
      ['analysis', 'conditions', 'plot', 'plot_type', 'renderable', 'values'])
fc = {p['plot']: p['plot_type'] for p in viewstore.run_plots('FC')}
check('fuel-cell plots too', fc, {FC_PLOT: 'polcurve', FC_OVER: ''})

print('old per-plot-type buckets are named for the analysis they belong to')
store = Store({'ELX': elx_record(), 'FC': fc_record()})
store.bins['INDEX']['runs'].append({
    'sample_name': '260805_OLD', 'bin_id': 'OLD', 'timestamp': '2026-08-05T00:00:00Z',
    'Data': [{'Analysis': a, 'step': ''} for a in (
        'elx_model_fit', 'elx_nyquist', 'elx_eis_fit', 'elx_j_vs_cycle', 'elx_polcurve',
        'CLR', 'polcurve', 'unknown_thing', '')]})
jsonbin._request = store
viewstore.clear_cache()
res = viewstore.list_runs()
check('each to its analysis, by the rule the rebuild re-files them with',
      res['analysis_aliases'],
      {'elx_model_fit': 'elx_polcurve', 'elx_nyquist': 'elx_eis',
       'elx_eis_fit': 'elx_eis', 'elx_j_vs_cycle': 'elx_polcurve'})
check('analyses the registry lists, and ids it cannot place, are not aliased',
      [a for a in ('elx_polcurve', 'CLR', 'polcurve', 'unknown_thing', '')
       if a in res['analysis_aliases']], [])
check('worked out over the whole index, not the filtered runs',
      viewstore.list_runs(sample='FCS')['analysis_aliases'], res['analysis_aliases'])
use({'ELX': elx_record(), 'FC': fc_record()})
check('an index with none has none', viewstore.list_runs()['analysis_aliases'], {})
check('registry.aliases maps through the function it is given',
      registry.aliases(['x_pol', 'polcurve', 'x_eis', 'x_none'],
                       {'x_pol': 'polcurve', 'x_eis': 'elx_eis'}.get),
      {'x_pol': 'polcurve', 'x_eis': 'elx_eis'})

print('what Recall reads to group a run\'s plots')
from scripts.helpers.record import KEY_VALUES, plot_bucket          # noqa: E402
for a, e in registry.ANALYSES.items():
    for pt in list(e.get('other_plots') or {}) + list(e.get('main_plots') or []):
        check(f'{pt} is filed under {a}', plot_bucket(pt), a)
    check(f'{a}: no plot type is both main and folded',
          sorted(set(e.get('main_plots') or []) & set(e.get('other_plots') or {})), [])
check('the electrolyzer analyses fold their other plots under names',
      sorted(a for a, e in registry.ANALYSES.items() if e.get('other_plots')),
      ['durability', 'elx_eis', 'elx_polcurve'])
check('bookkeeping and condition fields are not metric sources',
      [(a, f) for a, e in registry.ANALYSES.items()
       for f in list(e.get('bookkeeping') or []) + list(e.get('summary_conditions') or {})
       if any(f in m['sources'] for m in registry.METRICS.get(a, []))], [])
check('values added for run detail only stay out of the index',
      [n for a in ('eis', 'elx_eis', 'durability') for n, _ in KEY_VALUES[a]],
      ['HFR', 'HFR', 'Degradation rate', 'V_initial', 'V_final', 'HFR_initial',
       'HFR_final', 'Duration'])
check('the fitted resistances shown with HFR, in its unit',
      [(m['name'], m['unit'], m['show'], m['scale'], m['index'])
       for m in registry.METRICS['eis'][1:]],
      [(n, 'Ω·cm²', 'mΩ·cm²', 1000, False) for n in ('R_ohm', 'R_ct', 'R_mt')])
from scripts.helpers import admin                                   # noqa: E402
check('the rebuild and the views name the same electrolyzer scripts',
      list(admin.ELECTROLYZER_SCRIPTS), registry.FAMILIES['elx']['scripts'])
check('every analysis is in a listed family',
      sorted({e['family'] for e in registry.ANALYSES.values()}), sorted(registry.FAMILIES))
sent = viewstore.list_runs()['registry']['analyses']
check('and the views get all of it', (sent['elx_eis']['other_plots']['elx_eis_fit'],
                                      sent['durability']['bookkeeping'][0],
                                      sent['durability']['summary_conditions']),
      ('per-spectrum fits', 'geo_area', {'j_hold': 'j_hold_Acm2'}))
check('with the families', viewstore.list_runs()['registry']['families']['elx']['label'],
      'Electrolyzer')

print(f'\n{passed} passed, {failed} failed')
sys.exit(1 if failed else 0)
