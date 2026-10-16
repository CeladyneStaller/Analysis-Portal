"""Electrolyzer runs — script output through to the index, the merge and a re-run.

Synthesizes potentiostatic electrolyzer pol curves (four cycles, a 1.25 V EIS
each cycle, current-dependent EIS on the last), runs the real run(), and drives
the output through the record layer and the real push path against an
in-memory JSONBin. The gate for the electrolyzer record work:

  1. one four-cycle run gives two index units, (elx_polcurve, a1) and
     (elx_eis, a1), carrying the expected values;
  2. two cells under one sample each keep their own plots and values;
  3. re-running a cell drops its stale per-spectrum plots and summary rows
     and leaves the other cell alone.

Also covered: records stored under the old rules (one bucket per plot type, no
cell on the plots or summary rows, a guessed stand) brought up to date by a
re-run; the declared conditions (temperature and pressure; hold current for
durability); the cell id reduced to a safe step; HFR taken from one fixed
spectrum; voltages from the last complete sweep; no stand guessed from .csv
files; and, under the size limit, the incoming run's plots and then main plots
kept ahead of the rest.

Run:  python3 scripts/helpers/test_elx_record.py
Exit code is non-zero if any assertion fails.
"""

import contextlib
import copy
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np                                                   # noqa: E402
from scipy.optimize import brentq                                    # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    from scripts.electrolyzer_polcurve import run as elx_run         # noqa: E402
    from scripts.electrolyzer_durability import run as dur_run       # noqa: E402
from scripts.helpers import jsonbin, registry                        # noqa: E402
from scripts.helpers.record import (                                 # noqa: E402
    build_detail_record, build_index_entry, decode_sidecars, load_sidecars,
    normalise_electrolyzer, normalise_step, plot_conditions, select_sidecars,
    sidecar_sizes, sidecar_step,
)

passed = failed = 0


def check(name, got, want):
    global passed, failed
    if got == want:
        passed += 1
    else:
        failed += 1
        print(f'  FAIL {name}\n       got  {got!r}\n       want {want!r}')


def close(name, got, want, tol):
    global passed, failed
    if got is not None and want is not None and abs(got - want) <= tol:
        passed += 1
    else:
        failed += 1
        print(f'  FAIL {name}: got {got!r}, want {want!r} +/- {tol}')


def true(name, cond, detail=''):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        print(f'  FAIL {name} {detail}')


# ── synthetic electrolyzer cell ─────────────────────────────────────
E_REV, AREA = 1.1837, 5.0
SETPOINTS = np.round(np.arange(1.40, 1.91, 0.05), 2)
SAMPLE, SCRIPT, SHORT = '260728_ELXT', 'Electrolyzer Pol Curve', 'ElxPolCurve'
NCYC = 4
# cell -> (ASR Ω·cm², kinetic offset V, ASR growth per cycle Ω·cm²)
CELLS = {'a1': (0.132, 0.006, 0.003), 'a2': (0.150, 0.012, 0.004)}


def v_of_j(j, c, cell):
    asr, kin, deg = CELLS[cell]
    return (E_REV + kin + 0.035 * np.log10(j / 1e-6) + 0.02 * np.log10(j / 1e-3)
            + j * (asr + deg * c) + 0.004 * j * j)


def j_of_v(v, c, cell):
    return brentq(lambda j: v_of_j(j, c, cell) - v, 1e-7, 30)


def expected_v_at(j_target, c, cell):
    """What the script should report: the sweep's (j, V) points, interpolated."""
    js = np.array([j_of_v(v, c, cell) for v in SETPOINTS])
    return float(np.interp(j_target, js, SETPOINTS))


def write_cell(folder, cell, current_dependent_eis=True, seed=0,
               last_sweep_max=None):
    """One cell's files as the bench writes them: 1_<cell>_polcurve.csv plus
    one CSV per EIS spectrum, named so scan_folder finds them by cell id.
    `last_sweep_max` stops the final sweep early, at that voltage."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    asr, _, deg = CELLS[cell]
    rows, t, eis = [], 0.0, []
    for c in range(NCYC):
        for _ in range(60):
            rows.append((t, 1.25 + rng.normal(0, 1e-4),
                         0.0005 * AREA + rng.normal(0, 1e-5), 1, c + 1,
                         'Constant Potential'))
            t += 1
        eis.append((t, 1.25, c, None))
        t += 120
        for r, v in enumerate(SETPOINTS):
            if last_sweep_max and c == NCYC - 1 and v > last_sweep_max + 1e-9:
                break
            jj = j_of_v(v, c, cell)
            for k in range(30):
                rows.append((t, v + rng.normal(0, 1e-4),
                             jj * AREA * (1 + 0.02 * np.exp(-k / 5))
                             + rng.normal(0, 1e-3), 2, r + 1, 'Constant Potential'))
                t += 1
            if (current_dependent_eis and c == NCYC - 1
                    and v in (1.50, 1.60, 1.70, 1.80)):
                eis.append((t, v, c, jj))
                t += 120
    with open(folder / f'1_{cell}_polcurve.csv', 'w') as fh:
        fh.write('Elapsed Time (s),Working Electrode (V),Current (A),'
                 'Step number,Repeats,Step name\n')
        for r in rows:
            fh.write(f'{r[0]:.1f},{r[1]:.5f},{r[2]:.5f},{r[3]},{r[4]},{r[5]}\n')
    f = np.logspace(5, -1, 61)
    w = 2 * np.pi * f
    for n, (te, dcv, c, jj) in enumerate(eis):
        rct = 0.03 if jj is None else 0.01 / max(jj, 0.05)
        z = ((asr * 0.2 + deg * c) / AREA + rct / (1 + 1j * w * rct * 0.1)
             + 0.008 / (1 + 1j * w * 0.008 * 2.0) + 1j * w * 3e-8)
        with open(folder / f'{n + 2}_{cell}_EIS_{dcv:.2f}V.csv', 'w') as fh:
            fh.write('Elapsed Time (s),Frequency (Hz),Z\' (Ohms),-Z" (Ohms),'
                     'DC Working Electrode (V)\n')
            for i in range(len(f)):
                fh.write(f'{te + i:.1f},{f[i]:.4f},{z[i].real:.8f},'
                         f'{-z[i].imag:.8f},{dcv + rng.normal(0, 2e-4):.5f}\n')


tmp = Path(tempfile.mkdtemp())
SRC = tmp / 'data' / SAMPLE                  # both cells, as one upload
RERUN = tmp / 'data_rerun' / SAMPLE          # cell a1 again, without the
write_cell(SRC, 'a1', seed=1)                # current-dependent EIS
write_cell(SRC, 'a2', seed=2)
write_cell(RERUN, 'a1', current_dependent_eis=False, seed=3)
TRUNC = tmp / 'data_trunc' / SAMPLE          # last sweep stopped at 1.70 V
write_cell(TRUNC, 'a1', current_dependent_eis=False, seed=4, last_sweep_max=1.70)


def run_cell(src, cell, job, p_cathode=30.0):
    """run() for one cell, then main._run_job's output rename."""
    out = tmp / job / 'output'
    out.mkdir(parents=True)
    with contextlib.redirect_stdout(io.StringIO()):
        res = elx_run(str(src), str(out), {
            'geo_area': str(AREA), 'T_C': '80', 'p_cathode_barg': str(p_cathode),
            'p_anode_barg': '0', 'cell_id': cell, 'eis_ref_voltage': '',
            'image_format': 'png', 'sample_name': SAMPLE})
    for f in sorted(out.rglob('*'), reverse=True):
        if f.is_file():
            f.rename(f.parent / f'{SAMPLE}-{SHORT}-{f.name}')
    files = [str(f.relative_to(src.parent)) for f in src.rglob('*.csv')]
    return out, res, files


DECLARED = {'T_C': 80.0, 'P_value': 30.0, 'P_unit': 'barg', 'P_anode_barg': 0.0}

# ════════════════════════════════════════════════════════════════════
print('1. one four-cycle run -> two units with the expected values')
out1, res1, files1 = run_cell(SRC, 'a1', 'job1')
check('status', res1.get('status'), 'success')
summary1 = res1.get('summary') or []
check('two summary rows, one per bucket',
      sorted(r.get('Analysis') for r in summary1), ['elx_eis', 'elx_polcurve'])
check('every row carries the cell id as its step',
      [r.get('step') for r in summary1], ['a1', 'a1'])
pol1 = next((r for r in summary1 if r.get('Analysis') == 'elx_polcurve'), {})
eis1 = next((r for r in summary1 if r.get('Analysis') == 'elx_eis'), {})

c_last = NCYC - 1
want_v1 = expected_v_at(1.0, c_last, 'a1')
want_v2 = expected_v_at(2.0, c_last, 'a1')
want_dv = (want_v2 - expected_v_at(2.0, 0, 'a1')) * 1000.0
close('V_at_1Acm2 from the last cycle', pol1.get('V_at_1Acm2'), want_v1, 0.002)
close('V_at_2Acm2 from the last cycle', pol1.get('V_at_2Acm2'), want_v2, 0.002)
close('dV_at_2Acm2_mV, first cycle to last', pol1.get('dV_at_2Acm2_mV'),
      want_dv, 2.0)
true('the cell degrades, so dV is positive', (pol1.get('dV_at_2Acm2_mV') or 0) > 10)
close('HFR in the summary is R0 in Ω·cm²', eis1.get('HFR'),
      (eis1.get('R0_asr_mohm_cm2') or 0) / 1000.0, 1e-9)

outputs = [f for f in out1.rglob('*') if f.is_file()]
true('the cell id is in every output filename',
     outputs and all(Path(f.name).stem.endswith('_a1') for f in outputs),
     f"(not: {[f.name for f in outputs if not Path(f.name).stem.endswith('_a1')]})")
sidecars1 = load_sidecars(out1)
true('plots were written', len(sidecars1) >= 12, f'(got {len(sidecars1)})')
decl = [sc.get('metadata', {}).get('conditions') for sc in sidecars1.values()]
check('every sidecar declares the cell and its conditions',
      all(d == {**DECLARED, 'step': 'a1'} for d in decl), True)
types1 = {sc.get('plot_type') for sc in sidecars1.values()}
true('CLR plots are typed elx_clr_*',
     {'elx_clr_analysis', 'elx_clr_model_fit'} <= types1, f'(types {sorted(types1)})')
true('no fuel-cell CLR type left', not any(t.startswith('CLR') for t in types1))

rec1 = build_detail_record(job_id='job1', sample_name=SAMPLE, script=SCRIPT,
                           timestamp='2026-10-06T00:00:00Z', input_files=files1,
                           output_dir=out1, summary=summary1)
check('two buckets, nothing stray', sorted(rec1['metrics']),
      ['elx_eis', 'elx_polcurve'])
check('the CLR plots are filed with the EIS',
      {sidecars1[n]['plot_type'] for n in rec1['metrics']['elx_eis']}
      >= {'elx_clr_analysis', 'elx_clr_model_fit'}, True)
steps = {e['conditions'].get('step') for p in rec1['metrics'].values()
         for e in p.values()}
check('every plot is filed under the cell', steps, {'a1'})
true('no voltage setpoint misread from eis_fit_V_dc1.250V',
     not any('V_setpoint' in e['conditions'] for p in rec1['metrics'].values()
             for e in p.values()))

entry1 = build_index_entry(rec1, 'BIN1')
units = {(u['Analysis'], u['step']): u for u in entry1['Data']}
check('two units', sorted(units), [('elx_eis', 'a1'), ('elx_polcurve', 'a1')])
kv_pol = units.get(('elx_polcurve', 'a1'), {}).get('key_values') or {}
kv_eis = units.get(('elx_eis', 'a1'), {}).get('key_values') or {}
check('pol-curve values promoted', sorted(kv_pol),
      sorted(['V @ 1 A/cm²', 'V @ 2 A/cm²', 'ΔV @ 2 A/cm²']))
close('index V @ 1 A/cm²', kv_pol.get('V @ 1 A/cm²'), want_v1, 0.002)
close('index V @ 2 A/cm²', kv_pol.get('V @ 2 A/cm²'), want_v2, 0.002)
close('index ΔV @ 2 A/cm²', kv_pol.get('ΔV @ 2 A/cm²'), want_dv, 2.0)
close('index HFR is the summary HFR', kv_eis.get('HFR'), eis1.get('HFR'), 1e-6)
true('fit-derived values stay out of the index',
     not ({'ASR_total', 'Tafel slope (anode)', 'E_rev', 'Fit RMSE'} & set(kv_pol)))
check('units carry the declared conditions',
      [units[k].get('Conditions') for k in sorted(units)], [DECLARED, DECLARED])
check('no stand guessed from .csv files', entry1.get('stand'), None)

# ════════════════════════════════════════════════════════════════════
print('2. two cells under one sample keep their own plots and values')


class MemStore:
    """JSONBin in memory: POST creates a bin, PUT replaces, GET reads."""

    def __init__(self):
        self.bins = {'INDEX': {'schema': 2, 'runs': []}}
        self.n = 0

    def __call__(self, url, method='GET', body=None, extra_headers=None):
        if method == 'POST':
            self.n += 1
            bid = f'BIN{self.n}'
            self.bins[bid] = copy.deepcopy(jsonbin.json_safe(body))
            return {'metadata': {'id': bid}}
        if method == 'PUT':
            self.bins[url.rstrip('/').split('/')[-1]] = copy.deepcopy(
                jsonbin.json_safe(body))
            return {}
        bid = url.rstrip('/').split('/')[-2]
        if bid not in self.bins:
            raise RuntimeError(f'404 {bid}')
        return {'record': copy.deepcopy(self.bins[bid])}

    def entry(self):
        runs = [e for e in self.bins['INDEX']['runs']
                if e.get('sample_name') == SAMPLE]
        return runs

    def detail(self):
        return self.bins[self.entry()[0]['bin_id']]


store = MemStore()
jsonbin._request = store
os.environ.update(JSONBIN_API_KEY='k', JSONBIN_COLLECTION_ID='C',
                  JSONBIN_INDEX_BIN_ID='INDEX')


def push(job, out, res, files):
    return jsonbin.push_job_metrics(
        job_id=job, sample_name=SAMPLE, script=SCRIPT, output_dir=out,
        input_files=files, summary=res.get('summary'), script_short=SHORT)


r1 = push('job1', out1, res1, files1)
true('first cell pushed', r1['pushed'], f"({r1['reason']})")
out2, res2, files2 = run_cell(SRC, 'a2', 'job2')
r2 = push('job2', out2, res2, files2)
true('second cell pushed and merged', r2['pushed'] and r2['merged'],
     f"({r2['reason']})")
check('one index entry for the sample', len(store.entry()), 1)

entry = store.entry()[0]
units = {(u['Analysis'], u['step']): u.get('key_values') or {}
         for u in entry['Data']}
check('a unit per bucket per cell', sorted(units),
      [('elx_eis', 'a1'), ('elx_eis', 'a2'),
       ('elx_polcurve', 'a1'), ('elx_polcurve', 'a2')])
close('a1 keeps its own V @ 2 A/cm²',
      units.get(('elx_polcurve', 'a1'), {}).get('V @ 2 A/cm²'), want_v2, 0.002)
close('a2 has its own V @ 2 A/cm²',
      units.get(('elx_polcurve', 'a2'), {}).get('V @ 2 A/cm²'),
      expected_v_at(2.0, c_last, 'a2'), 0.002)
true('and the two differ',
     abs((units.get(('elx_polcurve', 'a1'), {}).get('V @ 2 A/cm²') or 0)
         - (units.get(('elx_polcurve', 'a2'), {}).get('V @ 2 A/cm²') or 0)) > 0.01)

names1, names2 = set(sidecars1), set(load_sidecars(out2))
detail = store.detail()
sc = decode_sidecars(detail)
check('neither cell overwrote the other', set(sc), names1 | names2)
true('the two cells wrote distinct plot names', not (names1 & names2))
by_step = {}
for b, plots in detail['metrics'].items():
    for name, e in plots.items():
        by_step.setdefault(e['conditions'].get('step'), set()).add(name)
check('metrics hold both cells in full', (by_step.get('a1'), by_step.get('a2')),
      (names1, names2))
check('summary rows for both cells',
      sorted((r['Analysis'], r['step']) for r in detail['summary']),
      [('elx_eis', 'a1'), ('elx_eis', 'a2'),
       ('elx_polcurve', 'a1'), ('elx_polcurve', 'a2')])

# ════════════════════════════════════════════════════════════════════
print('3. re-running a cell replaces only that cell')
out3, res3, files3 = run_cell(RERUN, 'a1', 'job3')
r3 = push('job3', out3, res3, files3)
true('re-run pushed', r3['pushed'], f"({r3['reason']})")
names3 = set(load_sidecars(out3))
stale = names1 - names3
true('the re-run really has fewer per-spectrum plots', len(stale) >= 5,
     f'(stale {sorted(stale)})')
detail = store.detail()
sc = decode_sidecars(detail)
a1_now = {n for n, s in sc.items() if sidecar_step(n, s) == 'a1'}
check("a1's sidecars are exactly the re-run's", a1_now, names3)
check('no stale per-spectrum sidecar survives', sorted(stale & set(sc)), [])
true("a2's sidecars are untouched", names2 <= set(sc))
a1_metrics = {n for p in detail['metrics'].values() for n, e in p.items()
              if e['conditions'].get('step') == 'a1'}
check("a1's metrics are exactly the re-run's", a1_metrics, names3)
rows = sorted((r['Analysis'], r['step']) for r in detail['summary'])
check("a1's summary rows replaced, not piled up", rows,
      [('elx_eis', 'a1'), ('elx_eis', 'a2'),
       ('elx_polcurve', 'a1'), ('elx_polcurve', 'a2')])
pol3 = next(r for r in res3['summary'] if r['Analysis'] == 'elx_polcurve')
entry = store.entry()[0]
units = {(u['Analysis'], u['step']): u.get('key_values') or {}
         for u in entry['Data']}
close("a1's index value is the re-run's",
      units.get(('elx_polcurve', 'a1'), {}).get('V @ 2 A/cm²'),
      pol3['V_at_2Acm2'], 1e-5)
check('still four units', len(units), 4)
check('three jobs recorded', entry.get('n_jobs'), 3)

print('HFR comes from one fixed spectrum')
eis3 = next(r for r in res3['summary'] if r['Analysis'] == 'elx_eis')
close('the reference spectrum at 1.25 V', eis1.get('HFR_dc_V'), 1.25, 0.01)
close('HFR is the same whether or not current-dependent EIS was measured',
      eis3.get('HFR'), eis1.get('HFR'), 1e-9)
asr_a1, _, deg_a1 = CELLS['a1']
close('and close to the cell\'s series resistance in the last cycle',
      eis1.get('HFR'), asr_a1 * 0.2 + deg_a1 * c_last, 0.0015)

# ════════════════════════════════════════════════════════════════════
print('a declared step is enough on its own, without the cell id in the name')
# Runs pushed before the filenames carried the cell id, and any script that
# declares a step without encoding it, rely on the declaration alone. The
# merge and the push carry-over must both read it, or a re-run keeps the
# stale sidecars its metrics already displaced.
S2 = '260801_ELXS'


def fake_sidecar(plot_type, declared):
    return {'plot_type': plot_type, 'metadata': {'conditions': declared},
            'data': {'axes': [{'texts': [], 'axhlines': [], 'axvlines': [],
                               'lines': []}]}}


def fake_output(job, names, sample=S2, cell='a1'):
    out = tmp / job / 'output'
    d = out / '_plot_data'
    d.mkdir(parents=True)
    for name, plot_type in names:
        (d / f'{sample}-{SHORT}-{name}.json').write_text(json.dumps(
            fake_sidecar(plot_type, {'step': cell, **DECLARED})))
    return out


first = [('polcurve', 'elx_polcurve'), ('nyquist', 'elx_nyquist'),
         ('eis_fit_V_dc1.250V', 'elx_eis_fit'), ('eis_fit_V_dc1.500V', 'elx_eis_fit')]
again = [('polcurve', 'elx_polcurve'), ('nyquist', 'elx_nyquist'),
         ('eis_fit_V_dc1.250V', 'elx_eis_fit')]
for job, names in (('fake1', first), ('fake2', again)):
    r = jsonbin.push_job_metrics(
        job_id=job, sample_name=S2, script=SCRIPT,
        output_dir=fake_output(job, names), input_files=['1_a1_polcurve.csv'],
        summary=[{'Analysis': 'elx_eis', 'HFR': 0.03, 'step': 'a1'}],
        script_short=SHORT)
    true(f'{job} pushed', r['pushed'], f"({r['reason']})")
e2 = next(e for e in store.bins['INDEX']['runs'] if e['sample_name'] == S2)
d2 = store.bins[e2['bin_id']]
check('only the re-run\'s sidecars remain', sorted(decode_sidecars(d2)),
      sorted(f'{S2}-{SHORT}-{n}' for n, _ in again))
check('and only its metrics', sorted(n for p in d2['metrics'].values() for n in p),
      sorted(f'{S2}-{SHORT}-{n}' for n, _ in again))
check('one summary row, not two', len(d2.get('summary') or []), 1)

# ════════════════════════════════════════════════════════════════════
print('records stored under the old rules are brought up to date')
# What a run pushed between Aug 4 and this change left behind: one bucket per
# plot type, no cell on the plots (their sidecars declare it), a voltage
# setpoint misread from an EIS fit's name, summary rows that name no step and
# have piled up across runs, and a stand guessed from .csv files.
OLD = [('polcurve', 'elx_polcurve', 'elx_polcurve'),
       ('j_vs_cycle', 'elx_j_vs_cycle', 'elx_j_vs_cycle'),
       ('model_fit', 'elx_model_fit', 'elx_model_fit'),
       ('nyquist', 'elx_nyquist', 'elx_nyquist'),
       ('eis_fit_V_dc1.250V', 'elx_eis_fit', 'elx_eis_fit'),
       ('CLR_analysis', 'CLR_analysis', 'CLR')]


def legacy_record(sample, hfrs=(0.040, 0.041), v2=1.79, declares=True):
    """A record as the old code wrote it. `declares=False` is a run from
    before the scripts declared their cell at all (before Jul 27)."""
    sidecars, metrics = {}, {}
    for stem, plot_type, bucket in OLD:
        name = f'{sample}-{SHORT}-{stem}'
        sidecars[name] = fake_sidecar(plot_type, {'step': 'a1'})
        if not declares:
            sidecars[name].pop('metadata')
        metrics.setdefault(bucket, {})[name] = {
            'conditions': {'V_setpoint': 250.0} if 'V_dc' in stem else {},
            'values': {}}
    rec = {'schema': 2, 'job_id': 'old', 'sample_name': sample, 'script': SCRIPT,
           'timestamp': '2026-08-20T00:00:00Z', 'input_files': ['1_a1_polcurve.csv'],
           'metrics': metrics, 'conditions': None,
           'summary': [{'Analysis': 'elx_polcurve', 'V_at_1Acm2': 1.61,
                        'V_at_2Acm2': v2}]
           + [{'Analysis': 'elx_eis', 'HFR': h} for h in hfrs]}
    jsonbin.attach_sidecars(rec, sidecars)
    return rec, set(sidecars)


def legacy_sample(sample, bin_id, **kw):
    rec, names = legacy_record(sample, **kw)
    store.bins[bin_id] = copy.deepcopy(jsonbin.json_safe(rec))
    entry = build_index_entry(rec, bin_id)
    entry['n_jobs'] = 2
    store.bins['INDEX']['runs'].append(entry)
    return names, entry


def push_fake(sample, job, cell, names, summary):
    return jsonbin.push_job_metrics(
        job_id=job, sample_name=sample, script=SCRIPT,
        output_dir=fake_output(job, names, sample, cell),
        input_files=[f'1_{cell}_polcurve.csv'], summary=summary,
        script_short=SHORT)


def units_of(sample):
    e = next(e for e in store.bins['INDEX']['runs'] if e['sample_name'] == sample)
    return e, {(u['Analysis'], u['step']): u.get('key_values') or {}
               for u in e['Data']}


# Re-running the same cell replaces everything the old runs stored.
S4 = '260820_ELXL'
old_names, old_entry = legacy_sample(S4, 'LEG4')
check('fixture: the old entry carries a guessed stand', old_entry.get('stand'), 'FCTS')
new = [('polcurve_a1', 'elx_polcurve'), ('model_fit_a1', 'elx_model_fit'),
       ('nyquist_a1', 'elx_nyquist')]
r = push_fake(S4, 'leg4', 'a1', new,
              [{'Analysis': 'elx_polcurve', 'V_at_1Acm2': 1.60, 'V_at_2Acm2': 1.78,
                'step': 'a1'},
               {'Analysis': 'elx_eis', 'HFR': 0.035, 'step': 'a1'}])
true('re-run pushed', r['pushed'], f"({r['reason']})")
d4 = store.bins['LEG4']
new_names = {f'{S4}-{SHORT}-{n}' for n, _ in new}
check('no old sidecar survives', set(decode_sidecars(d4)), new_names)
check('no old metrics entry survives',
      {n for p in d4['metrics'].values() for n in p}, new_names)
check('only the two current buckets', sorted(d4['metrics']),
      ['elx_eis', 'elx_polcurve'])
check('only the re-run\'s summary rows',
      sorted((x['Analysis'], x.get('HFR') or 0) for x in d4['summary']),
      [('elx_eis', 0.035), ('elx_polcurve', 0)])
e4, u4 = units_of(S4)
check('index units are the cell\'s two, nothing left over', sorted(u4),
      [('elx_eis', 'a1'), ('elx_polcurve', 'a1')])
close('with the re-run\'s HFR', u4.get(('elx_eis', 'a1'), {}).get('HFR'), 0.035, 1e-9)
check('the guessed stand is gone', e4.get('stand'), None)

# Another cell's run re-files the old cell's plots and rows under it.
S5 = '260821_ELXM'
old_names, _ = legacy_sample(S5, 'LEG5')
new = [('polcurve_a2', 'elx_polcurve'), ('nyquist_a2', 'elx_nyquist')]
r = push_fake(S5, 'leg5', 'a2', new,
              [{'Analysis': 'elx_polcurve', 'V_at_1Acm2': 1.70, 'V_at_2Acm2': 1.85,
                'step': 'a2'},
               {'Analysis': 'elx_eis', 'HFR': 0.050, 'step': 'a2'}])
true('other cell pushed', r['pushed'], f"({r['reason']})")
d5 = store.bins['LEG5']
check('every old plot kept, alongside the new',
      set(decode_sidecars(d5)), old_names | {f'{S5}-{SHORT}-{n}' for n, _ in new})
check('filed under the two current buckets', sorted(d5['metrics']),
      ['elx_eis', 'elx_polcurve'])
check('the old plots now carry their cell',
      {d5_e['conditions'].get('step') for p in d5['metrics'].values()
       for n, d5_e in p.items() if n in old_names}, {'a1'})
check('the CLR plot is with the electrolyzer EIS',
      f'{S5}-{SHORT}-CLR_analysis' in d5['metrics']['elx_eis'], True)
true('the misread setpoint is gone',
     not any('V_setpoint' in x['conditions'] for p in d5['metrics'].values()
             for x in p.values()))
check('old rows carry their cell, and the piled-up duplicate is gone',
      sorted((x['Analysis'], x.get('step', ''), x.get('HFR') or 0) for x in d5['summary']),
      [('elx_eis', 'a1', 0.041), ('elx_eis', 'a2', 0.05),
       ('elx_polcurve', 'a1', 0), ('elx_polcurve', 'a2', 0)])
e5, u5 = units_of(S5)
check('index units for both cells, nothing left over', sorted(u5),
      [('elx_eis', 'a1'), ('elx_eis', 'a2'),
       ('elx_polcurve', 'a1'), ('elx_polcurve', 'a2')])
close('the old cell keeps its latest HFR', u5.get(('elx_eis', 'a1'), {}).get('HFR'),
      0.041, 1e-9)
close('and its voltage', u5.get(('elx_polcurve', 'a1'), {}).get('V @ 2 A/cm²'), 1.79, 1e-9)
close('the new cell has its own', u5.get(('elx_eis', 'a2'), {}).get('HFR'), 0.050, 1e-9)

# A new cell without an EIS row takes no other cell's.
S6 = '260822_ELXN'
legacy_sample(S6, 'LEG6')
push_fake(S6, 'leg6', 'a2', [('polcurve_a2', 'elx_polcurve'), ('nyquist_a2', 'elx_nyquist')],
          [{'Analysis': 'elx_polcurve', 'V_at_2Acm2': 1.85, 'step': 'a2'}])
_, u6 = units_of(S6)
check('a cell whose EIS fits failed shows no HFR, not another cell\'s',
      'HFR' in u6.get(('elx_eis', 'a2'), {}), False)
# The same when the other cell's row could not be given a cell at all: in a
# record whose rows name their steps, an undeclared electrolyzer row is
# attached to no cell rather than to whichever asks first.
from scripts.helpers.record import _rows_for_unit                   # noqa: E402
check('an undeclared row is not handed to a cell',
      _rows_for_unit([{'Analysis': 'elx_eis', 'HFR': 0.04},
                      {'Analysis': 'elx_polcurve', 'V_at_2Acm2': 1.8, 'step': 'a2'}],
                     'elx_eis', [], 'a2'), [])
check('an old row with no step goes to its record\'s one cell',
      _rows_for_unit([{'Analysis': 'elx_eis', 'HFR': 0.04}], 'elx_eis', [], 'a1',
                     sole='a1'), [{'Analysis': 'elx_eis', 'HFR': 0.04}])
check('and to no cell when the record has several',
      _rows_for_unit([{'Analysis': 'elx_eis', 'HFR': 0.04}], 'elx_eis', [], 'a1'),
      [])

# A run that names no cell is a cell of its own, not a gap to fill. Adding a
# named cell and re-running it must leave the unnamed one exactly as it was.
S7 = '260823_ELXU'
blank = [('polcurve', 'elx_polcurve'), ('nyquist', 'elx_nyquist')]
named = [('polcurve_a1', 'elx_polcurve'), ('nyquist_a1', 'elx_nyquist')]
push_fake(S7, 'u1', '', blank,
          [{'Analysis': 'elx_polcurve', 'V_at_2Acm2': 1.822, 'step': ''},
           {'Analysis': 'elx_eis', 'HFR': 0.0415, 'step': ''}])
for job, v2, hfr in (('u2', 1.78, 0.035), ('u3', 1.77, 0.034)):
    push_fake(S7, job, 'a1', named,
              [{'Analysis': 'elx_polcurve', 'V_at_2Acm2': v2, 'step': 'a1'},
               {'Analysis': 'elx_eis', 'HFR': hfr, 'step': 'a1'}])
e7, u7 = units_of(S7)
check('the unnamed cell keeps its units beside the named one', sorted(u7),
      [('elx_eis', ''), ('elx_eis', 'a1'), ('elx_polcurve', ''), ('elx_polcurve', 'a1')])
close('and its own voltage', u7.get(('elx_polcurve', ''), {}).get('V @ 2 A/cm²'),
      1.822, 1e-9)
close('and its own HFR', u7.get(('elx_eis', ''), {}).get('HFR'), 0.0415, 1e-9)
close('the named cell has its re-run values',
      u7.get(('elx_eis', 'a1'), {}).get('HFR'), 0.034, 1e-9)
d7 = store.bins[e7['bin_id']]
check('the unnamed plots are still filed without a cell',
      sorted(e['conditions'].get('step', '') for p in d7['metrics'].values()
             for n, e in p.items() if not n.endswith('_a1')), ['', ''])
check('and both of its summary rows are kept',
      sorted((x['Analysis'], x.get('step')) for x in d7['summary']),
      [('elx_eis', ''), ('elx_eis', 'a1'), ('elx_polcurve', ''), ('elx_polcurve', 'a1')])

# Data from before the scripts declared their cell is no cell's: a new run
# beside it gives it nothing, and it keeps its values under no step.
S8 = '260705_ELXP'
legacy_sample(S8, 'LEG8', declares=False, hfrs=(0.044,))
for job in ('p1', 'p2'):            # the second push merges into a record
    push_fake(S8, job, 'a1', named,  # that already holds a current-rules cell
              [{'Analysis': 'elx_polcurve', 'V_at_2Acm2': 1.78, 'step': 'a1'},
               {'Analysis': 'elx_eis', 'HFR': 0.035, 'step': 'a1'}])
_, u8 = units_of(S8)
d8 = store.bins['LEG8']
check('older undeclared plots are not given the new cell',
      {e['conditions'].get('step', '') for p in d8['metrics'].values()
       for n, e in p.items() if not n.endswith('_a1')}, {''})
check('its rows are kept, marked as naming no cell',
      [x.get('step') for x in d8['summary']
       if x.get('HFR') == 0.044 or x.get('V_at_2Acm2') == 1.79], ['', ''])
close('it keeps its HFR under no step', u8.get(('elx_eis', ''), {}).get('HFR'),
      0.044, 1e-9)

# Old records the rules cannot read unambiguously: nothing in them is given
# to a cell, and nothing is deleted. Each is what the old code stored after
# two runs, followed by an unrelated push that brings it up to date.


def legacy_custom(sample, bin_id, plots, rows):
    sidecars, metrics = {}, {}
    for stem, plot_type, bucket, declared in plots:
        name = f'{sample}-{SHORT}-{stem}'
        sidecars[name] = fake_sidecar(plot_type, declared)
        if not declared:
            sidecars[name].pop('metadata')
        metrics.setdefault(bucket, {})[name] = {'conditions': {}, 'values': {}}
    rec = {'schema': 2, 'job_id': 'old', 'sample_name': sample, 'script': SCRIPT,
           'timestamp': '2026-08-20T00:00:00Z', 'input_files': ['1_a1_polcurve.csv'],
           'metrics': metrics, 'conditions': None, 'summary': rows}
    jsonbin.attach_sidecars(rec, sidecars)
    store.bins[bin_id] = copy.deepcopy(jsonbin.json_safe(rec))
    store.bins['INDEX']['runs'].append(build_index_entry(rec, bin_id))


def durability_push(sample, job):
    out = tmp / job / 'output'
    d = out / '_plot_data'
    d.mkdir(parents=True)
    (d / f'{sample}-ElxDurability-voltage_vs_time.json').write_text(json.dumps(
        fake_sidecar('durability_voltage', {'j_hold_Acm2': 1.0})))
    return jsonbin.push_job_metrics(
        job_id=job, sample_name=sample, script='Electrolyzer Durability',
        output_dir=out, input_files=['01/data.csv'], script_short='ElxDurability',
        summary=[{'Analysis': 'durability', 'Degradation rate': 30.0}])


# Cell a1, then cell a2: their names collided, so a2's plots replaced a1's,
# except a1's iR correction; a2's pol-curve row displaced a1's, and the EIS
# rows of both piled up. Which row is whose cannot be told.
S11 = '260825_ELXA'
two_cell_rows = [{'Analysis': 'elx_eis', 'HFR': 0.0313},
                 {'Analysis': 'elx_polcurve', 'V_at_2Acm2': 1.822},
                 {'Analysis': 'elx_eis', 'HFR': 0.0415}]
legacy_custom(S11, 'LEG11', [
    ('polcurve', 'elx_polcurve', 'elx_polcurve', {'step': 'a2'}),
    ('nyquist', 'elx_nyquist', 'elx_nyquist', {'step': 'a2'}),
    ('ir_correction', 'elx_ir_correction', 'elx_ir_correction', {'step': 'a1'})],
    copy.deepcopy(two_cell_rows))
r = durability_push(S11, 'leg11')
true('unrelated push to a two-cell old record', r['pushed'], f"({r['reason']})")
_, u11 = units_of(S11)
check('both cells have units', sorted(k for k in u11 if k[0] != 'durability'),
      [('elx_eis', 'a2'), ('elx_polcurve', 'a1'), ('elx_polcurve', 'a2')])
check('neither is handed a row it may not own',
      [u11[k] for k in sorted(u11) if k[0] != 'durability'], [{}, {}, {}])
check('and every old row is kept as it was',
      [x for x in store.bins['LEG11']['summary'] if x['Analysis'] != 'durability'],
      two_cell_rows)

# Cell a1, then a run that named no cell: the same collision, a1 keeping
# only its iR correction, and the cell-less run's rows beside a1's EIS row.
S12 = '260826_ELXB'
cellless_rows = [{'Analysis': 'elx_eis', 'HFR': 0.0313},
                 {'Analysis': 'elx_polcurve', 'V_at_2Acm2': 1.822},
                 {'Analysis': 'elx_eis', 'HFR': 0.0415}]
legacy_custom(S12, 'LEG12', [
    ('polcurve', 'elx_polcurve', 'elx_polcurve', None),
    ('nyquist', 'elx_nyquist', 'elx_nyquist', None),
    ('ir_correction', 'elx_ir_correction', 'elx_ir_correction', {'step': 'a1'})],
    copy.deepcopy(cellless_rows))
durability_push(S12, 'leg12')
_, u12 = units_of(S12)
check('the cell and the cell-less run both have units',
      sorted(k for k in u12 if k[0] != 'durability'),
      [('elx_eis', ''), ('elx_polcurve', ''), ('elx_polcurve', 'a1')])
check('a1 is not given the cell-less run\'s values',
      u12.get(('elx_polcurve', 'a1')), {})
check('and its own EIS row is not deleted',
      [x for x in store.bins['LEG12']['summary'] if x['Analysis'] != 'durability'],
      cellless_rows)
close('beside the new cell\'s', u8.get(('elx_eis', 'a1'), {}).get('HFR'), 0.035, 1e-9)

print('stored records merged into each other')
# Duplicate merging and migrate_samples pass a stored record as the incoming
# side. Two old runs of one cell: the newer replaces the older entirely.
older, _ = legacy_record('260824_ELXD', hfrs=(0.0313,), v2=1.80)
newer, newer_names = legacy_record('260824_ELXD', hfrs=(0.0352,), v2=1.78)
from scripts.helpers.record import merge_detail_record              # noqa: E402
m = merge_detail_record(older, newer)
check('two current buckets', sorted(m['metrics']), ['elx_eis', 'elx_polcurve'])
names_filed = [n for p in m['metrics'].values() for n in p]
check('each plot filed once', len(names_filed), len(set(names_filed)))
check('the newer run\'s rows only',
      sorted((x['Analysis'], x.get('HFR')) for x in m['summary']
             if x['Analysis'] == 'elx_eis'), [('elx_eis', 0.0352)])
um = {(u['Analysis'], u['step']): u.get('key_values') or {}
      for u in build_index_entry(m, 'BM')['Data']}
close('and the index takes the newer HFR', um.get(('elx_eis', 'a1'), {}).get('HFR'),
      0.0352, 1e-9)

print('fuel-cell index entries are kept as they were')
# A step-less OCV run, then one at b5: the index has always kept the step-less
# unit, whose plot is still in the bin, beside the new one.
S10 = '260710_FCS9'


def fc_output(job, names):
    out = tmp / job / 'output'
    d = out / '_plot_data'
    d.mkdir(parents=True)
    for name in names:
        (d / f'{S10}-OCV-{name}.json').write_text(json.dumps({
            'plot_type': 'ocv', 'data': {'axes': [{'texts': [], 'axhlines': [],
                                                   'axvlines': [], 'lines': []}]}}))
    return out


for job, names in (('fc1', ['ocv_GSMA']), ('fc2', ['ocv_b5_GSMA_80C'])):
    jsonbin.push_job_metrics(job_id=job, sample_name=S10, script='OCV Analysis',
                             output_dir=fc_output(job, names),
                             input_files=['x.fcd'], script_short='OCV')
_, u10 = units_of(S10)
check('both OCV units stay in the index', sorted(u10), [('ocv', ''), ('ocv', 'b5')])

# ════════════════════════════════════════════════════════════════════
print('under the size limit the incoming run comes first, then main plots')
rng = np.random.default_rng(11)


def noisy(plot_type, n):
    return {'plot_type': plot_type, 'data': {'axes': [{'lines': [{
        'x': [round(float(v), 6) for v in rng.normal(size=n)],
        'y': [round(float(v), 6) for v in rng.normal(size=n)]}]}]}}


stored_main = {f'old_main_{i}': noisy('elx_polcurve', 3000) for i in range(2)}
incoming = {f'new_fit_{i}': noisy('elx_eis_fit', 150) for i in range(10)}
both = {**stored_main, **incoming}
sz = sidecar_sizes(both)
budget = sum(sz[n] for n in incoming) + sz['old_main_0'] - 1
kept, dropped = select_sidecars(both, budget, exclude_buckets=set(),
                                protected=set(incoming))
check('selection keeps every incoming plot', set(incoming) <= set(kept), True)
true('and gives up a stored main plot instead',
     any(n in dropped for n in stored_main))

# Eviction can overshoot: the small fuel-cell plot goes first (it is not a
# main plot), then the large main plot has to go anyway. The small one must
# be offered back. The limit is set so selection admits both (it budgets 4x
# the remaining bytes, raw) while the large one cannot fit compressed.
small_fc = {'fc_polcurve': noisy('polcurve', 40)}
big_main = {'elx_cycles': noisy('elx_j_vs_cycle', 4000)}
rec_fc = {'schema': 2, 'job_id': 'x', 'sample_name': 'X', 'metrics': {}}
base_bytes = len(json.dumps(rec_fc, ensure_ascii=False).encode())
raw = sidecar_sizes({**small_fc, **big_main})
limit = base_bytes + (raw['fc_polcurve'] + raw['elx_cycles']) // 4 + 50
probe = copy.deepcopy(rec_fc)
jsonbin.attach_sidecars(probe, big_main)
true('fixture: the large plot cannot fit',
     len(json.dumps(probe, ensure_ascii=False).encode()) > limit)
saved = jsonbin.MAX_BODY_BYTES
try:
    jsonbin.MAX_BODY_BYTES = limit
    _, kept, dropped, body = jsonbin.fit_record_to_budget(
        copy.deepcopy(rec_fc), {**small_fc, **big_main},
        protected={'fc_polcurve', 'elx_cycles'})
finally:
    jsonbin.MAX_BODY_BYTES = saved
check('a plot evicted on the way is offered back when it fits',
      sorted(kept), ['fc_polcurve'])
check('only the plot that cannot fit is reported dropped', dropped, ['elx_cycles'])
true('within the limit', body <= limit, f'({body} > {limit})')

# ════════════════════════════════════════════════════════════════════
print('the cell id becomes a safe step')
check('markup characters collapse', normalise_step("a1');alert(1);('"), 'a1-alert-1')
check('case and spaces', normalise_step(' Cell 7 '), 'cell-7')
check('nothing usable', normalise_step('#%'), '')
check('a declared step is normalised by the record layer too',
      plot_conditions('p', {'metadata': {'conditions': {'step': '<b>X'}}})['step'],
      'b-x')
outh, resh, _ = run_cell(RERUN, "A1');x(", 'job_hostile')
check('a hostile cell id runs', resh.get('status'), 'success')
sch = load_sidecars(outh)
check('every sidecar declares the reduced id',
      {s['metadata']['conditions']['step'] for s in sch.values()}, {'a1-x'})
true('and every output name carries it',
     all(Path(f.name).stem.endswith('_a1-x') for f in outh.rglob('*') if f.is_file()))
check('as does every summary row', {x.get('step') for x in resh['summary']}, {'a1-x'})

# ════════════════════════════════════════════════════════════════════
print('a final sweep cut short does not set the voltages')
outt, rest, _ = run_cell(TRUNC, 'a1', 'job_trunc')
polt = next(x for x in rest['summary'] if x['Analysis'] == 'elx_polcurve')
close('V @ 1 A/cm² from the last complete sweep', polt.get('V_at_1Acm2'),
      expected_v_at(1.0, NCYC - 2, 'a1'), 0.002)
close('V @ 2 A/cm² from the same sweep', polt.get('V_at_2Acm2'),
      expected_v_at(2.0, NCYC - 2, 'a1'), 0.002)
close('ΔV from the first sweep to that one', polt.get('dV_at_2Acm2_mV'),
      (expected_v_at(2.0, NCYC - 2, 'a1') - expected_v_at(2.0, 0, 'a1')) * 1000, 2.0)

# ════════════════════════════════════════════════════════════════════
print('main plots outlast per-spectrum fits under the size limit')
main = {n for n, s in sidecars1.items() if registry.is_main_plot(s.get('plot_type'))}
true('the run has main and other plots', main and len(main) < len(sidecars1))
sizes = sidecar_sizes(sidecars1)
largest = next(iter(sizes))
true('a main plot is the largest (the model fit)', largest in main,
     f'(largest {largest})')
kept, dropped = select_sidecars(sidecars1, sum(sizes[n] for n in main) + 1,
                                exclude_buckets=set())
check('a budget that fits only the main plots keeps all of them',
      main <= set(kept), True)
true('and drops other plots instead', len(dropped) > 0
     and not any(registry.is_main_plot(sidecars1[n].get('plot_type')) for n in dropped))

base = copy.deepcopy(rec1)
saved = jsonbin.MAX_BODY_BYTES
try:
    jsonbin.attach_sidecars(base, {n: sidecars1[n] for n in main})
    jsonbin.MAX_BODY_BYTES = len(json.dumps(base, ensure_ascii=False).encode()) + 500
    _, kept, dropped, body = jsonbin.fit_record_to_budget(
        copy.deepcopy(rec1), dict(sidecars1), protected=set(sidecars1))
finally:
    jsonbin.MAX_BODY_BYTES = saved
true('the transport loop sheds non-main plots first',
     main <= set(kept) and dropped, f'(kept {len(kept)}, dropped {len(dropped)})')

# ════════════════════════════════════════════════════════════════════
print('durability declares its hold current and guesses no stand')


def dur_folder(path, t0_hr, hours, rng):
    cols = ['Elapsed Time (s)', 'Step Name', 'Step Number', 'Repeats',
            'Working Electrode (V)', 'Current (A)', 'Frequency (Hz)',
            "Z' (ohms)", '-Z" (ohms)', 'DC Working Electrode (V)']
    path.mkdir(parents=True, exist_ok=True)
    lines = [','.join(cols)]
    for k in range(hours):
        v = 1.72 + 40e-6 * (t0_hr + k) + rng.normal(0, 1e-3)
        lines.append(f'{k * 3600.0:.2f},Constant Current,1,1,{v:.6f},{2.0 * 25:.6f},,,,')
    (path / 'data.csv').write_text('\n'.join(lines) + '\n')


rng = np.random.default_rng(5)
dur_in, dur_out = tmp / 'dur' / 'input', tmp / 'dur' / 'output'
dur_out.mkdir(parents=True)
dur_folder(dur_in / '01_first', 0.0, 60, rng)
dur_folder(dur_in / '02_second', 60.0, 60, rng)
with contextlib.redirect_stdout(io.StringIO()):
    dres = dur_run(str(dur_in), str(dur_out),
                   {'geo_area': '25', 'eis_ref_voltage': '1.25',
                    'image_format': 'png', 'folder_order': ['01_first', '02_second']})
dsc = load_sidecars(dur_out)
true('durability plots written', len(dsc) >= 2, f'(got {len(dsc)})')
check('every durability sidecar declares the hold current',
      {json.dumps(s.get('metadata', {}).get('conditions'), sort_keys=True)
       for s in dsc.values()}, {json.dumps({'j_hold_Acm2': 2.0})})
drec = build_detail_record(job_id='dur', sample_name='260905_ELXD',
                           script='Electrolyzer Durability',
                           timestamp='2026-10-06T00:00:00Z',
                           input_files=['01_first/data.csv', '02_second/data.csv'],
                           output_dir=dur_out, summary=dres.get('summary'))
dentry = build_index_entry(drec, 'BIND')
check('one step-less unit', [(u['Analysis'], u['step']) for u in dentry['Data']],
      [('durability', '')])
check('its condition is the hold current',
      dentry['Data'][0].get('Conditions'), {'j_hold_Acm2': 2.0})
check('no stand guessed for a durability run', dentry.get('stand'), None)

# A fuel-cell run with the same .csv files still gets its derived stand.
fc_rec = {'metrics': {'polcurve': {}}, 'input_files': ['a.csv'], 'sample_name': 'S'}
check('fuel-cell stand derivation unchanged',
      build_index_entry(fc_rec, 'B').get('stand'), 'FCTS')

# ════════════════════════════════════════════════════════════════════
print('a run that names no cell says so in its rows')
# Blank rather than absent: a row with no step at all is one written under
# the old rules, and in a record whose old runs were on one cell it is given
# that cell. Here an old run on a1 is followed by a current run naming none.
outn, resn, filesn = run_cell(RERUN, '', 'job_nocell')
check('it runs', resn.get('status'), 'success')
check('each row names no cell, explicitly',
      sorted((x.get('Analysis'), x.get('step', 'absent')) for x in resn['summary']),
      [('elx_eis', ''), ('elx_polcurve', '')])
S9 = '260830_ELXN'
legacy_sample(S9, 'LEG9')
r9 = jsonbin.push_job_metrics(job_id='job_nocell', sample_name=S9, script=SCRIPT,
                              output_dir=outn, input_files=filesn,
                              summary=resn.get('summary'), script_short=SHORT)
true('pushed beside the old cell', r9['pushed'], f"({r9['reason']})")
_, u9 = units_of(S9)
eisn = next(x for x in resn['summary'] if x['Analysis'] == 'elx_eis')
close('the unnamed run has its own HFR', u9.get(('elx_eis', ''), {}).get('HFR'),
      eisn['HFR'], 1e-6)
close('and the old cell keeps its own', u9.get(('elx_eis', 'a1'), {}).get('HFR'),
      0.041, 1e-9)
again = normalise_electrolyzer(store.bins['LEG9'])
check('read again, its rows are still its own, not the old cell\'s',
      sorted((x['Analysis'], x.get('step')) for x in again.get('summary') or []),
      [('elx_eis', ''), ('elx_eis', 'a1'), ('elx_polcurve', ''), ('elx_polcurve', 'a1')])

shutil.rmtree(tmp, ignore_errors=True)
print(f'\n{passed} passed, {failed} failed')
sys.exit(1 if failed else 0)
