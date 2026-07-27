"""Electrolyzer durability — tier 1 summary through to the index entry.

Synthesizes a two-folder durability test with a known degradation slope, runs
the real run(), then drives the output through the real record builder. The
point is to check the numbers survive every hop: results dict -> summary ->
detail bin -> index key_values, with the readout boxes agreeing.
"""
import sys, os, json, shutil, tempfile
from pathlib import Path

sys.path.insert(0, '/home/claude/target')
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np

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
    if got is not None and abs(got - want) <= tol:
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


# ── synthetic test ──────────────────────────────────────────────────
GEO = 25.0
RATE_UV_HR = 40.0          # target degradation, microvolts per hour
V_START = 1.7200
J_HOLD = 1.0               # A/cm2
ASR_BY_FOLDER = [140.0, 156.0]   # mOhm.cm2 at the EIS sweeps

COLS = ['Elapsed Time (s)', 'Step Name', 'Step Number', 'Repeats',
        'Working Electrode (V)', 'Current (A)',
        'Frequency (Hz)', "Z' (ohms)", '-Z" (ohms)',
        'DC Working Electrode (V)']


def row(t, step_name, step_no, rep, v, i, f='', zr='', zi='', dcv=''):
    return [f'{t:.2f}', step_name, str(step_no), str(rep),
            f'{v:.6f}' if v != '' else '', f'{i:.6f}' if i != '' else '',
            f, zr, zi, dcv]


def write_folder(path, t0_hr, dur_hr, asr_mohm, rng):
    """One folder: a current hold, one EIS sweep, one polcurve."""
    path.mkdir(parents=True, exist_ok=True)
    rows = []
    t = 0.0

    # Current hold, hourly samples. Voltage rises at RATE_UV_HR from V_START
    # measured on absolute test time, so the two folders form one line.
    n = int(dur_hr)
    for k in range(n):
        abs_hr = t0_hr + k
        v = V_START + RATE_UV_HR * 1e-6 * abs_hr + rng.normal(0, 1e-3)
        rows.append(row(k * 3600.0, 'Constant Current', 1, 1, v, J_HOLD * GEO))
    t = n * 3600.0

    # EIS sweep at 1.25 V DC. extract_hfr looks for the frequency where the
    # stored -Z" column crosses from inductive (negative) to capacitive
    # (positive); that crossing is the high-frequency real-axis intercept.
    # A sweep without an inductive branch has no such crossing and falls back
    # to argmin(|Z"|), which lands on the *low* frequency intercept instead —
    # so the series inductance here is what makes the fixture measure the
    # quantity it claims to.  Z = R_hfr + R_ct/(1 + jwR_ct*C) + jwL
    r_hfr = asr_mohm / (GEO * 1000.0)
    R_ct, C_dl, L_s = 0.004, 0.05, 1e-9
    tau = R_ct * C_dl
    freqs = np.logspace(5, -1, 60)
    for f in freqs:
        w = 2 * np.pi * f
        x = w * tau
        zre = r_hfr + R_ct / (1 + x ** 2)
        minus_zpp = R_ct * x / (1 + x ** 2) - w * L_s
        rows.append(row(t, 'EIS Potentiostatic', 2, 1, '', '',
                        f=f'{f:.4f}', zr=f'{zre:.8f}', zi=f'{minus_zpp:.8f}',
                        dcv='1.2500'))
        t += 1.0

    # Polcurve: six potential setpoints, four samples each so the median
    # window in load_and_split_file has something to bite on.
    for si, vset in enumerate([1.45, 1.55, 1.65, 1.75, 1.85, 1.95]):
        # Rough Tafel-ish current so fit_polcurve has a curve to fit.
        j = max(0.01, (vset - 1.23 - 0.05) / (asr_mohm / 1000.0) * 0.35)
        for _ in range(4):
            rows.append(row(t, 'Constant Potential', 10 + si, 1,
                            vset + rng.normal(0, 1e-4), j * GEO))
            t += 1.0

    with open(path / 'data.csv', 'w', encoding='utf-8') as fh:
        fh.write(','.join(COLS) + '\n')
        for r in rows:
            fh.write(','.join(r) + '\n')


tmp = Path(tempfile.mkdtemp())
inp, out = tmp / 'input', tmp / 'output'
out.mkdir(parents=True)
rng = np.random.default_rng(7)
write_folder(inp / '01_first', 0.0, 500, ASR_BY_FOLDER[0], rng)
write_folder(inp / '02_second', 500.0, 500, ASR_BY_FOLDER[1], rng)

# ── run it ──────────────────────────────────────────────────────────
from scripts.electrolyzer_durability import run as dur_run

print('run()')
res = dur_run(str(inp), str(out),
              {'geo_area': str(GEO), 'eis_ref_voltage': '1.25',
               'sample_name': '260714_ELX1', 'image_format': 'png',
               'folder_order': ['01_first', '02_second']})
check('status', res.get('status'), 'success')
check('folders processed', res.get('folders_processed'), 2)
true('produced output files', len(res.get('files_produced') or []) > 0)

summary = res.get('summary')
true('summary is a list', isinstance(summary, list), f'(got {type(summary).__name__})')
check('one summary row', len(summary or []), 1)
s = (summary or [{}])[0]

print('tier 1 scalars')
check('Analysis tag', s.get('Analysis'), 'durability')
close('Degradation rate', s.get('Degradation rate'), RATE_UV_HR, 3.0)
close('Duration', s.get('Duration'), 999.0, 2.0)
close('V_initial', s.get('V_initial'), V_START, 5e-4)
close('V_final', s.get('V_final'),
      V_START + RATE_UV_HR * 1e-6 * 999.0, 5e-4)
true('V_final exceeds V_initial', s.get('V_final', 0) > s.get('V_initial', 1))
close("HFR_initial", s.get("HFR_initial"), ASR_BY_FOLDER[0], 0.5)
close("HFR_final", s.get("HFR_final"), ASR_BY_FOLDER[1], 0.5)
close('j_hold', s.get('j_hold'), J_HOLD, 0.02)
close('geo_area', s.get('geo_area'), GEO, 1e-9)

print('internal consistency of the three voltage metrics')
implied = ((s.get('V_final', 0) - s.get('V_initial', 0))
           / s.get('Duration', 1) * 1e6)
close('(V_final-V_initial)/Duration reproduces the rate exactly',
      implied, s.get('Degradation rate', 0), 1e-6)

print('readout boxes reach the sidecars')
from scripts.helpers.plot_compare import load_sidecar
pd_dir = out / '_plot_data'
true('sidecar directory written', pd_dir.is_dir())
sidecars = {p.stem: json.loads(p.read_text()) for p in pd_dir.glob('*.json')}
true('six sidecars', len(sidecars) >= 4, f'(got {len(sidecars)})')


def texts_of(name):
    sc = sidecars.get(name) or {}
    out_t = []
    for ax in (sc.get('data') or {}).get('axes', []):
        for t in ax.get('texts', []):
            out_t.append(t.get('text', ''))
        for ref in ax.get('axhlines', []) + ax.get('axvlines', []):
            if ref.get('label'):
                out_t.append(ref['label'])
    return '\n'.join(out_t)


from scripts.helpers.record import parse_metric_kv
vt = parse_metric_kv(texts_of('voltage_vs_time'))
true('voltage plot box carries the rate', 'Degradation rate' in vt, f'(keys {list(vt)})')
true('voltage plot box carries V_initial', 'V_initial' in vt)
true('voltage plot box carries Duration', 'Duration' in vt)
dr = parse_metric_kv(texts_of('degradation_rate'))
true('rolling plot refline uses the canonical key', 'Degradation rate' in dr,
     f'(keys {list(dr)})')
hf = parse_metric_kv(texts_of('hfr_vs_time'))
true('hfr plot box carries both endpoints',
     'HFR_initial' in hf and 'HFR_final' in hf, f'(keys {list(hf)})')

print('units survive parsing')
_v = vt.get('Degradation rate')
check('rate parsed with a unit', isinstance(_v, dict) and _v.get('unit'), '\u03bcV/hr')

print('record build -> index key_values')
from scripts.helpers.record import build_detail_record, build_index_entry
rec = build_detail_record(
    job_id='dur-1', sample_name='260714_ELX1',
    script='Electrolyzer Durability', timestamp='2026-07-20T00:00:00Z',
    input_files=['data.csv'], output_dir=out, summary=summary)
true('durability bucket present', 'durability' in (rec.get('metrics') or {}),
     f"(buckets {list(rec.get('metrics') or {})})")
check('summary stored on the detail record', len(rec.get('summary') or []), 1)

entry = build_index_entry(rec, 'BINDUR')
units = [(u['Analysis'], u['step']) for u in entry['Data']]
check('single merge unit', units, [('durability', '')])
kv = entry['Data'][0].get('key_values') or {}
for name in ('Degradation rate', 'V_initial', 'V_final',
             'HFR_initial', 'HFR_final', 'Duration'):
    true(f'index carries {name}', name in kv, f'(key_values {sorted(kv)})')
close('index rate matches summary', kv.get('Degradation rate'),
      s.get('Degradation rate'), 0.5)

print('non-vacuous: a run with too few points has no rate but still records')
row2 = __import__('scripts.electrolyzer_durability', fromlist=['x']).build_summary(
    np.array([0.0, 1.0]), np.array([1.7, 1.7]), None, [], [], 1.0, GEO)
true('no rate without a regression', 'Degradation rate' not in row2)
true('duration still present', 'Duration' in row2)
check('bucket tag still set', row2.get('Analysis'), 'durability')

shutil.rmtree(tmp, ignore_errors=True)
print(f'\n{passed} passed, {failed} failed')
sys.exit(1 if failed else 0)
