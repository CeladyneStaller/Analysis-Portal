"""H2 crossover: the crossover and, beside it, the short-corrected crossover
and R_short, wherever the crossover is reported.

The crossover is the mean of the averaged CV current over the window
(0.35-0.50 V), as FC Durability judges it. The line through the averaged
current over the same window, j_xover + V / R_short, gives the rest: its
value at 0 V is the crossover with the membrane short taken out, always
reported, and one over its slope is R_short, reported where the line rises.
A line that is flat or falls shows no short and has none.

Checked here, on synthetic H2/N2 CVs with known answers (the repository is
public; no lab data):
  1. analyze_crossover: the values, a current logged negative, no short, a
     falling line, the edge of a measurable slope; FC Durability's values of
     the same files.
  2. Reading a station CSV by its column names, and the files that are not one.
  3. The H2 Crossover script as the portal runs it: the summary row, the
     workbook, the plot and its readouts, the console, a station CSV beside a
     .fcd file, and only finite numbers in the summary.
  4. Into the store: the index's values and units, from the registry.
  5. Full Analysis: its summary rows and its crossover sheet.

Run on its own:  PYTHONPATH=. python scripts/helpers/test_h2_crossover.py
"""
import contextlib
import csv
import io
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
import warnings
import numpy as np
from openpyxl import load_workbook

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


with contextlib.redirect_stdout(io.StringIO()):
    from scripts import h2_crossover_analysis as H
    from scripts import fc_durability_analysis as F
    from scripts import fuelcell_analysis as FA
    from scripts.helpers import record as REC
    from scripts.helpers import registry as REG

# Two labels alike in their first 31 characters name the batch workbook's
# sheets alike; openpyxl numbers the second and says so.
warnings.filterwarnings('ignore', message='Title is more than 31 characters')
tmp = Path(tempfile.mkdtemp(prefix='h2x_'))
AREA = 5.0

# ── Synthetic CVs ────────────────────────────────────────────────────────────
# 2 mV/s over 0.12-0.80 V, three cycles, no vertex logged twice. The current
# density is the crossover plus a line in V (a short is a rising line), plus a
# capacitive current of opposite sign on the two sweeps, which the average of
# the sweeps cancels. Logged as total current (A) at 5 cm².
UP = np.round(0.12 + 0.002 * np.arange(341), 6)
TRI = np.concatenate([UP, UP[-2:0:-1]])
V_CV = np.concatenate([np.tile(TRI, 3), UP[:1]])
DIR = np.concatenate([np.tile(np.concatenate([np.ones(341), -np.ones(339)]), 3), [-1.0]])
T_CV = np.arange(V_CV.size) * 1.0
JX = 3.0e-3                      # the crossover, A/cm²
CAP = 2.0e-5                     # capacitive current density, A/cm²
# The window's grid points, as analyze_crossover lays them, and their mean V:
# the short adds slope × this to the window mean.
_g = np.linspace(0.12, 0.80, 500)
V_MID = float(_g[(_g >= 0.35) & (_g <= 0.50)].mean())


def cv_current(slope, sign=1.0, jx=JX):
    """Total current (A) of the CV: crossover `jx`, a line of `slope` A/cm²/V
    (1/R_short for a short), logged with oxidation positive (sign 1) or
    negative (-1)."""
    return sign * (jx + slope * V_CV + CAP * DIR) * AREA


def station_csv(path, I, extra=False, bom=False):
    """A station's own CSV: Time Stamp, Counter and Working Electrode Voltage,
    Current (A). `extra`: an Excel-edited one, with columns before and after."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    head = ['Time Stamp', 'Counter Electrode Voltage', 'Working Electrode Voltage', 'Current']
    rows = [[f'{t:.3f}', f'{1e-4:.6f}', f'{v:.6f}', f'{i:.12g}'] for t, v, i in zip(T_CV, V_CV, I)]
    if extra:
        head = ['Unnamed: 0'] + head + ['cd']
        rows = [[str(k)] + r + [f'{float(r[3]) * 200:.9g}'] for k, r in enumerate(rows)]
    with open(path, 'w', newline='', encoding='utf-8-sig' if bom else 'utf-8') as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)
    return path


def two_col_csv(path, I):
    """A plain export: voltage, then current (A), under one header row."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Voltage (V)', 'Current (A)'])
        w.writerows([[f'{v:.6f}', f'{i:.12g}'] for v, i in zip(V_CV, I)])
    return path


def fcd_file(path, I):
    """A FuelCell5 .fcd file: 'Key: value' lines, the column line, 'End
    Comments', then tab-separated rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ['Time (Sec)', 'I (A)', 'E_Stack (V)', 'Ctrl_Mode']
    head = ['FUELCELL DATA', 'Format: FuelCell5', 'SurfaceArea: 5', f'DataColumnCount: {len(cols)}',
            '\t'.join(cols), 'End Comments']
    with open(path, 'w', newline='') as f:
        f.write('\r\n'.join(head) + '\r\n')
        f.writelines(f'{t:.3f}\t{i:.12g}\t{v:.6f}\t0\r\n' for t, i, v in zip(T_CV, I, V_CV))
    return path


def num(x, nd=9):
    """A value to compare: rounded, or None for one that is not a number (the
    R² of a perfectly level line, in newer SciPy)."""
    return None if x is None or not math.isfinite(x) else round(float(x), nd)


def analyze(I):
    with contextlib.redirect_stdout(io.StringIO()):
        return H.analyze_crossover(V_CV, I / AREA)


# ── 1. analyze_crossover ────────────────────────────────────────────────────
R0 = 250.0
r = analyze(cv_current(1 / R0))
close('a short: the crossover is the window mean, the short in it',
      r['j_xover_mA_cm2'], (JX + V_MID / R0) * 1000, 1e-9)
close('... the short-corrected crossover, the window line at 0 V', r['j_xover_corrected_mA_cm2'],
      JX * 1000, 1e-9)
close('... R_short, one over its slope', r['R_short_ohm_cm2'], R0, 1e-6)
close('... the line fits', r['R_squared'], 1.0, 1e-9)
close('... the line as logged: its intercept', r['j_intercept_mA_cm2'], JX * 1000, 1e-9)
close('... and its slope (A/cm² per V)', r['slope'], 1 / R0, 1e-12)

rn = analyze(cv_current(1 / R0, sign=-1.0))
check('oxidation logged negative: the same crossover, corrected value and R_short, all positive',
      (round(rn['j_xover_mA_cm2'], 9), round(rn['j_xover_corrected_mA_cm2'], 9),
       round(rn['R_short_ohm_cm2'], 6)),
      (round(r['j_xover_mA_cm2'], 9), round(r['j_xover_corrected_mA_cm2'], 9),
       round(r['R_short_ohm_cm2'], 6)))
check('... the line itself as logged: intercept and slope negative',
      (rn['j_intercept_mA_cm2'] < 0, rn['slope'] < 0, rn['j_avg_mean_mA_cm2'] < 0), (True, True, True))

rf = analyze(cv_current(0.0))
check('no short (a flat line): no R_short', rf['R_short_ohm_cm2'], None)
close('... the corrected value is still reported, the crossover itself',
      rf['j_xover_corrected_mA_cm2'], JX * 1000, 1e-9)
close('... and equals the window mean', rf['j_xover_corrected_mA_cm2'], rf['j_xover_mA_cm2'], 1e-9)

rd = analyze(cv_current(-1 / 500.0))
check('a falling line: no R_short', rd['R_short_ohm_cm2'], None)
close('... the window mean below the crossover', rd['j_xover_mA_cm2'], (JX - V_MID / 500.0) * 1000, 1e-9)
close('... the corrected value above it, reported all the same', rd['j_xover_corrected_mA_cm2'],
      JX * 1000, 1e-9)
rdn = analyze(cv_current(-1 / 500.0, sign=-1.0))
check('... logged negative too: no R_short, the same corrected value',
      (rdn['R_short_ohm_cm2'], round(rdn['j_xover_corrected_mA_cm2'], 9)),
      (None, round(rd['j_xover_corrected_mA_cm2'], 9)))

rt = analyze(cv_current(2e-12))
close('a slope just measurable (2e-12 A/cm²/V): its R_short', rt['R_short_ohm_cm2'], 5e11, 5e11 * 1e-3)
check('... one under 1e-12: none', analyze(cv_current(5e-13))['R_short_ohm_cm2'], None)
close('... measurable logged negative too', analyze(cv_current(2e-12, sign=-1.0))['R_short_ohm_cm2'],
      5e11, 5e11 * 1e-3)

# FC Durability reads the same function on the same files: its values agree.
fdir = tmp / 'fcdur'
cases = {'short': cv_current(1 / R0), 'neg': cv_current(1 / R0, sign=-1.0),
         'flat': cv_current(0.0), 'fall': cv_current(-1 / 500.0)}
for name, I in cases.items():
    p = station_csv(fdir / f'b20_{name}_H2X_80Cell-100RH_0.4WH2-0.4WN2.csv', I)
    v, _, note = F.analyze_crossover_file(str(p), F.CharSettings())
    want = analyze(I)
    R = want['R_short_ohm_cm2']
    check(f'FC Durability, {name}: the crossover, the corrected value (its intercept), R_short, the fit',
          (round(v['j_xover'], 6), round(v['j_intercept'], 6),
           None if math.isnan(v['R_short']) else round(v['R_short'], 3), num(v['fit_R2'])),
          (round(want['j_xover_mA_cm2'], 6), round(want['j_xover_corrected_mA_cm2'], 6),
           None if R is None else round(R, 3), num(want['R_squared'])))

# ── 2. A station CSV's columns, by name ──────────────────────────────────────
sdir = tmp / 'cols'
check('a station CSV: Working Electrode Voltage and Current',
      H.station_csv_columns(str(station_csv(sdir / 'a.csv', cases['short']))), (2, 3))
check('... edited in Excel, with columns before and after, and a byte-order mark',
      H.station_csv_columns(str(station_csv(sdir / 'b.csv', cases['short'], extra=True, bom=True))), (3, 4))
check('a plain two-column export: not a station CSV',
      H.station_csv_columns(str(two_col_csv(sdir / 'c.csv', cases['short']))), None)
check('a .fcd file: not one (it has its own header)',
      H.station_csv_columns(str(fcd_file(sdir / 'd.fcd', cases['short']))), None)
(sdir / 'e.csv').write_text('Time Stamp,Working Electrode Voltage,Voltage\n0,0.1,0.2\n')
check('a header with the voltage but no current: not one', H.station_csv_columns(str(sdir / 'e.csv')), None)
check('a file that is not there: none', H.station_csv_columns(str(sdir / 'missing.csv')), None)

# ── 3. The H2 Crossover script as the portal runs it ─────────────────────────
NAME_S = 'b20_SYN_H2X_80Cell-100RH_0.4WH2-0.4WN2'
NAME_D = 'c14_SYN_H2X_80Cell-100RH_0.4WH2-0.4WN2'
up_dir, out_dir = tmp / 'up', tmp / 'out'
station_csv(up_dir / f'{NAME_S}.csv', cases['short'])
station_csv(up_dir / f'{NAME_D}.csv', cases['fall'])
with contextlib.redirect_stdout(io.StringIO()):
    res = H.run(str(up_dir), str(out_dir), {'image_format': 'png'})
rows = {row['Label']: row for row in res['summary']}
check('station CSVs are analysed: both, the crossover read off the right columns',
      (sorted(rows), round(rows[NAME_S]['j_xover_mA_cm2'], 6)),
      (sorted([NAME_S, NAME_D]), round((JX + V_MID / R0) * 1000, 6)))
check('the summary row: the crossover, then the corrected value, R_short and the fit, in that order',
      list(rows[NAME_S])[:5],
      ['Label', 'j_xover_mA_cm2', 'j_xover_corrected_mA_cm2', 'R_short_ohm_cm2', 'R_squared'])
check('... their values',
      (round(rows[NAME_S]['j_xover_corrected_mA_cm2'], 6), round(rows[NAME_S]['R_short_ohm_cm2'], 3),
       round(rows[NAME_S]['R_squared'], 9)),
      (round(JX * 1000, 6), round(R0, 3), 1.0))
check('a falling line: the corrected value, and R_short null in the row (measured as none, not absent)',
      (round(rows[NAME_D]['j_xover_corrected_mA_cm2'], 6), rows[NAME_D].get('R_short_ohm_cm2', 'absent')),
      (round(JX * 1000, 6), None))
check('... the same fields as a short\'s row, in the same order', list(rows[NAME_D]), list(rows[NAME_S]))

wb = load_workbook(out_dir / 'crossover_batch_data.xlsx')
ws = wb['Summary']
check('the workbook: the corrected value, R_short and the fit beside the crossover',
      [c.value for c in ws[1]],
      ['Label', 'j_xover (mA/cm2)', 'j_xover corrected (mA/cm2)', 'R_short (ohm·cm2)',
       'Window fit R²', 'H2 Flux (nmol/cm2/s)', 'H2 Flux (mL/min/cm2)'])
by_label = {ws.cell(row=k, column=1).value: [c.value for c in ws[k]] for k in (2, 3)}
check('... a short: its values', [round(x, 6) for x in by_label[NAME_S][1:5]],
      [round((JX + V_MID / R0) * 1000, 6), round(JX * 1000, 6), round(R0, 6), 1.0])
check('... a falling line: R_short says there is none, the corrected value stands',
      (by_label[NAME_D][3], round(by_label[NAME_D][2], 6)), (H.NO_SHORT_TEXT, round(JX * 1000, 6)))
check('... under the table, after a blank row, what the columns are',
      ([ws.cell(row=k, column=1).value for k in range(4, 8)]),
      [None] + list(H.XOVER_NOTES))
check('... the notes say which value is the crossover reported, and where none is given',
      ('the crossover reported' in H.XOVER_NOTES[0], '0 V' in H.XOVER_NOTES[1],
       'flat or falls' in H.XOVER_NOTES[2]), (True, True, True))

sidecars = REC.load_sidecars(out_dir)
vals_s = REC.extract_values(sidecars[f'crossover_{NAME_S}'])
vals_d = REC.extract_values(sidecars[f'crossover_{NAME_D}'])
check('the plot\'s readout: the crossover, the corrected value and R_short, as values',
      (vals_s['|j_xover|'], vals_s['j_xover corrected'], vals_s['R_short']),
      ({'value': round((JX + V_MID / R0) * 1000, 3), 'unit': 'mA/cm2'},
       {'value': 3.0, 'unit': 'mA/cm2'}, {'value': 250.0, 'unit': 'ohm·cm2'}))
check('... a falling line: the corrected value, and R_short as a word, which is no value',
      (vals_d['j_xover corrected'], vals_d.get('R_short'), REC._scalar(vals_d.get('R_short'))),
      ({'value': 3.0, 'unit': 'mA/cm2'}, H.NO_SHORT_TEXT, None))
texts_d = [t['text'] for ax in sidecars[f'crossover_{NAME_D}']['data']['axes'] for t in ax.get('texts', [])]
check('... and says so', any(f'R_short = {H.NO_SHORT_TEXT}' in t for t in texts_d), True)
lines = {ln['label']: ln for ln in sidecars[f'crossover_{NAME_S}']['data']['axes'][1]['lines']}
fit = lines.get('Window line, run back to 0 V', {})
check('the window line drawn from 0 V to the top of the window',
      (fit.get('x', [None])[0], round(fit.get('x', [0])[-1], 6)), (0.0, 0.5))
check('... through the averaged current there, and its value at 0 V marked',
      (round(fit.get('y', [0])[0], 6), round(fit.get('y', [0])[-1], 6),
       [round(y, 6) for y in lines.get('Short-corrected (line at 0 V)', {}).get('y', [])]),
      (round(JX * 1000, 6), round((JX + 0.5 / R0) * 1000, 6), [round(JX * 1000, 6)]))

# Logged negative: the line drawn as logged, the readout positive.
neg_dir, neg_out = tmp / 'up_neg', tmp / 'out_neg'
station_csv(neg_dir / f'{NAME_S}.csv', cases['neg'])
with contextlib.redirect_stdout(io.StringIO()):
    res_n = H.run(str(neg_dir), str(neg_out), {})
sc_n = REC.load_sidecars(neg_out)[f'crossover_{NAME_S}']
ln_n = {ln['label']: ln for ln in sc_n['data']['axes'][1]['lines']}
check('logged negative: the line drawn where the current is, the readout and row positive',
      (round(ln_n['Short-corrected (line at 0 V)']['y'][0], 6),
       REC.extract_values(sc_n)['j_xover corrected']['value'],
       round(res_n['summary'][0]['R_short_ohm_cm2'], 3)),
      (round(-JX * 1000, 6), 3.0, round(R0, 3)))

# The batch console: each CV's line and the closing table.
buf_b = io.StringIO()
with contextlib.redirect_stdout(buf_b):
    H.run(str(up_dir), str(tmp / 'out_console'), {})
cons = buf_b.getvalue()
check('the batch console: each CV\'s corrected value and R_short beside its crossover',
      (f'corrected=3.000 mA/cm2  R_short=250 ohm.cm2' in cons,
       f'corrected=3.000 mA/cm2  R_short={H.NO_SHORT_TEXT}' in cons),
      (True, True))
table = [l.split() for l in cons.splitlines() if l.strip().startswith(('b20_', 'c14_'))]
check('... and in the closing table', [t[1:4] for t in table if len(t) == 5],
      [[f'{(JX + V_MID / R0) * 1000:.3f}', '3.000', '250'],
       [f'{(JX - V_MID / 500.0) * 1000:.3f}', '3.000', 'none']])

# The command line's --file: a station CSV read by its names, the short reported.
cli_out = tmp / 'cli'
real_argv, real_show = sys.argv, H.plt.show
sys.argv = ['h2_crossover_analysis.py', '--file', str(up_dir / f'{NAME_S}.csv'), '--save-dir', str(cli_out)]
buf_c = io.StringIO()
try:
    with contextlib.redirect_stdout(buf_c):
        H.main()
finally:
    sys.argv = real_argv
cli_rows = {row[0].value: row[1].value for row in load_workbook(cli_out / 'crossover_data.xlsx')['Summary'].iter_rows()
            if row[0].value}
check('the command line\'s --file: a station CSV read by its names, the short reported',
      ('Short-corrected j:   3.000 mA/cm2' in buf_c.getvalue(),
       round(cli_rows['H2 Crossover j (mA/cm2)'], 6), round(cli_rows['R_short (ohm·cm2)'], 3),
       (cli_out / 'crossover.png').exists()),
      (True, round((JX + V_MID / R0) * 1000, 6), round(R0, 3), True))

for args_, src, want in (
        (['--area', '2.5'], up_dir / f'{NAME_S}.csv', (2 * (JX + V_MID / R0) * 1000, 2 * JX * 1000, R0 / 2)),
        ([], two_col_csv(tmp / 'cli_plain' / f'{NAME_S}.csv', cases['short']),
         (AREA * (JX + V_MID / R0) * 1000, AREA * JX * 1000, R0 / AREA))):
    sys.argv = ['h2_crossover_analysis.py', '--file', str(src), '--save-dir', str(tmp / 'cli2')] + args_
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            H.main()
    finally:
        sys.argv = real_argv
    got = {row[0].value: row[1].value
           for row in load_workbook(tmp / 'cli2' / 'crossover_data.xlsx')['Summary'].iter_rows() if row[0].value}
    check(f'--file {" ".join(args_) or "on a plain export"}: '
          + ('a station CSV\'s current divided by the area given' if args_
             else 'its current taken as per cm², as before (not divided)'),
          (round(got['H2 Crossover j (mA/cm2)'], 6), round(got['H2 Crossover j, short-corrected (mA/cm2)'], 6),
           round(got['R_short (ohm·cm2)'], 3)),
          tuple(round(x, 6 if k < 2 else 3) for k, x in enumerate(want)))

# A cell that is not a number is dropped as it is read: kept, it would make
# every value of the CV not a number.
nan_dir = tmp / 'up_nan'
p_nan = station_csv(nan_dir / f'{NAME_S}.csv', cases['short'])
lines = p_nan.read_text().splitlines()
# In the last cycle, which is the one analysed: a current at 0.42 V on its way
# down, a voltage at 0.46 V on its way up.
at = lambda v: [k for k, l in enumerate(lines[1:], 1) if abs(float(l.split(',')[2]) - v) < 1e-9]
k_i, k_v = at(0.42)[-1], at(0.46)[-2]
lines[k_i] = ','.join(lines[k_i].split(',')[:3] + ['nan'])
lines[k_v] = ','.join(lines[k_v].split(',')[:2] + ['inf'] + lines[k_v].split(',')[3:])
p_nan.write_text('\n'.join(lines) + '\n')
with contextlib.redirect_stdout(io.StringIO()):
    res_nan = H.run(str(nan_dir), str(tmp / 'out_nan'), {})
row_nan = res_nan['summary'][0]
check('a current or voltage that is not a number, in the window: that row dropped, the CV read',
      (round(row_nan['j_xover_mA_cm2'], 6), round(row_nan['j_xover_corrected_mA_cm2'], 6),
       round(row_nan['R_short_ohm_cm2'], 3)),
      (round((JX + V_MID / R0) * 1000, 6), round(JX * 1000, 6), round(R0, 3)))
check('R_short as shown: the value; none for a fitted line that is flat or falls; blank where no '
      'line was fitted',
      (H.r_short_shown({'R_short_ohm_cm2': 250.0, 'slope': 0.004}),
       H.r_short_shown({'R_short_ohm_cm2': None, 'slope': -0.002}),
       H.r_short_shown({'R_short_ohm_cm2': None, 'slope': float('nan')})),
      (250.0, H.NO_SHORT_TEXT, None))
nofit = dict(r, label='nofit', j_xover_mA_cm2=float('nan'), j_xover_corrected_mA_cm2=float('nan'),
             R_short_ohm_cm2=None, slope=float('nan'), R_squared=float('nan'),
             J_H2_nmol_cm2_s=float('nan'), J_H2_mL_min_cm2=float('inf'),
             membrane_thickness_um=25.0, K_H2_mol_cm_s_Pa=float('nan'))
with contextlib.redirect_stdout(io.StringIO()):
    H.save_batch_crossover_excel([nofit], str(tmp / 'nofit.xlsx'))
check('... a CV with no line fitted: its cells left blank, not called a flat line',
      [c.value for c in load_workbook(tmp / 'nofit.xlsx')['Summary'][2]], ['nofit'] + [None] * 6 + [25.0, None])
import zipfile
sheet_xml = zipfile.ZipFile(tmp / 'nofit.xlsx').read('xl/worksheets/sheet1.xml').decode()
with contextlib.redirect_stdout(io.StringIO()):
    H.save_crossover_excel(nofit, str(tmp / 'nofit_one.xlsx'), label='nofit')
    FA.save_consolidated_excel({'crossover': [nofit]}, str(tmp / 'nofit_fa.xlsx'))


def empty_numbers(path):
    """Worksheets of a workbook holding a numeric cell with no value."""
    z = zipfile.ZipFile(path)
    return [n for n in z.namelist() if n.startswith('xl/worksheets/')
            and ('<v />' in z.read(n).decode() or '<v/>' in z.read(n).decode())]


check('... blank cells, not numbers with no value: the batch workbook, the one-file one, Full Analysis\'s',
      ('<v />' in sheet_xml or '<v/>' in sheet_xml, empty_numbers(tmp / 'nofit_one.xlsx'),
       empty_numbers(tmp / 'nofit_fa.xlsx')), (False, [], []))
fa_nofit = load_workbook(tmp / 'nofit_fa.xlsx')['Crossover Summary']
check('... Full Analysis\'s row for it: blank, R_short not called none', [c.value for c in fa_nofit[2]][:7],
      ['nofit'] + [None] * 6)
real_analyze = H.analyze_crossover
H.analyze_crossover = lambda *a, **k: dict(nofit)
buf_n = io.StringIO()
try:
    with contextlib.redirect_stdout(buf_n):
        H.run_batch([str(up_dir / f'{NAME_S}.csv')], ['nofit'], geo_area=AREA, delimiter=',', skip=1,
                    v_col=0, j_col=1, current_is_total=True, avg_V_min=0.35, avg_V_max=0.50)
finally:
    H.analyze_crossover = real_analyze
check('... and the batch console says so the same way, its line and its table',
      ('R_short=-  ' in buf_n.getvalue(),
       [l.split()[3] for l in buf_n.getvalue().splitlines() if l.strip().startswith('nofit ')]),
      (True, ['-']))

# The console.
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    H.print_results(r)
    H.print_results(rd)
out_txt = buf.getvalue()
check('the console: the corrected value, R_short, the fit; none for a falling line',
      ('Short-corrected j:   3.000 mA/cm2' in out_txt, 'R_short:             250 ohm.cm2' in out_txt,
       'Window fit R2:       1.0000' in out_txt, f'R_short:             {H.NO_SHORT_TEXT}' in out_txt),
      (True, True, True, True))

# The one-file workbook (the command line's).
one = tmp / 'one.xlsx'
with contextlib.redirect_stdout(io.StringIO()):
    H.save_crossover_excel(r, str(one), label='one')
    H.save_crossover_excel(rd, str(tmp / 'one_d.xlsx'), label='one_d')
rows1 = {row[0].value: row[1].value for row in load_workbook(one)['Summary'].iter_rows() if row[0].value}
rows1d = {row[0].value: row[1].value for row in load_workbook(tmp / 'one_d.xlsx')['Summary'].iter_rows()
          if row[0].value}
check('the one-file workbook: the corrected value, R_short and the fit after the crossover',
      (round(rows1['H2 Crossover j, short-corrected (mA/cm2)'], 6), round(rows1['R_short (ohm·cm2)'], 3),
       round(rows1['Window fit R²'], 9), list(rows1).index('H2 Crossover j, short-corrected (mA/cm2)')
       - list(rows1).index('H2 Crossover j (mA/cm2)')),
      (round(JX * 1000, 6), round(R0, 3), 1.0, 1))
check('... a falling line: none', rows1d['R_short (ohm·cm2)'], H.NO_SHORT_TEXT)

# A station CSV and a .fcd file in one upload: each read its own way.
mix_dir, mix_out = tmp / 'up_mix', tmp / 'out_mix'
station_csv(mix_dir / f'{NAME_S}.csv', cases['short'])
fcd_file(mix_dir / 'c14_CV-2mVs_0o12to0o8V_80C-100RH_0o4H2-0o4N2_0kPa.fcd', cases['flat'])
with contextlib.redirect_stdout(io.StringIO()):
    res_m = H.run(str(mix_dir), str(mix_out), {})
mix = {row['Label']: row for row in res_m['summary']}
check('a station CSV beside a .fcd file: both read',
      sorted((k[:3], round(v['j_xover_mA_cm2'], 6), round(v['j_xover_corrected_mA_cm2'], 6),
              num(v.get('R_short_ohm_cm2', -1), 3)) for k, v in mix.items()),
      [('b20', round((JX + V_MID / R0) * 1000, 6), round(JX * 1000, 6), round(R0, 3)),
       ('c14', round(JX * 1000, 6), round(JX * 1000, 6), None)])

# A plain two-column export is still read by position.
two_dir, two_out = tmp / 'up_two', tmp / 'out_two'
two_col_csv(two_dir / f'{NAME_S}.csv', cases['short'])
with contextlib.redirect_stdout(io.StringIO()):
    res_t = H.run(str(two_dir), str(two_out), {})
check('a two-column export: read as before, by position',
      (round(res_t['summary'][0]['j_xover_mA_cm2'], 6), round(res_t['summary'][0]['R_short_ohm_cm2'], 3)),
      (round((JX + V_MID / R0) * 1000, 6), round(R0, 3)))

# Whatever a stand's preset says, a station CSV is read by its names: its
# header row, comma-separated, the current in A. One cycle, so a preset that
# skipped 51 rows would cut the first sweep and move the window's grid.
V1 = np.concatenate([UP, UP[-2::-1]])
D1 = np.concatenate([np.ones(341), -np.ones(340)])
I1 = (JX + V1 / R0 + CAP * D1) * AREA
one_csv = tmp / 'one_cycle' / f'{NAME_S}.csv'
one_csv.parent.mkdir(parents=True)
with open(one_csv, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['Time Stamp', 'Counter Electrode Voltage', 'Working Electrode Voltage', 'Current'])
    w.writerows([[f'{k:.1f}', '0.0001', f'{v:.6f}', f'{i:.12g}'] for k, (v, i) in enumerate(zip(V1, I1))])
with contextlib.redirect_stdout(io.StringIO()):
    want1 = H.analyze_crossover(V1, I1 / AREA)
    got1 = H.run_batch([str(one_csv)], ['one'], geo_area=AREA, delimiter='\t', skip=51, v_col=5, j_col=1,
                       current_is_total=False, avg_V_min=0.35, avg_V_max=0.50, mode_col=28,
                       mode_exclude={5})
check('a station CSV under a Scribner-like preset (tab, 51 rows, columns 5 and 1, A/cm², mode 5 out): '
      'read by its names, every row, the current per cm²',
      [(round(x['j_xover_mA_cm2'], 9), round(x['j_xover_corrected_mA_cm2'], 9),
        round(x['R_short_ohm_cm2'], 6), x['V_anodic'].size) for x in got1],
      [(round(want1['j_xover_mA_cm2'], 9), round(want1['j_xover_corrected_mA_cm2'], 9),
        round(want1['R_short_ohm_cm2'], 6), want1['V_anodic'].size)])

# The command line's one-file analysis reads it so too, under its FCTS preset
# (which takes the current as A/cm² and the first column as the voltage).
answers = iter(['1', '1', '5', '0.35', '0.50', '', '101.325', str(one_csv), ''])
import builtins
real_input, real_show = builtins.input, H.plt.show
builtins.input, H.plt.show = (lambda *a: next(answers)), (lambda *a, **k: None)
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        H.run_interactive()
finally:
    builtins.input, H.plt.show = real_input, real_show
check('the command line\'s one file: a station CSV read by its names, the short reported',
      (f'H2 crossover j:     {want1["j_xover_mA_cm2"]:.3f} mA/cm2' in buf.getvalue(),
       f'Short-corrected j:   {want1["j_xover_corrected_mA_cm2"]:.3f} mA/cm2' in buf.getvalue(),
       f'R_short:             {want1["R_short_ohm_cm2"]:.0f} ohm.cm2' in buf.getvalue()),
      (True, True, True))

# Only finite numbers reach the summary row: one NaN or infinity makes the job
# record unreadable.
real_batch = H.run_batch


def fake_batch(*a, save_dir=None, **k):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    (Path(save_dir) / 'x.txt').write_text('x')
    return [{'label': 'x', 'j_xover_mA_cm2': float('nan'), 'j_xover_corrected_mA_cm2': 1.5,
             'R_short_ohm_cm2': float('inf'), 'R_squared': np.float64(0.5),
             'J_H2_nmol_cm2_s': None, 'geo_area': True}]


H.run_batch = fake_batch
try:
    res_f = H.run(str(up_dir), str(tmp / 'out_fake'), {})
finally:
    H.run_batch = real_batch
check('the summary row keeps finite numbers only (no NaN, infinity, None or flag)',
      res_f['summary'], [{'Label': 'x', 'j_xover_corrected_mA_cm2': 1.5, 'R_squared': 0.5}])

# ── 4. Into the store ────────────────────────────────────────────────────────
rec = REC.build_detail_record(job_id='job1', sample_name='261110_SYN-X', script='H2 Crossover',
                              timestamp='2026-11-20T10:00:00Z',
                              input_files=[f'{NAME_S}.csv', f'{NAME_D}.csv'],
                              output_dir=out_dir, summary=res['summary'])
idx = REC.build_index_entry(rec, 'BIN1')
kv = {d['step']: d['key_values'] for d in idx['Data'] if d['Analysis'] == 'crossover'}
check('the index: each CV\'s crossover, corrected value and R_short',
      kv.get('b20'), {'|j_xover|': REC._round_sig((JX + V_MID / R0) * 1000),
                      'j_xover corrected': REC._round_sig(JX * 1000), 'R_short': 250.0})
check('... a falling line: no R_short', kv.get('c14'),
      {'|j_xover|': REC._round_sig((JX - V_MID / 500.0) * 1000),
       'j_xover corrected': REC._round_sig(JX * 1000)})
check('... from the summary, not the rounded readout',
      REC.build_key_values('crossover', {'j_xover corrected': {'value': 3.0, 'unit': 'mA/cm2'},
                                         'R_short': {'value': 250.0, 'unit': 'ohm·cm2'}},
                           {'j_xover_corrected_mA_cm2': 2.99987, 'R_short_ohm_cm2': 250.123}),
      {'j_xover corrected': 2.99987, 'R_short': 250.123})
check('... and never from a readout, which a unit merges across its CVs (a null R_short stays none)',
      (REC.build_key_values('crossover', {'j_xover corrected': {'value': 3.0, 'unit': 'mA/cm2'},
                                          'R_short': {'value': 250.0, 'unit': 'ohm·cm2'}}, {}),
       REC.build_key_values('crossover', {'R_short': {'value': 250.0, 'unit': 'ohm·cm2'}},
                            {'j_xover_corrected_mA_cm2': 3.0, 'R_short_ohm_cm2': None})),
      ({}, {'j_xover corrected': 3.0}))
def index_of(names_cases, tag):
    """Each crossover unit's index values for a run of the files given."""
    d, o = tmp / f'up_{tag}', tmp / f'out_{tag}'
    for name, case in names_cases:
        station_csv(d / f'{name}.csv', cases[case])
    with contextlib.redirect_stdout(io.StringIO()):
        res_x = H.run(str(d), str(o), {})
    rec_x = REC.build_detail_record(job_id='j', sample_name='261110_SYN-X', script='H2 Crossover',
                                    timestamp='2026-11-20T10:00:00Z', input_files=[], output_dir=o,
                                    summary=res_x['summary'])
    return [(u['step'], u.get('key_values')) for u in REC.build_index_entry(rec_x, 'B')['Data']]


SP_S, SP_D = NAME_S.replace('_', ' ', 3), NAME_D.replace('_', ' ', 3)
X_S, X_D = REC._round_sig((JX + V_MID / R0) * 1000), REC._round_sig((JX - V_MID / 500.0) * 1000)
check('file names with spaces: each summary row labelled as its plot is named, so each unit\'s three '
      'come from its own row',
      index_of([(SP_S, 'short'), (SP_D, 'fall')], 'spaces'),
      [('b20', {'|j_xover|': X_S, 'j_xover corrected': 3.0, 'R_short': 250.0}),
       ('c14', {'|j_xover|': X_D, 'j_xover corrected': 3.0})])
# (A space sorts before a dot, so '… repeat.csv' is read, and its row
# written, before '….csv'.)
check('... two CVs in one unit: all three from the first row, whichever it is',
      (index_of([(SP_S, 'short'), (SP_S + ' repeat', 'fall')], 'sp_sf'),
       index_of([(SP_S, 'fall'), (SP_S + ' repeat', 'short')], 'sp_fs')),
      ([('b20', {'|j_xover|': X_D, 'j_xover corrected': 3.0})],
       [('b20', {'|j_xover|': X_S, 'j_xover corrected': 3.0, 'R_short': 250.0})]))
# Two files whose names differ only by a space and an underscore: a plot each,
# the second's name made unique, and each row labelled as its own plot.
tw_d, tw_o = tmp / 'up_twin', tmp / 'out_twin'
station_csv(tw_d / f'{SP_S}.csv', cases['short'])
station_csv(tw_d / f'{NAME_S}.csv', cases['fall'])
with contextlib.redirect_stdout(io.StringIO()):
    res_tw = H.run(str(tw_d), str(tw_o), {})
rec_tw = REC.build_detail_record(job_id='j', sample_name='261110_SYN-X', script='H2 Crossover',
                                 timestamp='2026-11-20T10:00:00Z', input_files=[], output_dir=tw_o,
                                 summary=res_tw['summary'])
check('two files named alike but for a space and an underscore: a plot each, each row labelled by its own',
      ([row['Label'] for row in res_tw['summary']], sorted(REC.load_sidecars(tw_o))),
      ([NAME_S, NAME_S + '_2'], [f'crossover_{NAME_S}', f'crossover_{NAME_S}_2']))
check('... one unit, its three values from the first, whose plot is stored',
      [(u['step'], u.get('key_values')) for u in REC.build_index_entry(rec_tw, 'B')['Data']],
      [('b20', {'|j_xover|': X_S, 'j_xover corrected': 3.0, 'R_short': 250.0})])
check('... the second\'s values with its own plot',
      REC.extract_values(REC.load_sidecars(tw_o)[f'crossover_{NAME_S}_2'])['|j_xover|']['value'],
      round((JX - V_MID / 500.0) * 1000, 3))
fa_tw = FA._build_summary({'results': {'crossover': [dict(r, label='b20 SYN', plot_label='b20_SYN_2')]}})
check('... Full Analysis labels its rows by the same name', [row['Label'] for row in fa_tw], ['b20_SYN_2'])

check('the label as a plot\'s name writes it: spaces and slashes made underscores, in both summaries '
      '(a pol curve\'s left as it was)',
      (H.plot_label('b20 SYN/H2X\\a'),
       [row['Label'] for row in FA._build_summary({'results': {'crossover': [dict(r, label='b20 SYN/H2X')],
                                                               'polcurve': [{'label': 'b4 SYN PC',
                                                                             'OCV': 0.95}]}})]),
      ('b20_SYN_H2X_a', ['b4 SYN PC', 'b20_SYN_H2X']))

crossover_metrics = {m['name']: (m['sources'], m['unit'], m['better'], m['index'])
                     for m in REG.METRICS['crossover']}
check('the registry: the corrected value lower-better, R_short higher-better, both in the index, '
      'from the summary row',
      (crossover_metrics['j_xover corrected'], crossover_metrics['R_short']),
      ((['j_xover_corrected_mA_cm2'], 'mA/cm²', -1, True), (['R_short_ohm_cm2'], 'Ω·cm²', 1, True)))
check('... the fit in run detail only', crossover_metrics['Fit R²'], (['R_squared'], '', 0, False))
# Two CVs at one step and the same conditions are one unit, which takes each
# summary field from the first row that has it: its crossover, corrected value
# and R_short all come from that CV, whichever way round they are, never one
# CV's crossover beside another's R_short.
for first, second, want in (('fall', 'short', {'|j_xover|': REC._round_sig((JX - V_MID / 500.0) * 1000),
                                                'j_xover corrected': REC._round_sig(JX * 1000)}),
                            ('short', 'fall', {'|j_xover|': REC._round_sig((JX + V_MID / R0) * 1000),
                                               'j_xover corrected': REC._round_sig(JX * 1000),
                                               'R_short': 250.0})):
    rep_dir, rep_out = tmp / f'up_rep_{first}', tmp / f'out_rep_{first}'
    station_csv(rep_dir / f'{NAME_S}.csv', cases[first])
    station_csv(rep_dir / f'{NAME_S}_repeat.csv', cases[second])
    with contextlib.redirect_stdout(io.StringIO()):
        res_rep = H.run(str(rep_dir), str(rep_out), {})
    rec_rep = REC.build_detail_record(job_id='job2', sample_name='261110_SYN-X', script='H2 Crossover',
                                      timestamp='2026-11-20T10:00:00Z', input_files=[],
                                      output_dir=rep_out, summary=res_rep['summary'])
    check(f'two CVs at one step and the same conditions, the {first} first: one unit, all three from it',
          [d['key_values'] for d in REC.build_index_entry(rec_rep, 'BIN2')['Data']], [want])
check('... their units in the index', (REC.KEY_VALUE_UNITS['j_xover corrected'],
                                      REC.KEY_VALUE_UNITS['R_short']), ('mA/cm²', 'Ω·cm²'))

# ── 5. Full Analysis ─────────────────────────────────────────────────────────
fa_dir, fa_out = tmp / 'fa', tmp / 'fa_out'
station_csv(fa_dir / f'{NAME_S}.csv', cases['short'])
station_csv(fa_dir / f'{NAME_D}.csv', cases['fall'])
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    res_fa = FA.run(str(fa_dir), str(fa_out), {'stand': 'FCTS 1'})
fa_rows = {row['Label']: row for row in res_fa['summary'] if row.get('Analysis') == 'crossover'}
check('Full Analysis: its crossover rows carry the corrected value, R_short and the fit',
      (round(fa_rows[NAME_S]['j_xover_corrected_mA_cm2'], 6), round(fa_rows[NAME_S]['R_short_ohm_cm2'], 3),
       round(fa_rows[NAME_S]['R_squared'], 9)),
      (round(JX * 1000, 6), round(R0, 3), 1.0))
check('... a falling line: R_short null', fa_rows[NAME_D].get('R_short_ohm_cm2', 'absent'), None)
fws = load_workbook(fa_out / 'fuelcell_summary.xlsx')['Crossover Summary']
check('... its crossover sheet: the same columns beside the crossover',
      [c.value for c in fws[1]],
      ['Label', 'j_xover (mA/cm²)', 'j_xover corrected (mA/cm²)', 'R_short (Ω·cm²)', 'Window fit R²',
       'H₂ Flux (nmol/cm²/s)', 'H₂ Flux (mL/min/cm²)'])
fa_by = {fws.cell(row=k, column=1).value: [c.value for c in fws[k]] for k in (2, 3)}
check('... their values, and none for a falling line',
      ([round(x, 6) for x in fa_by[NAME_S][2:5]], fa_by[NAME_D][3]),
      ([round(JX * 1000, 6), round(R0, 6), 1.0], H.NO_SHORT_TEXT))
check('... what the columns are, under the table', [fws.cell(row=k, column=1).value for k in range(4, 8)],
      [None] + list(H.XOVER_NOTES))

# A station CSV beside a .fcd file under the Scribner stand: read by its names.
fm_dir, fm_out = tmp / 'fa_mix', tmp / 'fa_mix_out'
station_csv(fm_dir / f'{NAME_S}.csv', cases['short'])
fcd_file(fm_dir / 'c14_CV-2mVs_0o12to0o8V_80C-100RH_0o4H2-0o4N2_0kPa.fcd', cases['flat'])
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    res_fm = FA.run(str(fm_dir), str(fm_out), {'stand': 'Scribner 1'})
fm = sorted((row['Label'][:3], round(row['j_xover_mA_cm2'], 6)) for row in res_fm['summary']
            if row.get('Analysis') == 'crossover')
check('... a station CSV beside a .fcd file, on the Scribner stand: both read',
      fm, [('b20', round((JX + V_MID / R0) * 1000, 6)), ('c14', round(JX * 1000, 6))])

shutil.rmtree(tmp, ignore_errors=True)
print(f'\n{passed} passed, {failed} failed')
sys.exit(1 if failed else 0)
