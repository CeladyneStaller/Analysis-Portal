"""FC durability, milestones M1-M3: names, ordering, holds, clock, the OCV
analysis and death rule, characterizations, the crossover rule and the
verdict, outputs.

Builds synthetic runs in the station format -- one CSV per step, the real
column layouts and file-name grammar, Excel-edited extra columns, H2/N2 steps
and the 30 s every step drops from the end of its log -- in a temp dir, so
the harness needs no lab data. The repository is public; the real runs stay
off it. The OCV cases are the plan's verification cases 2 (death mid-hold 4),
4 (a transient at a hold's start) and 5 (a hold split across two files);
the characterization cases are 1 (crossover death), 3 (alive), 6 (BOL
failure), 7 (short-dominated) and 8 (missing crossover); each comes with
the guards its part of the rule exists for. Characterization files are
synthetic CVs, GEIS sweeps and stepped pol curves with known answers.

Run on its own:  PYTHONPATH=. python scripts/helpers/test_fc_durability.py
"""
import io
import json
import os
import shutil
import sys
import tempfile
import contextlib
from dataclasses import replace as dc_replace
from pathlib import Path

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


# Importing the package prints which registered scripts are absent; that is
# noise here.
with contextlib.redirect_stdout(io.StringIO()):
    from scripts.fc_durability_analysis import (
        parse_step_name, natural_key, build_timeline, order_folders, run,
        analyze_ocv, load_hold_trace, below_runs, rolling_decay, OcvSettings,
        PLOT_MAX_BINS, analyze_characterizations, decide, CharSettings, _pick_curves,
        analyze_crossover_file, _series_labels)
    from scripts.ecsa_analysis import compute_ecsa_hupd, generate_synthetic_cv
    from scripts.eis_analysis import extract_hfr, MODELS


# ── station-format writers ──────────────────────────────────────────

FOUR = 'Time Stamp,Counter Electrode Voltage,Working Electrode Voltage,Current'
POL = 's,mV,mA,mW'
GEIS = ('Time Stamp,DC Electrode Voltage,DC Current,Frequency,Abs Impedance,'
        'Phase Angle,Real Impedance,Imaginary Impedance,Total harmonic distortion')


def stamps(programmed_s, dt, t0=None):
    """Log stamps as the stations write them: from one interval in, ending
    30 s before the step does."""
    start = dt if t0 is None else t0
    return np.arange(start, programmed_s - 30 + 1e-9, dt)


def write_four(path, programmed_s, dt, v=0.9, i=0.0, extra=None, t0=None):
    t = stamps(programmed_s, dt, t0)
    vs = v(t) if callable(v) else np.full(len(t), v)
    head = FOUR + (',' + ','.join(extra) if extra else '')
    pad = ',' * len(extra) if extra else ''
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fh:
        fh.write(head + '\n')
        for tt, vv in zip(t, vs):
            fh.write(f'{tt:.6f},{1e-5:.6g},{vv:.6f},{i:.6g}{pad}\n')
    return len(t), float(t[0]), float(t[-1])


def write_pol(path, programmed_s, dt):
    t = stamps(programmed_s, dt)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fh:
        fh.write(POL + '\n')
        for k, tt in enumerate(t):
            mv = 900 - (k % 40) * 15
            ma = (k % 40) * 180.0
            fh.write(f'{tt:.3f},{mv},{ma:.1f},{mv * ma / 1000:.1f}\n')


def write_geis(path):
    f = np.logspace(5, -1, 61)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fh:
        fh.write(GEIS + '\n')
        for k, ff in enumerate(f):
            fh.write(f'{2.0 * (k + 1):.3f},0.10,-0.501,{ff:.4f},0.012,-10.0,0.010,'
                     f'{-0.003 * np.sin(k / 9):.6f},0.01\n')


def touch(path, data=b''):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)


def scribner_run(root):
    """One folder, the Gore-A shape: break-in and BOL, an 8 h H2/N2 step that
    the old 50-minute rule would have called a hold, six back-to-back 10 h
    holds (c5..c10, so string order would separate c10 from c9), EOL with
    repeated crossovers, and the clutter a real folder carries."""
    d = Path(root) / '231127_Synth-A'
    n = 'Synth-A'
    write_four(d / f'a0_{n}_OCV_80.3Cell-100RH_0.05WH2-0.05WN2.csv', 5400, 10, v=0.09)
    write_four(d / f'a4_{n}_H2X_80.3Cell-100RH_0.4WH2-0.4WN2.csv', 720, 0.5, v=0.4, i=0.018)
    write_pol(d / f'b1_{n}_Breakin_80.3Cell-100RH_0.2WH2-0.2WA.csv', 32400, 60)
    write_four(d / f'b2_{n}_OCV_80.3Cell-95RH_0.1WH2-0.1WA.csv', 3600, 10, v=0.88)
    write_pol(d / f'b3_{n}_PolarizationCurve_80.3Cell-95RH_0.2WH2-0.2WA.csv', 4900, 5)
    touch(d / f'b3_{n}_PolarizationCurve_80.3Cell-95RH_0.2WH2-0.2WA_FILTERDATA.csv',
          b'mV,mA/cm2,mW/cm2\n878,0.0,0.0\n')
    touch(d / f'b3_{n}_PolarizationCurve_80.3Cell-95RH_0.2WH2-0.2WA_IVCurve.tiff', b'II*\x00')
    write_four(d / f'b6_{n}_OCV_80.3Cell-100RH_0.05WN2-0.5DN2B3.csv', 600, 10, v=0.7)
    write_four(d / f'b14_{n}_OCV_90.3Cell-100RH_0.05WH2-0.05WN2.csv', 28800, 30, v=0.1)
    write_four(d / f'b20_{n}_H2X_80.3Cell-100RH_0.4WH2-0.4WN2.csv', 720, 0.5, v=0.4,
               i=0.0185, extra=['cd', '', ''])
    write_four(d / f'b24_{n}_ECSA_80.3Cell-100RH_0.05WH2-0.05WN2.csv', 140, 0.5, v=0.5, i=0.01)
    write_geis(d / f'b26_{n}_GEIS_80.3Cell-100RH_0.05WH2-0.05WN2.csv')
    write_four(d / f'c2_{n}_OCV_95Cell-30RH_0.05WH2-0.05WN2.csv', 7200, 10, v=0.09)
    write_four(d / f'c3_{n}_OCV_95Cell-30RH_0.05WH2-0.05WA.csv', 60, 1, v=0.87)
    write_four(d / f'c4_{n}_OCV_95Cell-30RH_0.1WH2-0.2WA.csv', 60, 1, v=0.92)
    for k in range(5, 11):
        write_four(d / f'c{k}_{n}_OCV_95Cell-30RH_0.1WH2-0.2WA.csv', 36000, 30,
                   v=lambda t, k=k: 0.92 - 0.01 * (k - 5) - 1e-7 * t)
    write_four(d / f'c20_{n}_OCV_80.3Cell-95RH_0.05WH2-0.05WA.csv', 3600, 10, v=0.79)
    for k in (32, 40, 41):
        write_four(d / f'c{k}_{n}_H2X_80.3Cell-100RH_0.4WH2-0.4WN2.csv', 720, 0.5, v=0.4, i=0.02)
    touch(d / 'Synth-A_OCV-Notes.txt', b'notes\n')
    touch(d / 'Synth-A_Full-OCV.xlsx', b'PK\x03\x04')
    touch(d / 'readme_export.csv', b'a,b\n1,2\n')
    return d


def fcts_folder(d, sample='01M12umSynth_2'):
    """One 40 h FCTS folder: two RH cycles' worth of the real protocol
    collapsed to one -- 10 h at 30% RH, then 10.425 h at 0% RH."""
    n = sample
    write_four(d / f'c1_{n}_OCV_90Cell-30RH_0.1WH2-0.1WN2.csv', 3600, 10, v=0.11)
    write_four(d / f'c2_{n}_OCV_90Cell-30RH_0.1WH2-0.1WA.csv', 60, 1, v=0.95)
    write_four(d / f'c3_{n}_OCV_90Cell-30RH_0.1WH2-0.1WA.csv', 36000, 30, v=0.92)
    write_four(d / f'c4_{n}_OCV_90Cell-100RH_0.1WH2-0.1WN2.csv', 3600, 10, v=0.12)
    write_four(d / f'c5_{n}_OCV_90Cell-0RH_0.1WH2-0.1DA.csv', 60, 1, v=0.92)
    write_four(d / f'c6_{n}_OCV_90Cell-0RH_0.1WH2-0.1DA.csv', 37530, 30, v=0.90)
    write_four(d / f'c9_{n}_OCV_90Cell-30RH_0.1WH2-0.1WN2.csv', 600, 1,
               v=lambda t: 0.84 - 0.41 * t / t[-1])
    write_four(d / f'e11_{n}_H2X_80.3Cell-100RH_0.4WH2-0.4WN2.csv', 720, 0.5, v=0.4, i=0.009)


def fcts_run(root):
    """A BOL folder named so it sorts last alphabetically, and three 40 h
    folders whose step letters restart -- 40H_10 must follow 40H_2."""
    base = Path(root) / 'FCTS_Synth'
    bol = base / 'Conditioning_BOL'
    n = '01M12umSynth_2'
    write_four(bol / f'a0_{n}_OCV_80.3Cell-100RH_0.05WH2-0.05WN2.csv', 3600, 10, v=0.09)
    write_pol(bol / f'b1_{n}_Breakin_80.3Cell-100RH_0.2WH2-0.2WA.csv', 7200, 60)
    write_four(bol / f'b20_{n}_H2X_80.3Cell-100RH_0.4WH2-0.4WN2.csv', 720, 0.5, v=0.4, i=0.0085)
    for name in ('40H_10', '40H_2', '40H_1'):
        fcts_folder(base / name)
    return base


tmp = Path(tempfile.mkdtemp(prefix='fcdur_m1_'))

# ── 1. file names ────────────────────────────────────────────────────
print('1. file names')
r = parse_step_name('c5_Synth-A_OCV_95Cell-30RH_0.1WH2-0.2WA.csv')
check('step', (r['step'], r['letter'], r['number']), ('c5', 'c', 5))
check('sample', r['sample'], 'Synth-A')
check('kind', (r['kind'], r['kind_token']), ('ocv', 'OCV'))
check('T and RH', (r['T_C'], r['RH_pct']), (95.0, 30.0))
check('anode', (r['anode_gas'], r['anode_slpm'], r['anode_humid']), ('H2', 0.1, 'wet'))
check('cathode', (r['cathode_gas'], r['cathode_slpm'], r['cathode_humid']), ('Air', 0.2, 'wet'))
check('no suffix', r['suffix'], '')

r = parse_step_name('e5_01M12umSynth_2_OCV_80.3Cell-100RH_0.05WN2-0.5DN2B3.csv')
check('sample with underscore', r['sample'], '01M12umSynth_2')
check('N2/N2 with B3 tail', (r['anode_gas'], r['cathode_gas'], r['cathode_slpm'],
                             r['cathode_humid']), ('N2', 'N2', 0.5, 'dry'))
check('dry air', parse_step_name('c6_S_OCV_90Cell-0RH_0.1WH2-0.1DA.csv')['cathode_humid'], 'dry')
check('EClean50x', parse_step_name('a2_S_EClean50x_80.3Cell-100RH_0.2WH2-0.2WN2.csv')['kind'], 'eclean')
check('GEIS', parse_step_name('b26_S_GEIS_80.3Cell-100RH_0.05WH2-0.05WN2.csv')['kind'], 'eis')
check('Breakin', parse_step_name('b1_S_Breakin_80.3Cell-100RH_0.2WH2-0.2WA.csv')['kind'], 'breakin')
r = parse_step_name('b5_S_PolarizationCurve_80.3Cell-95RH_0.75WH2-1WA.csv')
check('pol curve flows', (r['kind'], r['anode_slpm'], r['cathode_slpm']), ('polcurve', 0.75, 1.0))
check('FILTERDATA kept as suffix',
      parse_step_name('b3_S_PolarizationCurve_80.3Cell-95RH_0.2WH2-0.2WA_FILTERDATA.csv')['suffix'],
      'FILTERDATA')
check('O2 cathode, upper-case step', (lambda q: (q['step'], q['cathode_gas']))(
    parse_step_name('C12_S_OCV_80Cell-50RH_0.4WH2-0.4WO2.csv')), ('c12', 'O2'))
check('Air spelled out', parse_step_name('c3_S_OCV_80Cell-50RH_0.4WH2-0.4WAir.csv')['cathode_gas'], 'Air')
check('kind word inside the sample', (lambda q: (q['sample'], q['kind']))(
    parse_step_name('c3_OCV-batch_OCV_80Cell-50RH_0.4WH2-0.4WA.csv')), ('OCV-batch', 'ocv'))
check('unknown kind', parse_step_name('c3_S_Hold_80Cell-50RH_0.4WH2-0.4WA.csv')['kind'], 'other')
check('not a step file', parse_step_name('readme_export.csv'), None)
check('no conditions token', parse_step_name('c3_S_OCV_notes.csv'), None)
check('no sample', parse_step_name('c3_OCV_80Cell-50RH_0.4WH2-0.4WA.csv'), None)
check('natural sort', sorted(['40H_10', '40H_2', '40H_1'], key=natural_key),
      ['40H_1', '40H_2', '40H_10'])

# ── 2. single-folder run (Scribner shape) ────────────────────────────
print('2. single-folder run')
sc_dir = scribner_run(tmp / 'scribner')
tl = build_timeline(str(tmp / 'scribner'))
roles = {}
for f in tl.files:
    roles[f.role] = roles.get(f.role, 0) + 1
check('roles', roles, {'h2n2': 3, 'crossover': 5, 'breakin': 1, 'air-step': 4,
                       'polcurve': 1, 'n2n2': 1, 'ecsa': 1, 'eis': 1, 'hold': 6})
check('step files', len(tl.files), 23)
check('holds are c5-c10 in order', [f.step for f in tl.holds],
      ['c5', 'c6', 'c7', 'c8', 'c9', 'c10'])
check('one hold segment', len(tl.segments), 1)
close('hours tested = 6 x (35,970 + 30) s', tl.hours_tested, 60.0, 1e-9)
b14 = next(f for f in tl.files if f.step == 'b14')
check('8 h H2/N2 step is not a hold', (b14.role, b14.length_s), ('h2n2', 28800.0))
check('1 h air OCV is a step', next(f for f in tl.files if f.step == 'b2').role, 'air-step')
check('steps in order', [f.step for f in tl.files][:6], ['a0', 'a4', 'b1', 'b2', 'b3', 'b6'])
check('c10 follows c9', [f.step for f in tl.files].index('c10') -
      [f.step for f in tl.files].index('c9'), 1)
b20 = next(f for f in tl.files if f.step == 'b20')
check('BOL crossover stamped at 0 h', (b20.interval, b20.H_start_h), (0, 0.0))
check('Excel-edited file reads', b20.n_rows, 1380)
c41 = next(f for f in tl.files if f.step == 'c41')
check('EOL crossover stamped after the hold', (c41.interval, round(c41.H_start_h, 6)), (1, 60.0))
c7 = next(f for f in tl.files if f.step == 'c7')
check('hold file clock', (c7.segment, round(c7.H_start_h, 6), round(c7.H_end_h, 6), c7.clock_s),
      (1, 20.0, 30.0, 36000.0))
check('ignored files', tl.ignored, {'.tiff': 1, '.txt': 1, '.xlsx': 1})
check('skipped files', sorted(reason for _, reason in tl.skipped),
      ['name does not follow the step-file pattern', 'station pol-curve summary (FILTERDATA)'])
check('gap note', [n for n in tl.notes if 'no file for' in n][:1],
      ['231127_Synth-A: no file for a1, a2, a3'])
check('conditions text', tl.segments[0].conditions, '95 °C, 30% RH, H2/Air 0.1/0.2 slpm')

tl2 = build_timeline(str(tmp / 'scribner'), hold_min_h=0.5)
check('hold_min_h 0.5 makes the 1 h air steps holds', [f.step for f in tl2.holds],
      ['b2', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c20'])
check('... in three segments', len(tl2.segments), 3)
close('... adding 2 h', tl2.hours_tested, 62.0, 1e-9)
close('step_tail_s 0 counts logged time only',
      build_timeline(str(tmp / 'scribner'), step_tail_s=0).hours_tested, 6 * 35970 / 3600, 1e-9)

# ── 3. multi-folder run (FCTS shape) ─────────────────────────────────
print('3. multi-folder run')
fcts_run(tmp / 'fcts')
tl = build_timeline(str(tmp / 'fcts'))
check('folder order', tl.folders, ['FCTS_Synth/Conditioning_BOL', 'FCTS_Synth/40H_1',
                                   'FCTS_Synth/40H_2', 'FCTS_Synth/40H_10'])
check('holds', [(f.folder.split('/')[-1], f.step) for f in tl.holds],
      [('40H_1', 'c3'), ('40H_1', 'c6'), ('40H_2', 'c3'), ('40H_2', 'c6'),
       ('40H_10', 'c3'), ('40H_10', 'c6')])
check('six segments', len(tl.segments), 6)
close('hours tested = 3 x (10 + 10.425) h', tl.hours_tested, 61.275, 1e-9)
check('segment ends', [round(s.H_end_h, 6) for s in tl.segments],
      [10.0, 20.425, 30.425, 40.85, 50.85, 61.275])
e11 = [(f.folder.split('/')[-1], f.interval, round(f.H_start_h, 6))
       for f in tl.files if f.step == 'e11']
check('characterization stamped per folder', e11,
      [('40H_1', 2, 20.425), ('40H_2', 4, 40.85), ('40H_10', 6, 61.275)])
c9 = next(f for f in tl.files if f.step == 'c9')
check('purge falling through 0.7 V is a gas step', c9.role, 'h2n2')
check('missing c7, c8 noted per folder',
      sum(1 for n in tl.notes if n.endswith('no file for c7, c8')), 3)
check('dry hold conditions', tl.segments[1].conditions, '90 °C, 0% RH, H2/dry Air 0.1/0.1 slpm')

# Folder names and segment hours label the timeline from the figure, so the
# stored copy keeps only the readout: every text of an axes is stored as
# readout, and a re-render listed the folders and hours there.
from scripts.fc_durability_analysis import plot_timeline
_fig = plot_timeline(tl)
check('timeline: folders and hours labelled from the figure, the axes holding only the readout',
      (len(_fig.axes[0].texts), sorted(t.get_text().strip() for t in _fig.texts)),
      (1, sorted(['40H_1', '40H_10', '40H_2', 'Conditioning_BOL']
                 + [f'{s.H_end_h:.2f} h' for s in tl.segments])))
import matplotlib.pyplot as _plt
_plt.close(_fig)

# Explicit order: exact paths win, the rest follow canonically.
tl = build_timeline(str(tmp / 'fcts'),
                    folder_order=['FCTS_Synth/40H_10', 'FCTS_Synth/Conditioning_BOL'])
check('explicit folder order', tl.folders, ['FCTS_Synth/40H_10', 'FCTS_Synth/Conditioning_BOL',
                                            'FCTS_Synth/40H_1', 'FCTS_Synth/40H_2'])
true('note on folders left out', any('placed after it' in n for n in tl.notes), tl.notes)

# ── 4. ordering and segment edge cases ───────────────────────────────
print('4. edge cases')
check('exact folder matching, no substrings',
      order_folders({'Trap/A_hold1': 'c', 'Trap/hold1': 'c'}, ['Trap/hold1', 'Trap/A_hold1'])[0],
      ['Trap/hold1', 'Trap/A_hold1'])
order, notes = order_folders({'Trap/A_hold1': 'c', 'Trap/hold1': 'c'}, ['hold1'])
check('partial path matches nothing', order, ['Trap/A_hold1', 'Trap/hold1'])
true('... and says so', any('matches no folder exactly' in n for n in notes), notes)

cond = tmp / 'cond'
write_four(cond / 'c3_S_OCV_90Cell-30RH_0.1WH2-0.1WA.csv', 36000, 30)
write_four(cond / 'c4_S_OCV_90Cell-0RH_0.1WH2-0.1DA.csv', 36000, 30)
write_four(cond / 'c5_S_OCV_90Cell-0RH_0.1WH2-0.1DA.csv', 36000, 30)
tl = build_timeline(str(cond))
check('a change of conditions starts a new hold', [[f.step for f in s.files] for s in tl.segments],
      [['c3'], ['c4', 'c5']])

edge = tmp / 'edge'
write_four(edge / 'c3_S_OCV_90Cell-30RH_0.1WH2-0.1WA.csv', 7200, 30)
write_four(edge / 'c5_S_OCV_90Cell-30RH_0.1WH2-0.1WN2.csv', 600, 10)
write_four(edge / 'c6_S_OCV_90Cell-30RH_0.1WH2-0.1WA.csv', 7170, 30)
tl = build_timeline(str(edge))
check('a 2 h step logged every 30 s is a hold; 1.99 h is not',
      [(f.step, f.role) for f in tl.files if f.cathode_gas == 'Air'],
      [('c3', 'hold'), ('c6', 'air-step')])

split = tmp / 'split'
write_four(split / 'part_1' / 'c3_S_OCV_90Cell-30RH_0.1WH2-0.1WA.csv', 36000, 30)
write_four(split / 'part_2' / 'c3_S_OCV_90Cell-30RH_0.1WH2-0.1WA.csv', 36000, 30)
tl = build_timeline(str(split))
check('a hold never spans folders', [len(s.files) for s in tl.segments], [1, 1])
close('... clock still adds up', tl.hours_tested, 20.0, 1e-9)

late = tmp / 'late'
write_four(late / 'c3_S_OCV_90Cell-30RH_0.1WH2-0.1WA.csv', 41030, 30, t0=5000)
tl = build_timeline(str(late))
check('late-starting log counts its span', tl.holds[0].clock_s, 36030.0)
true('... with a note', 'log starts at 5000 s' in tl.holds[0].note, tl.holds[0].note)

empty = tmp / 'empty'
touch(empty / 'only.tiff', b'II*\x00')
tl = build_timeline(str(empty))
check('no step files', (len(tl.files), tl.hours_tested), (0, 0.0))

nohold = tmp / 'nohold'
write_four(nohold / 'b20_S_H2X_80.3Cell-100RH_0.4WH2-0.4WN2.csv', 720, 0.5)
tl = build_timeline(str(nohold))
true('no-hold note', any(n.startswith('no OCV holds found') for n in tl.notes), tl.notes)

# ── 5. run() end to end ──────────────────────────────────────────────
print('5. run()')
out = tmp / 'out'
with contextlib.redirect_stdout(io.StringIO()):
    res = run(str(tmp / 'scribner'), str(out), {'image_format': 'png', 'loading': '0.2'})
check('status', res['status'], 'success')
row = res['summary'][0]
check('summary identity', (row['Analysis'], row['Label']), ('fc_durability', 'fcdur'))
close('summary hours', row['Hours tested'], 60.0, 1e-9)
check('summary counts', (row['Hold files'], row['Hold segments'], row['Step files']), (6, 1, 23))
check('loading and area parameters', (row['loading'], row['geo_area']), (0.2, 5.0))
true('timeline png', (out / 'fcdur_timeline.png').exists())
sidecar_path = out / '_plot_data' / 'fcdur_timeline.json'
true('timeline sidecar', sidecar_path.exists())


def _no_nan(text):
    raise ValueError(f'non-finite value {text} in sidecar')


sidecar = json.loads(sidecar_path.read_text(), parse_constant=_no_nan)
check('sidecar plot type', sidecar['plot_type'], 'fcdur_timeline')
ax0 = sidecar['data']['axes'][0]
check('one square-wave line per lane', len(ax0['lines']), 9)
true('lanes kept as lines, not axhlines',
     all(len(ln['x']) >= 6 for ln in ax0['lines']) and not ax0['axhlines'])
from scripts.helpers.record import extract_values
vals = extract_values(sidecar)
close('readout reaches the record layer', (vals.get('Hours tested') or {}).get('value'), 60.0, 1e-9)

import openpyxl
wb = openpyxl.load_workbook(out / 'fc_durability.xlsx', read_only=True)
check('workbook sheets', wb.sheetnames, ['Summary', 'Cycles', 'Holds', 'OCV by condition',
                                         'OCV events', 'Decay rate', 'OCV trace', 'Hold files',
                                         'Characterizations', 'Manifest', 'Notes'])
check('manifest rows: 23 step files + 2 skipped', wb['Manifest'].max_row - 1, 25)
check('holds sheet rows', wb['Holds'].max_row - 1, 1)
summary_items = {r[0]: r[1] for r in wb['Summary'].iter_rows(min_row=2, values_only=True)}
check('nothing left out at upload: said so, and no summary field',
      (summary_items.get('Folders left out at upload'), 'Folders left out' in row), ('none', False))
wb.close()

# Folders skipped in the upload page's folder list never arrive; the page
# names them, and every output says the analysis did not see them.
out_left = tmp / 'out_left'
with contextlib.redirect_stdout(io.StringIO()):
    res_left = run(str(tmp / 'scribner'), str(out_left),
                   {'image_format': 'none', 'skipped_folders': ['Run/40H_3/', '  ', 'Run/extra']})
wb = openpyxl.load_workbook(out_left / 'fc_durability.xlsx', read_only=True)
summary_items = {r[0]: r[1] for r in wb['Summary'].iter_rows(min_row=2, values_only=True)}
notes_left = [r[0] for r in wb['Notes'].iter_rows(min_row=2, values_only=True)]
wb.close()
check('skipped folders: named in the workbook summary, blanks dropped',
      summary_items.get('Folders left out at upload'), 'Run/40H_3, Run/extra')
true('... noted as not in the analysis',
     any('left out at upload' in str(n) and 'Run/40H_3, Run/extra' in str(n) for n in notes_left),
     notes_left)
check('... and in the stored summary row', res_left['summary'][0].get('Folders left out'),
      'Run/40H_3, Run/extra')
with contextlib.redirect_stdout(io.StringIO()):
    res_bad = run(str(tmp / 'scribner'), str(tmp / 'out_left_bad'),
                  {'image_format': 'none', 'skipped_folders': 'Run/40H_3'})
check('skipped_folders that is not a list is ignored',
      'Folders left out' in res_bad['summary'][0], False)
# Names as the page sends them: a control character cannot go into a workbook
# cell, a non-string is not a name, and a folder that was read is not left out.
read_folder = build_timeline(str(tmp / 'scribner')).folders[0]
try:
    with contextlib.redirect_stdout(io.StringIO()):
        res_odd = run(str(tmp / 'scribner'), str(tmp / 'out_left_odd'),
                      {'image_format': 'none',
                       'skipped_folders': ['Run/40H\x01_9', None, 7, read_folder + '/']})
    odd = res_odd['summary'][0].get('Folders left out')
except Exception as e:                               # noqa: BLE001
    odd = f'run failed: {type(e).__name__}: {e}'
check('left-out names cleaned: control characters dropped, non-names and read folders ignored',
      odd, 'Run/40H_9')

out2 = tmp / 'out_noimg'
with contextlib.redirect_stdout(io.StringIO()):
    run(str(tmp / 'scribner'), str(out2), {'image_format': 'none'})
check('no images: workbook only', sorted(p.name for p in out2.rglob('*') if p.is_file()),
      ['fc_durability.xlsx'])

try:
    with contextlib.redirect_stdout(io.StringIO()):
        run(str(empty), str(tmp / 'out_empty'), {})
    true('empty upload raises', False)
except RuntimeError:
    true('empty upload raises', True)

# ── 6. OCV analysis ──────────────────────────────────────────────────
print('6. OCV analysis')

HOLD = '{}_S_OCV_95Cell-30RH_0.1WH2-0.2WA.csv'
DRY = '{}_S_OCV_95Cell-0RH_0.1WH2-0.2DA.csv'
rng = np.random.default_rng(7)


def write_ocv(path, programmed_s, dt, H0_s, v, i=None, drop=None, header=FOUR, extra=None,
              no_current=False, dup=False):
    """An OCV step whose voltage, and optionally current, are functions of
    hold-clock seconds, for a file that sits at H0_s on the clock. `drop`
    removes samples (a gap in the log) where it is true; a NaN current is
    written as a blank cell; `no_current` leaves the Current column out and
    `dup` writes every row twice. Returns the clock times and voltages as
    written, to work out expected values independently."""
    t = stamps(programmed_s, dt)
    H = H0_s + t
    keep = ~drop(H) if drop else np.ones(len(t), bool)
    vs = np.round(np.asarray(v(H), float), 6)
    cs = np.asarray(i(H), float) if i else np.zeros(len(t))
    if no_current:
        header = header.rsplit(',', 1)[0]
    head = header + (',' + ','.join(extra) if extra else '')
    pad = ',' * len(extra) if extra else ''
    rows = []
    for tt, vv, cc in zip(t[keep], vs[keep], cs[keep]):
        cur = '' if no_current else ',' + ('' if np.isnan(cc) else f'{cc:.6g}')
        rows += [f'{tt:.6f},1e-05,{vv:.6f}{cur}{pad}\n'] * (2 if dup else 1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fh:
        fh.write(head + '\n')
        fh.writelines(rows)
    return H[keep], vs[keep]


def gas_step(d, step):
    write_four(Path(d) / f'{step}_S_OCV_95Cell-30RH_0.1WH2-0.1WN2.csv', 600, 10, v=0.1)


def xover(d, step):
    write_four(Path(d) / f'{step}_S_H2X_80.3Cell-100RH_0.4WH2-0.4WN2.csv', 720, 0.5,
               v=0.4, i=0.0185)


def flat(level):
    return lambda H: np.full(len(H), level)


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


# The stretch finder and the rate fit on their own.
t1 = np.arange(0.0, 1800.0, 1.0)
check('1,800 samples at 1 s last 30 min', below_runs(t1, np.full(1800, 0.6), 0.7, 600.0),
      [(0, 1799, 1800.0)])
gap_t = np.r_[np.arange(0.0, 600.0), np.arange(1201.0, 1800.0)]
check('a gap longer than the window ends a stretch',
      [(a, b) for a, b, _ in below_runs(gap_t, np.full(len(gap_t), 0.6), 0.7, 600.0)],
      [(0, 599), (600, 1198)])
roll_t = np.r_[np.arange(0.0, 600.0), np.arange(631.0, 1800.0)]
check('a file rollover (31 s) does not', len(below_runs(roll_t, np.full(len(roll_t), 0.6), 0.7, 600.0)), 1)
check('isolated samples last one interval each',
      [s for _, _, s in below_runs(np.arange(10.0) * 10, np.where(np.arange(10) % 2, 0.6, 0.8),
                                   0.7, 600.0)], [10.0] * 5)
check('nothing under the threshold, no stretches', below_runs(t1, np.full(1800, 0.8), 0.7, 600.0), [])
check('no samples, no stretches', below_runs(np.empty(0), np.empty(0), 0.7, 600.0), [])
bins_h = (np.arange(240) + 0.5) / 60.0
rates = rolling_decay(bins_h, 0.9 - 0.001 * bins_h, 0.0, 4.0)
check('a hold shorter than the window gets one window over all of it',
      [(r['start'], r['end']) for r in rates], [(0.0, 4.0)])
close('... with the exact slope, positive while falling', rates[0]['decay'], 1000.0, 1e-6)
rates = rolling_decay((np.arange(720) + 0.5) / 60.0, np.full(720, 0.9), 0.0, 12.0)
check('windows end every hour from H0 + 10 h', [r['end'] for r in rates], [10.0, 11.0, 12.0])
sparse = rolling_decay((np.arange(0, 240, 2) + 0.5) / 60.0, np.full(120, 0.9), 0.0, 4.0)
true('data in half its minutes: no rate', np.isnan(sparse[0]['decay'])
     and abs(sparse[0]['coverage'] - 0.5) < 1e-9, sparse)

# Case 2: OCV drops below 0.7 V mid-hold 4 and stays.
death = tmp / 'death'
xover(death, 'b20')
T_DEATH = 12 * 3600 + 9000.0                     # 2.5 h into hold 4: 14.5 h
for j, step in enumerate(['c2', 'c4', 'c6', 'c8', 'c10']):
    if j:
        gas_step(death, f'c{2 * j + 1}')
    if j < 3:
        prof = lambda H: 0.90 - 1e-7 * H         # noqa: E731
    elif j == 3:
        prof = lambda H: np.where(H < T_DEATH, 0.90, 0.60)  # noqa: E731
    else:
        prof = flat(0.58)
    write_ocv(death / HOLD.format(step), 14400, 10, j * 14400.0, prof)
xover(death, 'c11')
res = analyze_ocv(build_timeline(str(death)))
check('case 2: five holds', len(res.holds), 5)
close('case 2: death within one bin of the built-in time', res.death_h, T_DEATH / 3600, 60 / 3600)
check('... in hold 4', (res.death.segment, res.death.step), (4, 'c8'))
check('... and later stretches never move it', [r.verdict for r in res.runs], ['death', 'after death'])
check('... no recovery', res.recovered_h, None)
check('decay rate from the latest window ending by the death, not the dead trace',
      res.decay_window_h, (8.0, 12.0))
close('... at the 360 μV/h the holds before it fell', res.decay_uV_h, 360.0, 0.5)
true('death note', any(n.startswith('OCV death at 14.50 h in hold 4 (c8)') for n in res.notes),
     res.notes)

# Case 4 and the guards: a transient at a hold's start, a 20-min dip, a
# noisy dry hold, a load pulse, and a logging gap between two dips.
dips = tmp / 'dips'
write_ocv(dips / HOLD.format('c2'), 14400, 10, 0.0, lambda H: np.where(H < 300, 0.30, 0.92))
gas_step(dips, 'c3')
write_ocv(dips / HOLD.format('c4'), 14400, 10, 14400.0,
          lambda H: np.where((H >= 14400 + 5400) & (H < 14400 + 6600), 0.60, 0.91))
gas_step(dips, 'c5')
write_ocv(dips / DRY.format('c6'), 14400, 10, 28800.0,
          lambda H: 0.76 + rng.uniform(-0.08, 0.08, len(H)))
gas_step(dips, 'c7')


def on_load(H):
    return (H >= 43200 + 3600) & (H < 43200 + 6000)


write_ocv(dips / HOLD.format('c8'), 14400, 10, 43200.0,
          lambda H: np.where(on_load(H), 0.55, 0.91), i=lambda H: np.where(on_load(H), 0.5, 0.0))
gas_step(dips, 'c9')


def in_dip(H):
    x = H - 57600
    return ((x >= 3600) & (x < 4800)) | ((x >= 5700) & (x < 6900))


write_ocv(dips / HOLD.format('c10'), 14400, 10, 57600.0,
          lambda H: np.where(in_dip(H), 0.60, 0.91),
          drop=lambda H: (H - 57600 >= 4800) & (H - 57600 < 5700))
res = analyze_ocv(build_timeline(str(dips)))
check('case 4: five holds, no death', (len(res.holds), res.death), (5, None))
h1, h2, h3, h4, h5 = res.holds
check('case 4: 5 min under 0.7 V at a hold start is never judged', (h1.runs, h1.settling), ([], 59))
close('... and the settled start lies past it', h1.start_v, 0.92, 1e-9)
check('a 20-min dip is a dip, not a death', [(r.verdict, round(r.minutes)) for r in h2.runs],
      [('dip', 20)])
close('... dated at its start', h2.runs[0].start_h, (14400 + 5400) / 3600, 60 / 3600)
check('noisy dry hold: the median never trips', h3.runs, [])
true('... though raw samples fall under 0.7 V', h3.raw_below > 100, h3.raw_below)
check('a load pulse is not judged', (h4.on_load, h4.runs), (240, []))
check('a logging gap longer than the window splits a stretch',
      [(r.verdict, round(r.minutes)) for r in h5.runs], [('dip', 20), ('dip', 20)])
true('dip note', any(n.startswith('hold 2 (c4): 10-min median under 0.7 V 1 time(s)')
                     for n in res.notes), res.notes)
true('load note', any(n.startswith('hold 4 (c8): 240 samples drew more than 1 mA')
                      for n in res.notes), res.notes)
check('two conditions, the first hold the reference', [r['Reference'] for r in res.by_condition],
      ['yes', ''])

# Case 5: a hold split across two files, dying across the rollover.
roll = tmp / 'rollover'
write_ocv(roll / HOLD.format('c2'), 14400, 10, 0.0, flat(0.90))
gas_step(roll, 'c3')
T_DIE = 14400 + 13170.0                          # 20 min before c4's log ends
write_ocv(roll / HOLD.format('c4'), 14400, 10, 14400.0, lambda H: np.where(H < T_DIE, 0.90, 0.60))
write_ocv(roll / HOLD.format('c5'), 14400, 10, 28800.0,
          lambda H: np.where(H < 28800 + 1200, 0.60, 0.90))
gas_step(roll, 'c6')
write_ocv(roll / HOLD.format('c7'), 7200, 10, 43200.0, flat(0.90))
tl = build_timeline(str(roll))
res = analyze_ocv(tl)
check('case 5: a hold split across two files is one hold',
      [[f.step for f in s.files] for s in tl.segments], [['c2'], ['c4', 'c5'], ['c7']])
close('... its durations summed', tl.segments[1].H_end_h - tl.segments[1].H_start_h, 8.0, 1e-9)
check('... settling timed once, from the start of the hold', res.holds[1].settling, 59)
close('case 5: death across the rollover, dated at its start', res.death_h, T_DIE / 3600, 60 / 3600)
close('... a 40 min stretch', res.death.minutes, 40.4, 0.3)
close('recovery flagged when the median comes back', res.recovered_h, (28800 + 1200) / 3600, 0.1)
true('... and the death stands', any('Flagged; the death at 7.66 h stands' in n for n in res.notes),
     res.notes)
comeback = tmp / 'comeback'
write_ocv(comeback / HOLD.format('c2'), 14400, 10, 0.0, lambda H: np.where(H < 7200, 0.90, 0.60))
gas_step(comeback, 'c3')
write_ocv(comeback / HOLD.format('c4'), 14400, 10, 14400.0, flat(0.90))
res = analyze_ocv(build_timeline(str(comeback)))
close('a death that holds to the end of its hold', res.death_h, 2.0, 60 / 3600)
close('... is flagged as recovered at the next hold\'s first judged sample', res.recovered_h,
      (14400 + 600) / 3600, 1e-9)
tr = load_hold_trace(tl.segments[1])
check('stitched trace: one rollover, the unlogged tail its only gap',
      (int(np.sum(np.diff(tr.file_idx) != 0)), round(float(np.max(np.diff(tr.t_s))), 6)), (1, 40.0))
tr = load_hold_trace(build_timeline(str(late)).segments[0])
check('a log that started late is placed by its own span', (tr.t_s[0], tr.t_s[-1]), (0.0, 36000.0))
res = analyze_ocv(build_timeline(str(dips)), OcvSettings(settle_min=0))
check('settle_min 0 judges the start transient, as a dip', [r.verdict for r in res.holds[0].runs],
      ['dip'])

# The median, not single samples: noise around 0.67 V still trips it.
noisy = tmp / 'noisy'
write_ocv(noisy / HOLD.format('c2'), 14400, 10, 0.0,
          lambda H: np.where(H < 7200, 0.90, 0.67 + rng.uniform(-0.06, 0.06, len(H))))
res = analyze_ocv(build_timeline(str(noisy)))
true('noisy death: the median trips though raw samples cross back over 0.7 V',
     res.death is not None and 0 <= res.death_h - 2.0 <= 5 / 60, res.death_h)

# Exact values on a linear decline over three files, with a start overshoot
# the settle window must skip and Excel columns in the middle file.
lin = tmp / 'linear'
prof = lambda H: np.where(H < 300, 0.98, 0.95 - 0.001 * H / 3600)  # noqa: E731
written = [write_ocv(lin / HOLD.format(f'c{5 + j}'), 36000, 30, j * 36000.0, prof,
                     extra=['cd', '', ''] if j == 1 else None) for j in range(3)]
H = np.concatenate([w[0] for w in written])
V = np.concatenate([w[1] for w in written])
res = analyze_ocv(build_timeline(str(lin)))
h = res.holds[0]
Hj, Vj = H[H >= 600], V[H >= 600]
start, end = np.median(Vj[Hj < Hj[0] + 1800]), np.median(Vj[Hj >= Hj[-1] - 1800])
close('settled start: median of the first 30 min after the settle window', h.start_v, start, 1e-9)
close('end: median of the last 30 min', h.end_v, end, 1e-9)
close('loss', res.loss_pct, (start - end) / start * 100, 1e-6)
check('settling counted once over three files', (h.settling, h.judged),
      (int((H < 600).sum()), int((H >= 600).sum())))
close('lowest 10-min median: the centred window at the last sample', res.lowest_v,
      np.median(Vj[Hj > Hj[-1] - 300]), 1e-9)
check('windows end every hour from 10 h to 30 h', [round(r['end'], 6) for r in h.rates],
      [float(e) for e in range(10, 31)])
true('every window fits the 1,000 μV/h slope', all(abs(r['decay'] - 1000) < 0.5 for r in h.rates),
     [round(r['decay'], 3) for r in h.rates])
close('OCV decay rate is the last window', res.decay_uV_h, 1000.0, 0.5)
check('... which spans 20-30 h', tuple(round(x, 6) for x in res.decay_window_h), (20.0, 30.0))

# Loss and rate follow the first hold's conditions, even when the run ends
# on another (the FCTS cycle ends on its dry hold).
multi = tmp / 'multi'
for j, (pat, level) in enumerate([(HOLD, 0.92), (DRY, 0.89), (HOLD, 0.91), (DRY, 0.88)]):
    if j:
        gas_step(multi, f'c{2 * j + 1}')
    write_ocv(multi / pat.format(f'c{2 * j + 2}'), 14400, 10, j * 14400.0, flat(level))
res = analyze_ocv(build_timeline(str(multi)))
check('reference: the first hold\'s conditions', res.reference, '95 °C, 30% RH, H2/Air 0.1/0.2 slpm')
check('initial and final at the reference conditions', (res.initial_v, res.final_v), (0.92, 0.91))
close('reference loss', res.loss_pct, 1 / 0.92, 1e-9)
check('every condition in its own row',
      [(r['Conditions'], r['Holds'], r['First settled start (V)'], r['Last end (V)'])
       for r in res.by_condition],
      [('95 °C, 30% RH, H2/Air 0.1/0.2 slpm', 2, 0.92, 0.91),
       ('95 °C, 0% RH, H2/dry Air 0.1/0.2 slpm', 2, 0.89, 0.88)])
close('flat 4 h holds decay at 0 μV/h', res.decay_uV_h, 0.0, 1e-6)
true('reference note', any("first hold's conditions" in n for n in res.notes), res.notes)
# The headline rate is the reference conditions' too: this run ends on a dry
# hold falling 2 mV/h while its reference hold is flat.
mr = tmp / 'multirate'
write_ocv(mr / HOLD.format('c2'), 14400, 10, 0.0, flat(0.92))
gas_step(mr, 'c3')
write_ocv(mr / DRY.format('c4'), 14400, 10, 14400.0, lambda H: 0.89 - 2e-3 * (H - 14400.0) / 3600)
res = analyze_ocv(build_timeline(str(mr)))
check("the headline rate is the reference conditions', not the last hold's",
      (round(res.decay_uV_h, 1),
       [(r['Reference'], r['Latest decay rate (μV/h)']) for r in res.by_condition]),
      (0.0, [('yes', 0.0), ('', 2000.0)]))

bad = tmp / 'badcol'
gas_step(bad, 'c1')
write_ocv(bad / HOLD.format('c2'), 14400, 10, 0.0, flat(0.9),
          header='Time Stamp,Counter Electrode Voltage,Cell Voltage,Current')
res = analyze_ocv(build_timeline(str(bad)))
check('a hold file without a voltage column still counts its hours',
      (len(res.holds), res.holds[0].judged, res.death), (1, 0, None))
true('... and says it could not be read',
     any("could not be read (ValueError: no 'Working Electrode Voltage' column)" in n
         for n in res.notes), res.notes)
check('... and that the rule could not be applied', (res.judged, res.notes[-1]),
      (False, 'no hold has OCV to judge, so the OCV death rule could not be applied'))

# Samples whose current is unknown are not judged; a file with no current
# column at all is taken as at open circuit, and says so.
blank = tmp / 'blank'
write_ocv(blank / HOLD.format('c2'), 14400, 10, 0.0,
          lambda H: np.where((H >= 3600) & (H < 6000), 0.30, 0.92),
          i=lambda H: np.where((H >= 3600) & (H < 6000), np.nan, 0.0))
gas_step(blank, 'c3')
write_ocv(blank / HOLD.format('c4'), 14400, 10, 14400.0, flat(0.92), no_current=True)
res = analyze_ocv(build_timeline(str(blank)))
check('blank current readings are not judged', (res.death, res.holds[0].not_finite), (None, 240))
true('... with a note', any(n.startswith('hold 1 (c2): 240 samples lack a time, voltage or current')
                            for n in res.notes), res.notes)
check('a file without a current column is judged as at open circuit',
      (res.holds[1].judged, res.holds[1].on_load), (len(stamps(14400, 10)) - 59, 0))
true('... with a note', any('has no Current column' in n for n in res.notes), res.notes)

# Duplicated rows do not shorten a stretch: 181 samples 10 s apart under
# 0.7 V make a 30 min median stretch whether or not each row is doubled.
for doubled in (False, True):
    d = tmp / f'dup_{doubled}'
    write_ocv(d / HOLD.format('c2'), 14400, 10, 0.0,
              lambda H: np.where((H >= 3600) & (H <= 5400), 0.60, 0.90), dup=doubled)
    res = analyze_ocv(build_timeline(str(d)))
    check(f'a 30 min stretch is a death, rows doubled {doubled}',
          (res.death is not None and round(res.death.minutes, 6), res.death_h), (30.0, 1.0))

# OCV initial falls back to a later reference hold with data, and says so.
fb = tmp / 'fallback'
write_ocv(fb / HOLD.format('c2'), 14400, 10, 0.0, flat(0.9),
          header='Time Stamp,Counter Electrode Voltage,Cell Voltage,Current')
gas_step(fb, 'c3')
write_ocv(fb / DRY.format('c4'), 14400, 10, 14400.0, flat(0.89))
gas_step(fb, 'c5')
write_ocv(fb / HOLD.format('c6'), 14400, 10, 28800.0, flat(0.91))
res = analyze_ocv(build_timeline(str(fb)))
check('initial and final from the reference hold that has data',
      (res.initial_v, res.initial_from.segment.number, res.final_from.segment.number), (0.91, 3, 3))
true('... with a note', any(n.startswith('OCV initial is the settled start of hold 3 (c6)')
                            for n in res.notes), res.notes)

# run(): plot, readout, workbook and summary on the death and the dip runs.
out = tmp / 'out_death'
res = quiet(run, str(death), str(out), {'image_format': 'png'})
row = res['summary'][0]
close('summary: OCV death', row.get('OCV death'), T_DEATH / 3600, 60 / 3600)
close('run() returns it', res['ocv_death_h'], T_DEATH / 3600, 60 / 3600)
true('summary row is JSON-safe', json.dumps(row, allow_nan=False) is not None)
true('OCV plot', (out / 'fcdur_ocv_vs_time.png').exists())
sc = json.loads((out / '_plot_data' / 'fcdur_ocv_vs_time.json').read_text(), parse_constant=_no_nan)
check('OCV sidecar plot type', sc['plot_type'], 'fcdur_ocv_vs_time')
vals = extract_values(sc)
close('readout: OCV death', (vals.get('OCV death') or {}).get('value'), T_DEATH / 3600, 0.006)
close('readout: OCV initial as in the summary', (vals.get('OCV initial') or {}).get('value'),
      row['OCV initial'], 5e-5)
top = sc['data']['axes'][0]
true('threshold and death lines', any(abs(a['y'] - 0.7) < 1e-12 for a in top['axhlines'])
     and any(abs(a['x'] - res['ocv_death_h']) < 1e-3 for a in top['axvlines']), top['axvlines'])
true('characterization ticks at 0 h and 20 h', {round(a['x'], 4) for a in top['axvlines']} >= {0.0, 20.0},
     top['axvlines'])
wb = openpyxl.load_workbook(out / 'fc_durability.xlsx', read_only=True)
check('events sheet', [(r[0], r[6]) for r in wb['OCV events'].iter_rows(min_row=2, values_only=True)],
      [(4, 'death'), (5, 'after death')])
holds_sheet = list(wb['Holds'].iter_rows(values_only=True))
col = holds_sheet[0].index('OCV death (h)')
check('holds sheet marks the death', [r[col] for r in holds_sheet[1:]],
      [None, None, None, round(res['ocv_death_h'], 4), None])
wb.close()

out = tmp / 'out_dips'
res = quiet(run, str(dips), str(out), {'image_format': 'png'})
row = res['summary'][0]
true('alive: no OCV death in the summary', 'OCV death' not in row and res['ocv_death_h'] is None, row)
check('summary carries the OCV numbers', sorted(k for k in row if k.startswith('OCV')),
      ['OCV decay rate', 'OCV final', 'OCV initial', 'OCV loss', 'OCV min'])
sc = json.loads((out / '_plot_data' / 'fcdur_ocv_vs_time.json').read_text(), parse_constant=_no_nan)
check('readout says no death', extract_values(sc).get('OCV death'), 'none')
pts = sum(len(ln['x']) for ln in sc['data']['axes'][0]['lines'])
true('plot within its point budget', pts <= 2 * (PLOT_MAX_BINS + 5) + 50, pts)
true('dip notes reach run()', any('not a death' in n for n in res['notes']), res['notes'])
res = quiet(run, str(dips), str(tmp / 'out_dips_p'), {'image_format': 'none', 'ocv_persist_min': '15'})
close('ocv_persist_min reaches the rule: the 20-min dip is now a death', res['ocv_death_h'],
      (14400 + 5400) / 3600, 60 / 3600)
res = quiet(run, str(dips), str(tmp / 'out_dips_q'),
            {'image_format': 'none', 'ocv_persist_min': 'abc', 'ocv_window_min': '-5',
             'ocv_threshold': '700', 'geo_area': '-5', 'rate_window_h': '1e300'})
check('unreadable or out-of-range parameters fall back to the defaults',
      (res['ocv_death_h'], res['summary'][0]['geo_area']), (None, 5.0))
check('... each with a note', [n.split(' = ')[0] for n in res['notes'] if n.startswith('parameter ')],
      ['parameter geo_area', 'parameter ocv_threshold', 'parameter ocv_window_min',
       'parameter ocv_persist_min', 'parameter rate_window_h'])

# Pairs of markers or rates never come back from the sidecar as reference
# lines: flat holds, and 10.425 h holds with two equal rate windows.
for name, src in (('multi', multi), ('fcts', tmp / 'fcts')):
    out = tmp / f'out_{name}'
    quiet(run, str(src), str(out), {'image_format': 'png'})
    sc = json.loads((out / '_plot_data' / 'fcdur_ocv_vs_time.json').read_text(),
                    parse_constant=_no_nan)
    check(f'{name}: only the threshold and zero lines are horizontal',
          [len(a['axhlines']) for a in sc['data']['axes']], [1, 1])
true('settled markers are kept as points', len(sc['data']['axes'][0]['scatters']) >= 1)
# Each hold's median line names its conditions, so a stored copy can be read
# per condition (Comparisons): six FCTS holds, two conditions alternating.
named_lines = [ln.get('label') for ln in sc['data']['axes'][0]['lines']
               if ln.get('label') and ln.get('label') != 'bin minimum']
check('fcts: every hold\'s OCV line labelled with its conditions',
      (len(named_lines), sorted(set(named_lines))),
      (6, ['90 °C, 0% RH, H2/dry Air 0.1/0.1 slpm', '90 °C, 30% RH, H2/Air 0.1/0.1 slpm']))
# ... while the legend still names each condition once.
from scripts.fc_durability_analysis import plot_ocv
_tl = build_timeline(str(tmp / 'fcts'))
_fig = plot_ocv(_tl, analyze_ocv(_tl))
_legend = [t.get_text() for t in _fig.axes[0].get_legend().get_texts()]
check('... and the legend names each condition once',
      sorted(t for t in _legend if ' °C' in t),
      ['90 °C, 0% RH, H2/dry Air 0.1/0.1 slpm', '90 °C, 30% RH, H2/Air 0.1/0.1 slpm'])
import matplotlib.pyplot as _plt
_plt.close(_fig)

out = tmp / 'out_bad'
res = quiet(run, str(bad), str(out), {'image_format': 'png'})
sc = json.loads((out / '_plot_data' / 'fcdur_ocv_vs_time.json').read_text(), parse_constant=_no_nan)
check('nothing judged: the readout says so rather than "none"',
      extract_values(sc).get('OCV death'), 'not judged')
check('... and the summary carries no OCV values',
      sorted(k for k in res['summary'][0] if k.startswith('OCV')), [])

out = tmp / 'out_nohold'
quiet(run, str(nohold), str(out), {'image_format': 'png'})
wb = openpyxl.load_workbook(out / 'fc_durability.xlsx', read_only=True)
check('no holds: no OCV sheets and no OCV plot',
      (wb.sheetnames, (out / 'fcdur_ocv_vs_time.png').exists()),
      (['Summary', 'Cycles', 'Holds', 'Characterizations', 'Manifest', 'Notes'], False))
wb.close()

# ── 7. Characterizations and verdict ─────────────────────────────────
print('7. characterizations and verdict')
import pandas as pd

AREA = 5.0
SETPOINTS = (0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35)


def write_cv(path, j_x, R=None, rate=0.01, v_lo=0.12, v_hi=0.80, dt=0.5, cycles=2, c_dl=0.6,
             extra=None, sign=1.0):
    """An H2/N2 crossover CV in the station layout: j = j_x + V/R plus or minus
    double-layer charging (mA/cm², Ω·cm²), so the window mean is j_x plus
    the short's share and the intercept is j_x. sign=-1 logs oxidation as
    negative current."""
    n = int(round((v_hi - v_lo) / rate / dt))
    up = np.linspace(v_lo, v_hi, n + 1)
    V = np.concatenate([np.r_[up, up[-2::-1]][:-1] for _ in range(cycles)] + [up[:1]])
    sweep = np.r_[np.sign(np.diff(V)), -1.0]
    j = sign * (j_x + (V / R * 1000.0 if R else 0.0) + sweep * c_dl)
    t = dt * np.arange(1, len(V) + 1)
    head = FOUR + (',' + ','.join(extra) if extra else '')
    pad = ',' * len(extra) if extra else ''
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fh:
        fh.write(head + '\n')
        fh.writelines(f'{a:.3f},1e-05,{v:.6f},{c * AREA / 1000:.8f}{pad}\n'
                      for a, v, c in zip(t, V, j))


def write_ecsa_cv(path, ecsa_cm2, rate=0.040, seed=1, cycles=3, t_scale=1.0):
    """Three cycles of the portal's synthetic H_UPD CV, logged at `rate` --
    40 mV/s, so a fixed 50 mV/s would read 25% high. t_scale=1000 stamps
    the log in milliseconds."""
    np.random.seed(seed)
    V1, I1 = generate_synthetic_cv(ecsa_cm2=ecsa_cm2, geo_area=AREA, scan_rate=rate)
    V, I = np.tile(V1, cycles), np.tile(I1, cycles)
    t = (0.5 + np.r_[0.0, np.cumsum(np.abs(np.diff(V)) / rate)]) * t_scale
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fh:
        fh.write(FOUR + '\n')
        fh.writelines(f'{a:.6f},1e-05,{v:.6f},{c:.8g}\n' for a, v, c in zip(t, V, I))


def write_geis_sweep(path, r_ohm, seed=1, L=2e-8):
    """A GEIS sweep from 100 kHz to 0.1 Hz in the station layout: an R-RC
    cell (r_ohm in Ω·cm²) behind lead inductance, so Z'' crosses zero."""
    rng = np.random.default_rng(seed)
    freq = np.logspace(5, -1, 61)
    w = 2 * np.pi * freq
    Z = MODELS['R-RC']['func']([r_ohm, 0.15, 0.01, 0.88], w) / AREA + 1j * w * L
    Z = Z + 2e-5 * (rng.standard_normal(len(Z)) + 1j * rng.standard_normal(len(Z)))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fh:
        fh.write(GEIS + '\n')
        for k, (f, z) in enumerate(zip(freq, Z)):
            fh.write(f'{0.5 * (k + 1):.3f},0.038,-0.4990,{f:.5f},{abs(z):.6g},'
                     f'{np.degrees(np.angle(z)):.4f},{z.real:.8g},{z.imag:.8g},1.5\n')


def pol_model(j, loss=0.0):
    return 1.0 - 0.065 * np.log10(1 + j / 2e-4) - (0.12 + loss) * j + 0.03 * np.log(1 - j / 2.0)


def pol_j(v, loss=0.0):
    jj = np.linspace(1e-5, 1.99, 40000)
    return float(np.interp(v, pol_model(jj, loss)[::-1], jj[::-1]))


def write_stepped_pol(path, loss=0.0, dwell=25, dt=1.0, sweeps=3):
    """A potentiostatic pol curve (s, mV, mA, mW): a dwell at each setpoint,
    down and up three times, the current settling within each dwell."""
    order = list(SETPOINTS) + list(SETPOINTS[-2::-1])
    rows, t = [], 0.0
    for v in order * sweeps + [SETPOINTS[0]]:
        jv = pol_j(v, loss)
        for k in range(dwell):
            t += dt
            rows.append((t, round(v * 1000), jv * AREA * 1000 * (1 + 0.02 * np.exp(-k / 4.0))))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fh:
        fh.write(POL + '\n')
        fh.writelines(f'{a:.2f},{mv},{ma:.1f},{mv * ma / 1000:.1f}\n' for a, mv, ma in rows)


def xo(d, step, j_x, R=None, rh=100, flows='0.4WH2-0.4WN2', **kw):
    write_cv(Path(d) / f'{step}_S_H2X_80.3Cell-{rh}RH_{flows}.csv', j_x, R, **kw)


def ec(d, step, ecsa_cm2, **kw):
    write_ecsa_cv(Path(d) / f'{step}_S_ECSA_80.3Cell-100RH_0.05WH2-0.05WN2.csv', ecsa_cm2, **kw)


def eis(d, step, r_ohm, rh=100):
    write_geis_sweep(Path(d) / f'{step}_S_GEIS_80.3Cell-{rh}RH_0.05WH2-0.05WN2.csv', r_ohm)


def pol(d, step, loss=0.0, rh=95, T='80.3', flows='0.2WH2-0.2WA', **kw):
    write_stepped_pol(Path(d) / f'{step}_S_PolarizationCurve_{T}Cell-{rh}RH_{flows}.csv', loss,
                      **kw)


def ast_hold(d, step, k, prof=None):
    """The k-th 2 h hold, from hold hour 2(k - 1)."""
    write_ocv(Path(d) / HOLD.format(step), 7200, 30, 7200.0 * (k - 1), prof or flat(0.9))


def judge(d):
    tl = build_timeline(str(d))
    ca = analyze_characterizations(tl)
    ocv = analyze_ocv(tl)
    return tl, ca, ocv, decide(tl, ocv, ca)


# Case 1: crossover passes 15 mA/cm² between C5 and C6. BOL has a pre-break-in
# repeat and a 50% RH crossover above 15 that must not count (decision 13);
# C5's first repeat is above 15 but its last is not (decision 12); C1's
# crossover line falls with V, so it has no R_short.
xd_dir = tmp / 'xdeath'
xo(xd_dir, 'a1', 4.0)
xo(xd_dir, 'a2', 4.2)
xo(xd_dir, 'a3', 16.0, rh=50)
ec(xd_dir, 'a4', 300.0)
eis(xd_dir, 'a5', 0.045)
pol(xd_dir, 'a6', 0.0)
ast_hold(xd_dir, 'c1', 1)
xo(xd_dir, 'c2', 5.0, R=-5000.0, extra=['cd', '', ''])
ec(xd_dir, 'c3', 280.0)
eis(xd_dir, 'c4', 0.05)
pol(xd_dir, 'c5', 0.02)
for k, (hold_step, steps) in enumerate([('c6', [('c7', 7.0)]), ('c8', [('c9', 9.0)]),
                                        ('c10', [('c11', 11.0)]),
                                        ('c12', [('c13', 16.0), ('c14', 13.0)]),
                                        ('c15', [('c16', 11.0), ('c17', 17.0)])], start=2):
    ast_hold(xd_dir, hold_step, k)
    for step, j_x in steps:
        xo(xd_dir, step, j_x)
ec(xd_dir, 'c18', 250.0)
eis(xd_dir, 'c19', 0.06)
pol(xd_dir, 'c20', 0.05)
tl, ca, ocv, v = judge(xd_dir)
check('characterizations: BOL, then one after each hold', [iv.label for iv in ca.intervals],
      ['BOL', '2 h', '4 h', '6 h', '8 h', '10 h', '12 h'])
by = {x.file.step: x for x in ca.results}
close('crossover: window mean, no short', by['a2'].values['j_xover'], 4.2, 1e-3)
close('... intercept', by['a2'].values['j_intercept'], 4.2, 1e-3)
true('... no R_short from a flat line', np.isnan(by['a2'].values['R_short']), by['a2'].values)
true('... nor from a falling one', np.isnan(by['c2'].values['R_short']), by['c2'].values)
check('the death rule reads only the 100% RH crossover', ca.death_series, ['Crossover, 100% RH'])
check('repeats: the last represents each block',
      [x.file.step for x in ca.death_blocks()], ['a2', 'c2', 'c7', 'c9', 'c11', 'c14', 'c17'])
check('case 1: death at C6, bracket back to C5', (ca.xover_death.H, ca.xover_death.low), (12.0, 10.0))
check('... the verdict', (v.dead, v.hours, v.cause, v.bracket_text),
      (True, 12.0, 'crossover', '(10.00, 12.00] h'))
true('repeats more than 20% apart are flagged',
     any(n.startswith('10 h: Crossover, 100% RH repeats differ by 23%') for n in ca.notes), ca.notes)
check('... repeats 5% apart are not', [n for n in ca.notes if n.startswith('BOL: Crossover')], [])
check('DOE crossover breached at BOL, R_short never', (ca.doe_xover.label, ca.doe_r_short),
      ('BOL', None))

ecsa_csv = pd.read_csv(next(xd_dir.glob('a4_*.csv')))
with contextlib.redirect_stdout(io.StringIO()):
    oracle = compute_ecsa_hupd(ecsa_csv['Working Electrode Voltage'].to_numpy(),
                               ecsa_csv['Current'].to_numpy(), 0.040, AREA, v_low=0.08,
                               v_high=0.40, loading_mg_cm2=0.4, cycle=2)['roughness_factor']
close('ECSA: roughness factor as compute_ecsa_hupd at the logged 40 mV/s', by['a4'].values['RF'],
      oracle, 1e-3 * oracle)
close('... scan rate read from the log', by['a4'].values['scan_rate'], 40.0, 0.1)
close('... ECSA in m²/g from the 0.4 mg/cm² loading', by['a4'].values['ECSA'],
      by['a4'].values['RF'] / 4.0, 1e-9)
ca2 = analyze_characterizations(build_timeline(str(xd_dir)), CharSettings(loading=0.2))
close('... and from whatever loading is given', next(
    x for x in ca2.results if x.file.step == 'a4').values['ECSA'], by['a4'].values['RF'] / 2.0, 1e-9)
geis_csv = pd.read_csv(next(xd_dir.glob('a5_*.csv')))
o = np.argsort(geis_csv['Frequency'].to_numpy())[::-1]
oracle = extract_hfr(*(geis_csv[c].to_numpy()[o] for c in
                       ('Frequency', 'Real Impedance', 'Imaginary Impedance'))) * AREA * 1000
close('EIS: HFR as extract_hfr, in mΩ·cm²', by['a5'].values['HFR'], oracle, 1e-6)
close('... near the 45 mΩ·cm² built in', by['a5'].values['HFR'], 45.0, 2.5)
close('pol curve: V @ 1 A/cm² from the last up-sweep', by['a6'].values['V_1A'],
      float(np.interp(1.0, [pol_j(s) for s in SETPOINTS], SETPOINTS)), 2e-3)
close('... after decay', by['c20'].values['V_1A'],
      float(np.interp(1.0, [pol_j(s, 0.05) for s in SETPOINTS], SETPOINTS)), 2e-3)

# Case 8: C3 has no crossover file; death at C4 brackets back to C2. The
# third hold also loses more than 20% of its OCV without dying (DOE); a
# 10-min dip that deep in the first hold is too short to count.
gap = tmp / 'gap'
xo(gap, 'a1', 3.0)
ast_hold(gap, 'c1', 1, lambda H: np.where((H >= 3600) & (H < 4200), 0.74, 0.95))
xo(gap, 'c2', 5.0)
ast_hold(gap, 'c3', 2, flat(0.95))
xo(gap, 'c4', 8.0)
ast_hold(gap, 'c5', 3, lambda H: np.where(H < 5 * 3600, 0.95, 0.74))
ec(gap, 'c6', 280.0)
ast_hold(gap, 'c7', 4, flat(0.95))
xo(gap, 'c8', 16.0)
tl, ca, ocv, v = judge(gap)
true('case 8: a characterization without a crossover is skipped, and noted',
     '6 h: no 100% RH crossover, so this characterization is skipped for the crossover rule'
     in ca.notes, ca.notes)
check('... the bracket reaches back to the last crossover', (ca.xover_death.H, ca.xover_death.low),
      (8.0, 4.0))
close('DOE OCV loss: the first 20% loss that persists', ocv.loss_breach_h, 5.0, 1 / 60)
check('... with no OCV death', ocv.death, None)

# Case 6: BOL failure.
bolf = tmp / 'bolfail'
xo(bolf, 'a1', 20.0)
ast_hold(bolf, 'c1', 1)
xo(bolf, 'c2', 21.0)
tl, ca, ocv, v = judge(bolf)
check('case 6: a BOL crossover over 15 is death at 0 h', (v.dead, v.hours, v.cause, v.at_bol,
                                                          v.bracket_text, v.text),
      (True, 0.0, 'crossover', True, 'at BOL', 'dead at BOL (0 h), crossover'))

# Case 7: a low R_short puts the window mean over 15 and leaves the intercept under.
short = tmp / 'short'
xo(short, 'a1', 3.0)
ast_hold(short, 'c1', 1)
xo(short, 'c2', 12.0, R=100.0)
tl, ca, ocv, v = judge(short)
c2 = next(x for x in ca.results if x.file.step == 'c2')
close('case 7: the window mean carries the short', c2.values['j_xover'], 16.25, 0.05)
close('... the intercept removes it', c2.values['j_intercept'], 12.0, 0.01)
close('... R_short', c2.values['R_short'], 100.0, 0.5)
check('case 7: the verdict follows the window mean (decision 3)', (v.dead, v.hours, v.cause),
      (True, 2.0, 'crossover'))
true('... and flags the short-corrected value',
     any('short-corrected intercept 12.00 is not' in n for n in ca.notes), ca.notes)
check('DOE R_short breached there', ca.doe_r_short.label, '2 h')

# Both rules: the OCV trips inside the bracket of a later crossover trip;
# then an OCV trip in an earlier interval than the crossover's.
both = tmp / 'both'
xo(both, 'a1', 3.0)
ast_hold(both, 'c1', 1)
xo(both, 'c2', 5.0)
ast_hold(both, 'c3', 2, lambda H: np.where(H < 3 * 3600, 0.9, 0.6))
xo(both, 'c4', 18.0)
tl, ca, ocv, v = judge(both)
check('both rules in one interval: the OCV time, cause both', (v.cause, round(v.hours, 3)),
      ('both', 3.0))
check('... the OCV time is exact, so no bracket is shown', (v.bracket_text, v.text),
      ('', 'dead at 3.00 h, both'))
first = tmp / 'ocvfirst'
xo(first, 'a1', 3.0)
ast_hold(first, 'c1', 1, lambda H: np.where(H < 3600, 0.9, 0.6))
xo(first, 'c2', 4.0)
ast_hold(first, 'c3', 2, flat(0.6))
xo(first, 'c4', 18.0)
tl, ca, ocv, v = judge(first)
check('OCV first, crossover in a later interval: cause OCV', (v.cause, round(v.hours, 3), v.bracket),
      ('OCV', 1.0, None))
close('DOE OCV loss breach at the OCV drop', ocv.loss_breach_h, 1.0, 1 / 60)

check('evolution keeps the first, the last and evenly between', _pick_curves(10), [0, 3, 6, 9])
check('... and all of a few', _pick_curves(3), [0, 1, 2])

# run() end to end on case 1.
out = tmp / 'out_xdeath'
res = quiet(run, str(xd_dir), str(out), {'image_format': 'png'})
row = res['summary'][0]
check('summary verdict', (row['Verdict'], row['Hours to death'], row['Death cause'],
                          row['Death bracket']), ('dead', 12.0, 'crossover', '(10.00, 12.00] h'))
close('summary j_xover initial is BOL', row['j_xover initial'], 4.2, 1e-3)
close('summary j_xover final is the last block', row['j_xover final'], 17.0, 1e-3)
true('no R_short final from a flat line', 'R_short final' not in row, row)
true('summary row is JSON-safe', json.dumps(row, allow_nan=False) is not None)
plots = ['fcdur_timeline', 'fcdur_ocv_vs_time', 'fcdur_xover_vs_time', 'fcdur_hfr_vs_time',
         'fcdur_ecsa_vs_time', 'fcdur_perf_vs_time', 'fcdur_polcurve_evolution',
         'fcdur_xover_evolution', 'fcdur_nyquist_evolution', 'fcdur_ecsa_evolution']
check('every plot written', sorted(p.stem for p in out.glob('*.png')), sorted(plots))
side = {n: json.loads((out / '_plot_data' / f'{n}.json').read_text(), parse_constant=_no_nan)
        for n in plots}
check('sidecar plot types', [side[n]['plot_type'] for n in plots], plots)
vals = extract_values(side['fcdur_timeline'])
check('timeline readout: the verdict', (vals.get('Verdict'), (vals.get('Hours to death') or {}).get('value'),
                                        vals.get('Death cause'), vals.get('Death bracket')),
      ('dead', 12.0, 'crossover', '(10.00, 12.00] h'))
vals = extract_values(side['fcdur_xover_vs_time'])
close('crossover readout: death', (vals.get('Crossover death') or {}).get('value'), 12.0, 1e-9)
close('... DOE breach at BOL', (vals.get('DOE crossover breach') or {}).get('value'), 0.0, 1e-9)
EVO_CAP = {'fcdur_polcurve_evolution': 40, 'fcdur_xover_evolution': 50,
           'fcdur_nyquist_evolution': 61, 'fcdur_ecsa_evolution': 60}
evo = {n: max(len(ln['x']) for ax in side[n]['data']['axes'] for ln in ax['lines']) for n in EVO_CAP}
true('evolution curves thinned', all(evo[n] <= EVO_CAP[n] for n in EVO_CAP), evo)
size = sum((out / '_plot_data' / f'{n}.json').stat().st_size for n in plots)
true('all sidecars together under 250 KB', size < 250_000, size)
wb = openpyxl.load_workbook(out / 'fc_durability.xlsx', read_only=True)
check('workbook: the verdict heads the summary',
      list(wb['Summary'].iter_rows(max_row=2, values_only=True)),
      [('Item', 'Value'), ('Verdict', 'dead at 12.00 h, crossover, between 10.00 and 12.00 h')])
check('Cycles: one row per characterization', wb['Cycles'].max_row - 1, 7)
check('Characterizations: every file', wb['Characterizations'].max_row - 1, 20)
wb.close()

# Case 3, and the crossover parameters reaching the rule.
res = quiet(run, str(xd_dir), str(tmp / 'out_alive'), {'image_format': 'none', 'xover_threshold': '20'})
row = res['summary'][0]
check('case 3: alive at the hours tested, no Hours to death',
      (row['Verdict'], 'Hours to death' in row, row['Hours tested'], res['verdict']),
      ('alive', False, 12.0, 'alive at 12.00 h'))
res = quiet(run, str(short), str(tmp / 'out_win'),
            {'image_format': 'none', 'xover_window_lo': '0.15', 'xover_window_hi': '0.20'})
check('the crossover window reaches the rule', res['verdict'], 'alive at 2.00 h')
res = quiet(run, str(short), str(tmp / 'out_win2'),
            {'image_format': 'none', 'xover_window_lo': '0.5', 'xover_window_hi': '0.4'})
check('an empty window falls back to 0.35-0.50 V, with a note',
      (res['verdict'], any('is empty' in n for n in res['notes'])),
      ('dead at 2.00 h, crossover, between 0.00 and 2.00 h', True))
res = quiet(run, str(tmp / 'scribner'), str(tmp / 'out_placeholder'), {'image_format': 'png'})
check('placeholder characterization files: alive, crossover rule not judged',
      (res['verdict'], any('crossover rule could not be applied' in n for n in res['notes'])),
      ('alive at 60.00 h', True))
sc = json.loads((tmp / 'out_placeholder' / '_plot_data' / 'fcdur_timeline.json').read_text(),
                parse_constant=_no_nan)
vals = extract_values(sc)
check('timeline readout when alive: Verdict alive as in the summary row, no stray keys',
      (vals.get('Verdict'), sorted(vals)),
      ('alive', ['Hold files', 'Hold segments', 'Hours tested', 'Verdict']))
true('... and the rule that could not be applied is shown',
     any('crossover rule not applied (no data)' in t['text']
         for ax in sc['data']['axes'] for t in ax['texts']), vals)

# ── 8. M3 review guards ──────────────────────────────────────────────
print('8. review guards')

# Two 100% RH crossover series (two flow settings) are judged apart, not as
# repeats of one: the 0.2/0.2 slpm series trips at 4 h while the 0.4/0.4 one
# is under 15 there, and its bracket reaches back to its own last
# measurement (BOL), not the other series' (2 h).
two = tmp / 'twoflows'
xo(two, 'a1', 4.0)
xo(two, 'a2', 4.0, flows='0.2WH2-0.2WN2')
ast_hold(two, 'c1', 1)
xo(two, 'c2', 5.0)
ast_hold(two, 'c3', 2)
xo(two, 'c4', 15.5, flows='0.2WH2-0.2WN2')
xo(two, 'c5', 6.0)
ast_hold(two, 'c6', 3)
xo(two, 'c7', 16.0)
tl, ca, ocv, v = judge(two)
check('two 100% RH crossover series are each judged; the earliest trip counts',
      (len(ca.death_series), ca.xover_death.result.file.step, ca.xover_death.H, ca.xover_death.low),
      (2, 'c4', 4.0, 0.0))
check('... the headline follows the series measured most often',
      (ca.primary_death_series, [x.file.step for x in ca.death_blocks()]),
      ('Crossover, 100% RH, 80.3 °C, H2/N2 0.4/0.4 slpm', ['a1', 'c2', 'c5', 'c7']))
true('... and the notes say so', any(n.startswith('2 100% RH crossover series') for n in ca.notes),
     ca.notes)
f0 = next(f for f in tl.files if f.role == 'crossover')
twin = dc_replace(f0, anode_humid=f0.anode_humid + 'x')     # differs where no name shows it
check('series no name tells apart are counted', sorted(_series_labels([f0, twin]).values()),
      ['Crossover, 100% RH, 80.3 °C, H2/N2 0.4/0.4 slpm (1)',
       'Crossover, 100% RH, 80.3 °C, H2/N2 0.4/0.4 slpm (2)'])

# Died on crossover before any OCV trip: the headline OCV decay rate is the
# latest window ending by the death (5 mV/h), not the faster fall after it.
xr = tmp / 'xrate'
xo(xr, 'a1', 3.0)
ast_hold(xr, 'c1', 1, lambda H: 0.95 - 5e-3 * H / 3600)
xo(xr, 'c2', 5.0)
ast_hold(xr, 'c3', 2, lambda H: 0.95 - 5e-3 * H / 3600)
xo(xr, 'c4', 16.0)
ast_hold(xr, 'c5', 3, lambda H: 0.93 - 0.065 * (H - 14400) / 3600)
xo(xr, 'c6', 17.0)
res = quiet(run, str(xr), str(tmp / 'out_xrate'), {'image_format': 'none'})
close('crossover death first: the decay rate stops there too', res['summary'][0]['OCV decay rate'],
      5000.0, 5.0)
true('... and says so', 'OCV decay rate is from the latest window ending by the crossover death '
     'at 4.00 h (2.00–4.00 h)' in res['notes'], res['notes'])

# A window the CVs do not reach falls back to 0.35-0.50 V, rather than
# leaving the rule unjudged and calling a dead sample alive.
lowcv = tmp / 'lowcv'
xo(lowcv, 'a1', 3.0, v_hi=0.6)
ast_hold(lowcv, 'c1', 1)
xo(lowcv, 'c2', 16.0, v_hi=0.6)
res = quiet(run, str(lowcv), str(tmp / 'out_lowcv'),
            {'image_format': 'none', 'xover_window_lo': '0.7', 'xover_window_hi': '0.8'})
check('a window outside the CVs falls back to 0.35-0.50 V, with a note',
      (res['verdict'],
       'crossover window 0.7–0.8 V lies outside the CVs; used 0.35–0.5 V' in res['notes']),
      ('dead at 2.00 h, crossover, between 0.00 and 2.00 h', True))

# Pol curves at two RH and two temperatures on the same flows: separate,
# readable series with their own readout lines. The 60 °C curve after the
# hold no longer reaches 1 A/cm²: it still represents its block, with j @
# 0.6 V and peak power, and the readout says 'not reached' rather than
# repeating an earlier V @ 1 A/cm².
pols = tmp / 'pols'
pol(pols, 'a1')
pol(pols, 'a2', rh=50)
pol(pols, 'a3', T='60')
ast_hold(pols, 'c1', 1)
pol(pols, 'c2', 0.02)
pol(pols, 'c3', 0.02, rh=50)
pol(pols, 'c4', 0.45, T='60')
out = tmp / 'out_pols'
res = quiet(run, str(pols), str(out), {'image_format': 'png'})
tl = build_timeline(str(pols))
check('pol-curve series apart by RH and by temperature',
      sorted(set(_series_labels([f for f in tl.files if f.role == 'polcurve']).values())),
      ['Pol curve, 50% RH, H2/Air 0.2/0.2 slpm',
       'Pol curve, 95% RH, H2/Air 0.2/0.2 slpm, 60 °C',
       'Pol curve, 95% RH, H2/Air 0.2/0.2 slpm, 80.3 °C'])
vals = extract_values(json.loads((out / '_plot_data' / 'fcdur_perf_vs_time.json').read_text(),
                                 parse_constant=_no_nan))
check('... each with its own readout lines',
      sum(k.startswith(('V @ 1 A/cm² ', 'j @ 0.6 V ')) for k in vals), 12)
close('... 50% RH final from its own curve',
      (vals.get('j @ 0.6 V final, 50% RH, H2/Air 0.2/0.2 slpm') or {}).get('value'),
      pol_j(0.6, 0.02), 5e-3)
check('a last curve short of 1 A/cm² reads not reached',
      vals.get('V @ 1 A/cm² final, 95% RH, H2/Air 0.2/0.2 slpm, 60 °C'), 'not reached')
close('... while its j @ 0.6 V is read',
      (vals.get('j @ 0.6 V final, 95% RH, H2/Air 0.2/0.2 slpm, 60 °C') or {}).get('value'),
      pol_j(0.6, 0.45), 5e-3)
wb = openpyxl.load_workbook(out / 'fc_durability.xlsx', read_only=True)
rows = list(wb['Characterizations'].iter_rows(values_only=True))
wb.close()
c4 = dict(zip(rows[0], next(r for r in rows if r[3] == 'c4')))
check('... it represents its block, unflagged, and says how far it reached',
      (c4['Represents block'], c4['V @ 1 A/cm² (V)'], c4['Error'], c4['Note']),
      ('yes', None, None, f'reaches {pol_j(0.35, 0.45):.2f} A/cm², short of 1 A/cm²'))

# Oxidation logged as negative current: flipped, with a note, same numbers;
# and the crossover scales with the geometric area.
neg = tmp / 'neg.csv'
write_cv(neg, 12.0, R=100.0, sign=-1.0)
nv, _, note = analyze_crossover_file(str(neg), CharSettings())
check('negative oxidation current: flipped and noted, R_short kept',
      (round(nv['j_xover'], 3), round(nv['j_intercept'], 3), round(nv['R_short'], 1), note),
      (16.246, 12.0, 100.0, 'oxidation current logged as negative; sign flipped'))
close('crossover per geometric area', analyze_crossover_file(str(neg), CharSettings(geo_area=2.5))[0]
      ['j_xover'], 2 * nv['j_xover'], 1e-9)

# ECSA: a log stamped in milliseconds gives an implausible scan rate, so 50
# mV/s is assumed, and said; a single-cycle CV is read on its last cycle, and
# said.
ecd = tmp / 'ecsa'
ec(ecd, 'a1', 300.0)
ast_hold(ecd, 'c1', 1)
ec(ecd, 'c2', 300.0, t_scale=1000.0)
ast_hold(ecd, 'c3', 2)
ec(ecd, 'c4', 300.0, cycles=1)
tl, ca, ocv, v = judge(ecd)
by = {x.file.step: x for x in ca.results}
ecsa_csv = pd.read_csv(next(ecd.glob('c2_*.csv')))
with contextlib.redirect_stdout(io.StringIO()):
    oracle = compute_ecsa_hupd(ecsa_csv['Working Electrode Voltage'].to_numpy(),
                               ecsa_csv['Current'].to_numpy(), 0.050, AREA, v_low=0.08,
                               v_high=0.40, loading_mg_cm2=0.4, cycle=2)['roughness_factor']
close('ECSA: an implausible logged scan rate falls back to 50 mV/s', by['c2'].values['RF'], oracle,
      1e-3 * oracle)
check('... said, with the scan rate left blank',
      ('2 h: c2 (ECSA) scan rate from the log (0.04 mV/s) is implausible; assumed 50 mV/s'
       in ca.notes, np.isnan(by['c2'].values['scan_rate'])), (True, True))
check('a single-cycle CV: its last cycle, said',
      (by['c4'].values['cycle'],
       '4 h: c4 (ECSA) only 1 cycle(s); the last was analyzed, not cycle 2' in ca.notes),
      (1.0, True))
close('... the same roughness factor as cycle 2 of three', by['c4'].values['RF'],
      by['a1'].values['RF'], 1e-6 * by['a1'].values['RF'])

# A long run -- BOL and eight characterizations, crossover at two RH, EIS
# at three, pol curves at two flows -- keeps its sidecars small: at most two
# series per evolution plot, four curves per series, each thinned.
long_run = tmp / 'long'


def characterize(d, k, n):
    xo(d, f'{k}1', 3.0 + 0.5 * n)
    xo(d, f'{k}2', 2.0 + 0.4 * n, rh=50)
    ec(d, f'{k}3', 300.0 - 10 * n)
    for m, rh in enumerate((100, 50, 30)):
        eis(d, f'{k}{4 + m}', 0.045 + 0.002 * n, rh=rh)
    pol(d, f'{k}7', 0.01 * n, sweeps=1)
    pol(d, f'{k}8', 0.01 * n, flows='0.1WH2-0.1WA', sweeps=1)


characterize(long_run, 'a', 0)
for n in range(1, 9):
    ast_hold(long_run, f'c{n}0', n)
    characterize(long_run, f'c{n}', n)
out = tmp / 'out_long'
res = quiet(run, str(long_run), str(out), {'image_format': 'png'})
side = {n: json.loads((out / '_plot_data' / f'{n}.json').read_text(), parse_constant=_no_nan)
        for n in EVO_CAP}
check('long run: evolution panels and curves capped',
      {n: (len(side[n]['data']['axes']), max(len(ax['lines']) for ax in side[n]['data']['axes']))
       for n in EVO_CAP},
      {'fcdur_polcurve_evolution': (2, 4), 'fcdur_xover_evolution': (2, 4),
       'fcdur_nyquist_evolution': (2, 4), 'fcdur_ecsa_evolution': (1, 4)})
evo = {n: max(len(ln['x']) for ax in side[n]['data']['axes'] for ln in ax['lines']) for n in EVO_CAP}
true('... and thinned', all(evo[n] <= EVO_CAP[n] for n in EVO_CAP), evo)
size = sum(p.stat().st_size for p in (out / '_plot_data').glob('*.json'))
true('... all sidecars together under 200 KB', size < 200_000, size)

# ── 9. The portal: registry, form, job runner, store ─────────────────
print('9. portal and store')
from concurrent.futures import Future
os.environ['JOBS_DIR'] = str(tmp / 'jobs')
os.environ.update(JSONBIN_API_KEY='test', JSONBIN_COLLECTION_ID='COLL', JSONBIN_INDEX_BIN_ID='INDEX')
with contextlib.redirect_stdout(io.StringIO()):
    import scripts as portal_scripts
    import app.main as portal
    from scripts.helpers import jsonbin
    from scripts.helpers.record import _round_sig
    from scripts.fc_durability_analysis import PARAM_RANGES, run as fcdur_run

check('registered as FC Durability', portal_scripts.SCRIPT_REGISTRY.get('FC Durability') is fcdur_run,
      True)
check('... with its short label', portal_scripts.SCRIPT_SHORT.get('FC Durability'), 'FCDurability')
form = {f['key']: f for f in portal_scripts.SCRIPT_PARAMS.get('FC Durability', [])}
check('form: no "No Images", which would store nothing (V6)',
      [o['value'] for o in form.get('image_format', {}).get('options', [])],
      ['png', 'svg', 'pdf', 'tiff'])
check('form: the stand kept or declared, no Auto-detect (the CSVs are the same on both stands)',
      ([o['value'] for o in form.get('stand', {}).get('options', [])], form.get('stand', {}).get('default')),
      (['keep', 'Scribner 1', 'Scribner 2', 'FCTS 1', 'FCTS 2', 'FCTS 3', 'FCTS 4'], 'keep'))
numbers = {k: f for k, f in form.items() if f['type'] == 'number'}
check('form: a field for every parameter run() reads (bar the station step tail)',
      sorted(numbers), sorted(k for k in PARAM_RANGES if k != 'step_tail_s'))
check('form: defaults as run() has them',
      {k: f['default'] for k, f in numbers.items()},
      {k: PARAM_RANGES[k][0] for k in numbers if k in PARAM_RANGES})
check('form: bounds inside run()\'s ranges, defaults on each field\'s step grid',
      [k for k, f in numbers.items() if k not in PARAM_RANGES
       or not PARAM_RANGES[k][1] <= f['min'] <= f['default'] <= f['max'] <= PARAM_RANGES[k][2]
       or abs((f['default'] - f['min']) / f['step'] - round((f['default'] - f['min']) / f['step']))
       > 1e-6], [])


class Store:
    """JSONBin in memory: bins and the index persist across pushes."""

    def __init__(self):
        self.index, self.bins = {'schema': 2, 'runs': []}, {}

    def __call__(self, url, method='GET', body=None, extra_headers=None):
        copy = (lambda x: json.loads(json.dumps(x)))
        if method == 'POST':
            bid = f'BIN{len(self.bins) + 1}'
            self.bins[bid] = copy(body)
            return {'record': body, 'metadata': {'id': bid}}
        if method == 'GET':
            bid = next((b for b in self.bins if f'/{b}/' in url), None)
            return {'record': copy(self.bins[bid] if bid else self.index)}
        if method == 'PUT':
            bid = next((b for b in self.bins if url.endswith(f'/{b}')), None)
            if bid:
                self.bins[bid] = copy(body)
            else:
                self.index = copy(body)
            return {'record': body}
        raise AssertionError(method)


store = Store()
jsonbin._request = store


def submit(job_id, src, params, keep=lambda rel: True):
    """One job through the portal's own runner and completion callback."""
    d = portal.JOBS_DIR / job_id
    (d / 'output').mkdir(parents=True)
    for p in sorted(Path(src).rglob('*')):
        if p.is_file() and keep(p.relative_to(src)):
            (d / 'input' / p.relative_to(src)).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, d / 'input' / p.relative_to(src))
    names = [str(p.relative_to(d / 'input')) for p in sorted((d / 'input').rglob('*')) if p.is_file()]
    portal.jobs[job_id] = {'job_id': job_id, 'status': 'running', 'script': 'FC Durability',
                           'input_files': names, 'params': dict(params)}
    fut = Future()
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            fut.set_result(portal._run_job(job_id, 'FC Durability', str(d / 'input'),
                                           str(d / 'output'), dict(params)))
        except Exception as e:                       # noqa: BLE001
            fut.set_exception(e)
        portal._on_job_done(job_id, fut)
    return portal.jobs[job_id]


DECISION_7 = ('Hours tested', 'Hours to death', 'OCV initial', 'OCV decay rate', 'OCV loss',
              'j_xover initial', 'j_xover final', 'R_short final')


def indexed(row):
    """What the index should hold for a summary row: the plan's decision-7
    keys it has, at six significant figures."""
    return {k: _round_sig(row[k]) for k in DECISION_7 if k in row}


SAMPLE = '260601_DUR-1'
params = {'sample_name': SAMPLE, 'image_format': 'png', 'stand': 'FCTS 2'}
# The first push is case 1 cut at 8 h, still alive; the second is the whole
# test, dead at 12 h: the same sample pushed again with more data.
early = lambda rel: (parse_step_name(rel.name) or {}).get('letter') == 'a' or \
    int((parse_step_name(rel.name) or {}).get('number') or 99) <= 11
j1 = submit('job-1', xd_dir, params, keep=early)
push = j1.get('metrics_push') or {}
check('first push: stored as a new sample', (j1['status'], push.get('pushed'), push.get('merged')),
      ('complete', True, False))
outs = j1['output_files']
check('every output named {sample}-FCDurability-..., none twice',
      (all(o.startswith(f'{SAMPLE}-FCDurability-') for o in outs), len(set(outs)) == len(outs),
       len(outs)), (True, True, 11))
entry = store.index['runs'][0]
row1 = j1['script_result']['summary'][0]
check('index: one step-less fc_durability unit',
      [(u['Analysis'], u['step']) for u in entry['Data']], [('fc_durability', '')])
check('... with the summary row\'s values', entry['Data'][0].get('key_values'), indexed(row1))
check('... and no Hours to death while alive',
      ('Hours to death' in entry['Data'][0].get('key_values', {}), row1['Verdict']),
      (False, 'alive'))
check('... stand as declared, run date from the name', (entry.get('stand'), entry.get('run_date')),
      ('FCTS 2', '2026-06-01'))
rec = store.bins[entry['bin_id']]
check('record: all ten plots in one bucket', {b: len(p) for b, p in rec['metrics'].items()},
      {'fc_durability': 10})
check('... one summary row', len(rec.get('summary') or []), 1)

j2 = submit('job-2', xd_dir, params)
push = j2.get('metrics_push') or {}
check('re-push: merged into the same sample', (push.get('pushed'), push.get('merged'),
                                               push.get('bin_id'), len(store.index['runs'])),
      (True, True, entry['bin_id'], 1))
entry = store.index['runs'][0]
row2 = j2['script_result']['summary'][0]
check('... still one unit, now with the dead run\'s values',
      [(u['Analysis'], u['step'], u.get('key_values')) for u in entry['Data']],
      [('fc_durability', '', indexed(row2))])
check('... Hours to death 12 h', entry['Data'][0].get('key_values', {}).get('Hours to death'), 12.0)
rec = store.bins[entry['bin_id']]
check('... one summary row, the new one, after two pushes',
      [(r.get('Label'), r.get('Verdict')) for r in rec.get('summary') or []], [('fcdur', 'dead')])
check('... ten plots, replaced rather than added', {b: len(p) for b, p in rec['metrics'].items()},
      {'fc_durability': 10})
check('... both jobs recorded, n_jobs 2', ([j['job_id'] for j in rec.get('jobs') or []],
                                          entry.get('n_jobs')), (['job-1', 'job-2'], 2))
check('... every sidecar kept, within the body limit',
      (push.get('sidecars_kept'), push.get('sidecars_dropped'),
       push.get('body_bytes', 1e9) < jsonbin.MAX_BODY_BYTES), (10, 0, True))

# Re-analysed with a 900 mA/cm² threshold the sample is alive again, pushed
# on the form's default, Keep stored stand: nothing of the dead verdict may
# survive, and the stand stays as stored.
j3 = submit('job-3', xd_dir, dict(params, xover_threshold='900', stand='keep'))
entry = store.index['runs'][0]
rec = store.bins[entry['bin_id']]
row3 = j3['script_result']['summary'][0]
check('dead back to alive: Hours to death gone from the index and the record',
      ([(u['Analysis'], u.get('key_values')) for u in entry['Data']],
       [r.get('Verdict') for r in rec.get('summary') or []]),
      ([('fc_durability', indexed(row3))], ['alive']))
check('... "Keep stored stand" keeps it, in both', (entry.get('stand'), rec.get('stand')),
      ('FCTS 2', 'FCTS 2'))
# A stand chosen on a re-push replaces the stored one (V5), across families.
submit('job-3b', xd_dir, dict(params, xover_threshold='900', stand='Scribner 2'))
entry = store.index['runs'][0]
rec = store.bins[entry['bin_id']]
check('a stand chosen on a re-push replaces it, in both (V5)',
      (entry.get('stand'), rec.get('stand'), len(store.index['runs'])), ('Scribner 2', 'Scribner 2', 1))
# A new sample pushed on "Keep stored stand" has no stand to keep, and its
# files cannot name one: none is recorded (the page asks for one first).
# The page asks the store whether a sample is in it just before uploading, so
# the runs list it reads must not be the 30 s cache, which here was filled
# before the push.
import asyncio
import inspect


def listed(force):
    with contextlib.redirect_stdout(io.StringIO()):
        out = portal.view_runs(sample=None, script=None, analysis=None, stand=None,
                               since=None, until=None, limit=None, force=force)
        if inspect.iscoroutine(out):
            out = asyncio.run(out)
    return {r['sample_name'] for r in out.get('runs', [])}


# Reading JSONBin blocks; the page asks before every upload on Keep, so the
# endpoint runs in FastAPI's thread pool rather than on the event loop.
check('the runs list is read off the event loop', inspect.iscoroutinefunction(portal.view_runs),
      False)


listed(False)
submit('job-new', xd_dir, dict(params, sample_name='260603_DUR-NEW', stand='keep'))
fresh = next((e for e in store.index['runs'] if e['sample_name'] == '260603_DUR-NEW'), {})
check('a new sample on "Keep stored stand": no stand recorded, none guessed from the CSVs',
      (fresh.get('stand'), store.bins.get(fresh.get('bin_id'), {}).get('stand'),
       bool(fresh.get('Data'))), (None, None, True))
import time
from scripts.helpers import viewstore as _vs
_vs._index_cache['at'] = time.time()        # still the pre-push cache, however long the job ran
check('the runs list read fresh on request shows a sample pushed a moment ago; the cache does not',
      ('260603_DUR-NEW' in listed(False), '260603_DUR-NEW' in listed(True)), (False, True))

# Comparisons ask for a durability plot by its type: the unit is step-less and
# holds ten plots, and (analysis, step) alone answers with the first stored.
from scripts.helpers import viewstore
TYPES = ('fcdur_ocv_vs_time', 'fcdur_xover_vs_time', 'fcdur_hfr_vs_time', 'fcdur_ecsa_vs_time',
         'fcdur_perf_vs_time')
key = store.index['runs'][0]['bin_id']
with contextlib.redirect_stdout(io.StringIO()):
    got = viewstore.plot_series_batch(
        [{'key': key, 'analysis': 'fc_durability', 'step': '', 'plot_type': t} for t in TYPES]
        + [{'key': key, 'analysis': 'fc_durability', 'step': '', 'plot_type': 'fcdur_nothing'}])
check('series by plot type: each durability trajectory, the unknown type missing',
      ([(s['plot_type'], s['plot'].endswith(t)) for s, t in zip(got['series'], TYPES)],
       [m.get('plot_type') for m in got['missing']]),
      ([(t, True) for t in TYPES], ['fcdur_nothing']))
xo = next((s for s in got['series'] if s['plot_type'] == 'fcdur_xover_vs_time'), {'axes': [{}]})
mean = next((ln for ln in xo['axes'][0].get('lines', []) if 'window mean' in ln['label']), {})
check('... the crossover series against hold hours, BOL to the last characterization',
      (mean.get('label'), round((mean.get('x') or [0])[0], 6), round((mean.get('x') or [0])[-1], 6)),
      ('Crossover, 100% RH, window mean', 0.0, 12.0))

# A sample named ..._fcdur_... that also holds a stepped analysis: every plot
# name contains the row label 'fcdur', and the durability row must still be
# replaced on each push rather than kept beside the new one.
FCD = '260602_fcdur_MEA7'
pc = tmp / 'pc_out' / '_plot_data'
pc.mkdir(parents=True)
(pc / f'{FCD}-FCAnalysis-polcurve_c03_80C.json').write_text(json.dumps(
    {'plot_type': 'polcurve', 'data': {'axes': [{'texts': [{'text': 'OCV = 0.95 V'}],
                                                 'axhlines': [], 'axvlines': [], 'lines': []}]}}))
quiet(jsonbin.push_job_metrics, job_id='job-pc', sample_name=FCD, script='Fuel Cell Full Analysis',
      output_dir=tmp / 'pc_out', input_files=['c03.csv'], script_short='FCAnalysis',
      summary=[{'Analysis': 'polcurve', 'Label': 'polcurve_c03', 'OCV': 0.95}])
submit('job-4', xd_dir, dict(params, sample_name=FCD))
j5 = submit('job-5', xd_dir, dict(params, sample_name=FCD, xover_threshold='900'))
entry = next(e for e in store.index['runs'] if e['sample_name'] == FCD)
rec = store.bins[entry['bin_id']]
check('"fcdur" in the sample name: one durability row after re-analysis, the other kept',
      ([(r['Analysis'], r.get('Verdict')) for r in rec.get('summary') or []],
       sorted((u['Analysis'], u['step']) for u in entry['Data']),
       next((u for u in entry['Data'] if u['Analysis'] == 'fc_durability'), {}).get('key_values')),
      ([('polcurve', None), ('fc_durability', 'alive')], [('fc_durability', ''), ('polcurve', 'c03')],
       indexed(j5['script_result']['summary'][0])))

shutil.rmtree(tmp, ignore_errors=True)
print(f'\n{passed} passed, {failed} failed')
sys.exit(1 if failed else 0)
