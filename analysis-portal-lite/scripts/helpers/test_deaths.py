"""Dismissing a false OCV death: the rules the analysis and the store share
(deaths.py), the analysis judged with an admin's dismissals, the rewrite of
a stored record, the admin's dismiss, restore and settle (admin.py), reruns
that keep a dismissal or flag it for review, and what Recall is shown and
drawn.

Synthetic runs in the station format, in a temp dir: the repository is
public, so no lab data. The main run is test_fc_durability's false death --
the gas out for 60 min in hold 2 (the OCV death at 5.50 h), a 20-min dip in
hold 3, the cell dead for good in hold 5 (a later death at 18.00 h). A
second dies on crossover between 2 and 4 h with an OCV death inside that
bracket, so its cause is 'both' until the OCV death is dismissed.

Run on its own:  PYTHONPATH=. python scripts/helpers/test_deaths.py
"""
import asyncio
import contextlib
import copy
import io
import itertools
import json
import os
import runpy
import shutil
import sys
import tempfile
from concurrent.futures import Future
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix='fcdur_deaths_'))
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ['JOBS_DIR'] = str(TMP / 'jobs')
os.environ.update(JSONBIN_API_KEY='test', JSONBIN_COLLECTION_ID='COLL',
                  JSONBIN_INDEX_BIN_ID='INDEX', ADMIN_CODE='code')
import numpy as np
import openpyxl
from matplotlib.colors import to_hex
import matplotlib.pyplot as plt

passed = failed = 0


def check(name, got, want):
    global passed, failed
    if got == want:
        passed += 1
    else:
        failed += 1
        print(f'  FAIL {name}\n       got  {got!r}\n       want {want!r}')


def true(name, cond, detail=''):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        print(f'  FAIL {name} {detail}')


def raises(name, fn, *a, want='', **k):
    """fn(*a, **k) refuses with a ValueError saying `want`."""
    try:
        fn(*a, **k)
    except ValueError as e:
        true(name, want in str(e), f'-- said {e!s}')
        return
    true(name, False, '-- it did not refuse')


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


# Importing the package prints which registered scripts are absent: noise here.
with contextlib.redirect_stdout(io.StringIO()):
    import app.main as portal
    from fastapi import HTTPException
    from scripts import fc_durability_analysis as FD
    from scripts.helpers import admin, deaths as D, jsonbin, record as REC, viewstore
    from scripts.helpers import plot_compare

# ── station-format writers, as test_fc_durability's ──────────────────
FOUR = 'Time Stamp,Counter Electrode Voltage,Working Electrode Voltage,Current'
AREA = 5.0
HOLD = '{}_S_OCV_95Cell-30RH_0.1WH2-0.2WA.csv'


def stamps(programmed_s, dt):
    """Log stamps as the stations write them: from one interval in, ending
    30 s before the step does."""
    return np.arange(dt, programmed_s - 30 + 1e-9, dt)


def write_four(path, programmed_s, dt, v=0.9, i=0.0):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fh:
        fh.write(FOUR + '\n')
        fh.writelines(f'{tt:.6f},1e-05,{v:.6f},{i:.6g}\n' for tt in stamps(programmed_s, dt))


def write_ocv(path, programmed_s, dt, H0_s, v):
    """An OCV step whose voltage is a function of hold-clock seconds, for a
    file that sits at H0_s on the clock."""
    t = stamps(programmed_s, dt)
    vs = np.round(np.asarray(v(H0_s + t), float), 6)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fh:
        fh.write(FOUR + '\n')
        fh.writelines(f'{tt:.6f},1e-05,{vv:.6f},0\n' for tt, vv in zip(t, vs))


def gas_step(d, step):
    write_four(Path(d) / f'{step}_S_OCV_95Cell-30RH_0.1WH2-0.1WN2.csv', 600, 10, v=0.1)


def xover(d, step):
    write_four(Path(d) / f'{step}_S_H2X_80.3Cell-100RH_0.4WH2-0.4WN2.csv', 720, 0.5,
               v=0.4, i=0.0185)


def write_cv(path, j_x, rate=0.01, v_lo=0.12, v_hi=0.80, dt=0.5, cycles=2, c_dl=0.6):
    """An H2/N2 crossover CV: j = j_x (mA/cm²) plus or minus double-layer
    charging, so the window mean is j_x."""
    n = int(round((v_hi - v_lo) / rate / dt))
    up = np.linspace(v_lo, v_hi, n + 1)
    V = np.concatenate([np.r_[up, up[-2::-1]][:-1] for _ in range(cycles)] + [up[:1]])
    j = j_x + np.r_[np.sign(np.diff(V)), -1.0] * c_dl
    t = dt * np.arange(1, len(V) + 1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fh:
        fh.write(FOUR + '\n')
        fh.writelines(f'{a:.3f},1e-05,{v:.6f},{c * AREA / 1000:.8f}\n' for a, v, c in zip(t, V, j))


def xo(d, step, j_x):
    write_cv(Path(d) / f'{step}_S_H2X_80.3Cell-100RH_0.4WH2-0.4WN2.csv', j_x)


# The false death.
FDIR = TMP / 'false_death'
OUT0, OUT1 = 4 * 3600 + 5400.0, 4 * 3600 + 9000.0      # hold 2: the gas out 1.5-2.5 h in
DIP0, DIP1 = 8 * 3600 + 3600.0, 8 * 3600 + 4800.0      # hold 3: a 20-min dip
DIE = 16 * 3600 + 7200.0                                # hold 5: dead from 2 h in


def fd_ocv(H):
    v = 0.95 - (H / 3600.0) * 400e-6                    # 400 μV/h
    v = np.where((H >= OUT0) & (H < OUT1), 0.30, v)
    v = np.where((H >= DIP0) & (H < DIP1), 0.60, v)
    return np.where(H >= DIE, 0.55, v)


xover(FDIR, 'b20')
for j_, step_ in enumerate(['c2', 'c4', 'c6', 'c8', 'c10']):
    if j_:
        gas_step(FDIR, f'c{2 * j_ + 1}')
    write_ocv(FDIR / HOLD.format(step_), 14400, 10, j_ * 14400.0, fd_ocv)
xover(FDIR, 'c11')

# Dead on crossover between 2 and 4 h (16 mA/cm² after hold 2), the gas out
# in hold 2 from 3 h: an OCV death inside the crossover's bracket. Dead for
# good in hold 3 from 4.5 h, after the crossover death.
XDIR = TMP / 'xover_both'
xo(XDIR, 'a1', 3.0)
write_ocv(XDIR / HOLD.format('c1'), 7200, 30, 0.0, lambda H: 0.95 - 5e-3 * H / 3600)
xo(XDIR, 'c2', 5.0)
write_ocv(XDIR / HOLD.format('c3'), 7200, 30, 7200.0,
          lambda H: np.where((H >= 3 * 3600) & (H < 3 * 3600 + 2400), 0.30,
                             0.95 - 5e-3 * H / 3600))
xo(XDIR, 'c4', 16.0)
write_ocv(XDIR / HOLD.format('c5'), 7200, 30, 14400.0,
          lambda H: np.where(H >= 4.5 * 3600, 0.50, 0.92))
xo(XDIR, 'c6', 17.0)


def rgba3(c):
    return tuple(round(x, 3) for x in D._rgba(c))


def vlines(ax, lo=-1e9, hi=1e9):
    """An axis' vertical lines as stored: x, style, colour, label."""
    return [(round(a['x'], 4), a['linestyle'], tuple(round(c, 3) for c in a['color']), a.get('label'))
            for a in ax.get('axvlines') or [] if lo < a['x'] < hi]


def same_json(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ── 1. the rules (deaths.py) ─────────────────────────────────────────
print('1. the rules')


def ev(start, step, off, folder='', trigger='median', cycles=None):
    return {'start_h': start, 'end_h': start + 1.0, 'minutes': 60.0, 'lowest_v': 0.3,
            'trigger': trigger, 'hold': 2, 'hold_label': f'hold 2 ({step})', 'folder': folder,
            'step': step, 'offset_h': off, 'back_h': start + 1.0, 'recovered_h': start + 1.0,
            'cycles': cycles, 'acted': '',
            'rate': {'uV_h': 100.0 * start, 'window': [start - 4.0, start]}}


def table(events, xd=None, **kw):
    t = {'v': 1, 'hours_tested': 20.0, 'cycles_tested': None, 'ocv_judged': True,
         'xover_judged': True, 'threshold_v': 0.7, 'persist_min': 30.0,
         'rate_uncut': {'uV_h': 50.0, 'window': [16.0, 20.0]}, 'xover': xd, 'events': events}
    t.update(kw)
    return t


def dis(e, **kw):
    return dict(D.new_dismissal(e, 'death', 'gas', 'H2 ran out'), **kw)


def brief(res):
    v = res['verdict']
    return (res['death'], res['later'], res['dismissed'], len(res['unmatched']), v['dead'],
            v['hours'], v['cause'], (res['rate'] or {}).get('uV_h'))


E0, E1 = ev(5.5, 'c4', 1.5), ev(18.0, 'c10', 2.0)
T = table([E0, E1])
d0, d1 = dis(E0), dis(E1)

check('a dismissal names the death that starts in its step file within 10 min of it',
      [D.match(dict(d0, offset_h=o), T['events'])
       for o in (1.5, 1.5 + 10 / 60, 1.5 - 10 / 60, 1.5 + 10.01 / 60, 1.5 - 10.01 / 60)],
      [0, 0, 0, None, None])
check('... and resolve() applies one so, at 10 min and not past it',
      [sorted(D.resolve(T, [dict(d0, offset_h=o)])['by_event'])
       for o in (1.5 + 10 / 60, 1.5 - 10 / 60, 1.5 + 10.01 / 60, 1.5 - 10.01 / 60)],
      [[0], [0], [], []])
check('... in that step and folder only, and not without a time into it',
      [D.match(dict(d0, step='c10'), T['events']), D.match(dict(d0, folder='40H_2'), T['events']),
       D.match(dict(d0, offset_h=None), T['events']),
       D.match(d0, [dict(E0, offset_h=None)])],
      [None, None, None, None])
two = [ev(5.5, 'c4', 1.5), ev(5.6, 'c4', 1.6)]
check('... the nearer of two within it; one already taken passed over',
      (D.match(dict(d0, offset_h=1.58), two), D.match(dict(d0, offset_h=1.52), two),
       D.match(dict(d0, offset_h=1.58), two, taken={1})), (1, 0, 0))
check('the death nearest to a dismissal in its step file, however far; none in another',
      (D.nearest(dict(d0, offset_h=3.9), [E0, E1, ev(7.0, 'c4', 3.0)]),
       D.nearest(dict(d0, offset_h=0.0), [E0, E1, ev(7.0, 'c4', 3.0)]),
       D.nearest(dict(d0, step='c6'), [E0, E1]), D.nearest(dict(d0, offset_h='x'), [E0])),
      (2, 0, None, None))

check('no dismissal: the first death counts, the rest are later deaths',
      brief(D.resolve(T, [])), (0, [1], [], 0, True, 5.5, 'OCV', 550.0))
check('the death dismissed: the next counts, the rate the one ending by it',
      brief(D.resolve(T, [d0])), (1, [], [0], 0, True, 18.0, 'OCV', 1800.0))
check('the later death dismissed: the death stands',
      brief(D.resolve(T, [d1])), (0, [], [1], 0, True, 5.5, 'OCV', 550.0))
check('both dismissed: alive at the hours tested, the rate uncut',
      brief(D.resolve(T, [d1, d0])), (None, [], [0, 1], 0, False, 20.0, '', 50.0))
check('a death dismissed twice: the second dismissal names none',
      brief(D.resolve(T, [d0, dict(d0, id='again')])), (1, [], [0], 1, True, 18.0, 'OCV', 1800.0))
check('a dismissal 30 min off: not applied, the death stands',
      brief(D.resolve(T, [dict(d0, offset_h=2.0)])), (0, [1], [], 1, True, 5.5, 'OCV', 550.0))
check('no rate for the death that counts: none',
      D.resolve(table([dict(E0, rate=None), E1]), [])['rate'], None)

X = {'H': 4.0, 'low': 2.0, 'at_bol': False, 'cycles': None,
     'rate': {'uV_h': 7.0, 'window': [2.0, 4.0]}}
TX = table([ev(3.0, 'c3', 1.0), ev(4.5, 'c5', 0.5)], xd=X)
dx0 = dis(TX['events'][0])
rx = D.resolve(TX, [])
check('an OCV death inside the crossover bracket: both, at the OCV death',
      (rx['verdict']['cause'], rx['verdict']['hours'], rx['verdict']['bracket'],
       D.verdict_text(rx['verdict']), rx['rate']['uV_h']),
      ('both', 3.0, [2.0, 4.0], 'dead at 3.00 h, both', 300.0))
rx = D.resolve(TX, [dx0])
check('... dismissed: the crossover death, before the next OCV death, and its rate',
      (rx['verdict']['cause'], rx['verdict']['hours'], D.verdict_text(rx['verdict']),
       rx['rate']['uV_h'], D.death_h(rx), D.bracket_text(rx['verdict'])),
      ('crossover', 4.0, 'dead at 4.00 h, crossover, between 2.00 and 4.00 h', 7.0, 4.5,
       '(2.00, 4.00] h'))
check('the two clocks within CLOCK_TOL_H are one time: the OCV death, inside the bracket',
      [D.resolve(table([ev(t, 'c5', 0.0)], xd=X), [])['verdict']['cause']
       for t in (4.0 + 5e-10, 4.0 - 5e-10, 4.0 + 2e-9, 2.0)],
      ['both', 'both', 'crossover', 'OCV'])
check('a crossover death at BOL',
      (D.verdict_text(D.resolve(table([], xd=dict(X, H=0.0, low=0.0, at_bol=True)), [])['verdict']),
       D.bracket_text(D.resolve(table([], xd=dict(X, H=0.0, low=0.0, at_bol=True)), [])['verdict'])),
      ('dead at BOL (0 h), crossover', 'at BOL'))
st = D.resolve(table([ev(5.5, 'c4', 1.5, trigger='station')]), [])
check('the station\'s own limit, said in the cause',
      (D.cause_text(st['verdict']), D.verdict_text(st['verdict']), D.cause_text(rx['verdict'])),
      ('OCV (station limit)', 'dead at 5.50 h, OCV (station limit)', 'crossover'))
tc = table([ev(5.5, 'c4', 1.5, cycles=7), ev(18.0, 'c10', 2.0, cycles=30)], cycles_tested=40)
check('RH cycles to death: the death\'s; the cycles tested when alive; none without cycling',
      (D.resolve(tc, [])['verdict']['cycles'],
       D.resolve(tc, [dis(tc['events'][0])])['verdict']['cycles'],
       D.resolve(tc, [dis(e) for e in tc['events']])['verdict']['cycles'],
       D.resolve(T, [])['verdict']['cycles'],
       D.resolve(table([], xd=dict(X, cycles=3), cycles_tested=40), [])['verdict']['cycles']),
      (7, 30, 40, None, 3))

check('the summary row\'s death fields, dead',
      D.summary_fields(D.resolve(T, [d0])),
      {'OCV decay rate': 1800.0, 'OCV death': 18.0, 'Verdict': 'dead', 'Hours to death': 18.0,
       'Death cause': 'OCV', 'OCV deaths dismissed': 1})
check('... alive: no time of death, nothing to death',
      D.summary_fields(D.resolve(T, [d0, d1])),
      {'OCV decay rate': 50.0, 'Verdict': 'alive', 'OCV deaths dismissed': 2})
check('... dead on crossover first: its bracket; the OCV death the next one',
      D.summary_fields(D.resolve(TX, [dx0])),
      {'OCV decay rate': 7.0, 'OCV death': 4.5, 'Verdict': 'dead', 'Hours to death': 4.0,
       'Death cause': 'crossover', 'Death bracket': '(2.00, 4.00] h', 'OCV deaths dismissed': 1})
check('... with RH cycles; none dismissed, none said',
      D.summary_fields(D.resolve(tc, [])),
      {'OCV decay rate': 550.0, 'OCV death': 5.5, 'Verdict': 'dead', 'Hours to death': 5.5,
       'RH cycles to death': 7, 'Death cause': 'OCV'})
check('a rewritten row in the analysis\'s order, keys it does not know after',
      list(D.ordered_row({'zz': 1, 'Verdict': 'dead', 'Analysis': 'fc_durability',
                          'OCV death': 5.0, 'Hours tested': 20.0, 'aa': 2})),
      ['Analysis', 'Hours tested', 'OCV death', 'Verdict', 'zz', 'aa'])

many = table([ev(1.0 + k, 'c4', 0.1 + k) for k in range(13)])
lines = D.ocv_lines(D.resolve(many, [dis(many['events'][2])]))
check('the OCV plot\'s death lines: the death solid red; the first ten later deaths dash-dot, one '
      'legend key saying so; a dismissed death grey and dotted',
      (len(lines), lines[0], [(l['x'], l['ls'], l['color'], l['lw']) for l in lines[1:11]],
       [l['label'] for l in lines[1:11]], lines[-1]),
      (12, {'x': 1.0, 'color': D.DEATH_COLOR, 'ls': '-', 'lw': 1.6, 'label': 'OCV death at 1.00 h'},
       [(x, '-.', D.DEATH_COLOR, 1.3) for x in (2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0)],
       ['OCV death rule met again (flagged), the first 10 of 11 drawn'] + ['_nolegend_'] * 9,
       {'x': 3.0, 'color': D.DISMISSED_COLOR, 'ls': ':', 'lw': 1.6,
        'label': 'OCV death dismissed by an admin'}))
check('... ten or fewer: all drawn, the key as it is; two dismissed, one key',
      ([l['label'] for l in D.ocv_lines(D.resolve(T, []))],
       [l['label'] for l in D.ocv_lines(D.resolve(T, [d0, d1]))],
       D.ocv_lines(D.resolve(table([ev(5.5, 'c4', 1.5, trigger='station')]), []))[0]['label']),
      (['OCV death at 5.50 h', 'OCV death rule met again (flagged)'],
       ['OCV death dismissed by an admin', '_nolegend_'],
       "OCV death at 5.50 h, the station's own limit"))
check('the OCV readout\'s death lines',
      [D.ocv_death_lines(D.resolve(T, s)) for s in ([], [d0], [d0, d1])],
      [['OCV death = 5.50 h', 'Later OCV deaths = 1', 'First later OCV death = 18.00 h'],
       ['OCV death = 18.00 h', 'Later OCV deaths = 0', 'Dismissed OCV deaths = 1'],
       ['OCV death: none', 'Dismissed OCV deaths = 2']])
check('... none judged, said so; the decay rate line, none without a rate',
      (D.ocv_death_lines(D.resolve(table([], ocv_judged=False), [])),
       D.ocv_rate_line(D.resolve(T, [])), D.ocv_rate_line(D.resolve(table([], rate_uncut=None), []))),
      (['OCV death: not judged'], 'OCV decay rate = 550 μV/h', None))
check('the timeline\'s verdict lines',
      [D.verdict_lines(D.resolve(t, s)['verdict'])
       for t, s in ((T, [d0]), (T, [d0, d1]), (table([], ocv_judged=False, xover_judged=False), []),
                    (TX, [dx0]), (tc, []))],
      [['Verdict: dead', 'Hours to death = 18.00 h', 'Death cause: OCV'],
       ['Verdict: alive'],
       ['Verdict: alive', 'OCV rule and crossover rule not applied (no data)'],
       ['Verdict: dead', 'Hours to death = 4.00 h', 'Death cause: crossover',
        'Death bracket: (2.00, 4.00] h'],
       ['Verdict: dead', 'Hours to death = 5.50 h', 'RH cycles to death = 7', 'Death cause: OCV']])
check('... the death line of the timeline and of the RH cycles plot; none when alive',
      (D.verdict_line(D.resolve(tc, [])['verdict']),
       D.verdict_line(D.resolve(tc, [])['verdict'], rh=True)['label'],
       D.verdict_line(D.resolve(T, [d0, d1])['verdict'])),
      ({'x': 5.5, 'color': D.DEATH_COLOR, 'ls': '-', 'lw': 1.6, 'label': 'death at 5.50 h'},
       'death at 5.50 h, RH cycle 7', None))

check('dismissals as given: those naming a step and a time into it',
      [d['id'] for d in D.clean_dismissals(
          [d0, 'x', None, dict(d1, step=''), dict(d1, step=None), dict(d1, step=4),
           dict(d1, offset_h='nan'), dict(d1, offset_h=None), dict(d1, id='ok', offset_h='2.0')])],
      [d0['id'], 'ok'])
check('... none from what is not a list', (D.clean_dismissals({'a': d0}), D.clean_dismissals(None)),
      ([], []))
check('dismissals judge alike in any order, whatever their notes; moved, not',
      (D.canon([d0, d1]) == D.canon([d1, d0]),
       D.canon([d0]) == D.canon([dict(d0, note='another', reason='other', at='x')]),
       D.canon([d0]) == D.canon([dict(d0, offset_h=1.6)]),
       D.canon([d0]) == D.canon([dict(d0, step='c6')]),
       D.canon([d0]) == D.canon([dict(d0, folder='f')]),
       D.canon([d0]) == D.canon([dict(d0, id='other')]), D.canon(None)),
      (True, True, False, False, False, False, []))
check('a day from a stored time',
      [D.day('2026-11-06T11:52:00Z'), D.day('junk'), D.day('2026-13-01T00:00:00Z'), D.day(None)],
      ['6 Nov 2026', '', '', ''])
check('a new dismissal names the death by where it starts',
      {k: d0[k] for k in ('start_h', 'folder', 'step', 'offset_h', 'hold_label', 'kind', 'trigger',
                          'reason', 'note', 'by')},
      {'start_h': 5.5, 'folder': '', 'step': 'c4', 'offset_h': 1.5, 'hold_label': 'hold 2 (c4)',
       'kind': 'death', 'trigger': 'median', 'reason': 'gas', 'note': 'H2 ran out', 'by': 'admin'})
true('... with its own id and the time it was made',
     len(d0['id']) == 12 and d0['id'] != d1['id'] and d0['at'].endswith('Z') and d0['at'][:2] == '20')
check('the reasons in words',
      [D.reason_text({'reason': r}) for r in ('gas', 'station', 'operator', 'other', 'odd', None)],
      ['gas supply ran out', 'station or controller fault', 'operator stop or test setup', 'other',
       'odd', 'other'])


def rec_of(t, ds=()):
    return {'fc_deaths': t, 'ocv_dismissals': list(ds)}


check('what the index keeps of a test\'s deaths',
      [D.index_summary(rec_of(T)), D.index_summary(rec_of(T, [d0])),
       D.index_summary(rec_of(T, [dict(d0, offset_h=2.5)])),
       D.index_summary(rec_of(T, [dict(d0, offset_h=2.5, kept=True)])),
       D.index_summary(rec_of(table([ev(5.5, 'c4', 1.5, trigger='station')])))],
      [{'death_h': 5.5, 'found': 2, 'dismissed': 0, 'dismissals': 0, 'review': 0, 'trigger': 'median'},
       {'death_h': 18.0, 'found': 2, 'dismissed': 1, 'dismissals': 1, 'review': 0, 'trigger': 'median'},
       {'death_h': 5.5, 'found': 2, 'dismissed': 0, 'dismissals': 1, 'review': 1, 'trigger': 'median'},
       {'death_h': 5.5, 'found': 2, 'dismissed': 0, 'dismissals': 1, 'review': 0, 'trigger': 'median'},
       {'death_h': 5.5, 'found': 1, 'dismissed': 0, 'dismissals': 0, 'review': 0, 'trigger': 'station'}])
check('... a test with no OCV death: none found; nothing for a record without the table',
      [D.index_summary(rec_of(table([]))), D.index_summary({}), D.index_summary(rec_of(T, [d0, d1]))
       ['trigger'], D.index_summary(rec_of(table([]), [d0]))],
      [{'death_h': None, 'found': 0, 'dismissed': 0, 'dismissals': 0, 'review': 0, 'trigger': ''},
       None, '', {'death_h': None, 'found': 0, 'dismissed': 0, 'dismissals': 1, 'review': 1,
                  'trigger': ''}])

# Nearest pairs first: of two deaths dismissed 5 min apart, a rerun finding one
# gives it to the dismissal nearest it, whatever their order.
pa, pb = dis(ev(5.5, 'c4', 1.0)), dis(ev(5.583, 'c4', 1.083))
check('dismissals take deaths nearest first, in either order',
      [([d['id'] for d in r['by_event'].values()], [d['id'] for d in r['unmatched']])
       for r in (D.resolve(table([ev(5.58, 'c4', 1.08)]), [pa, pb]),
                 D.resolve(table([ev(5.58, 'c4', 1.08)]), [pb, pa]))],
      [([pb['id']], [pa['id']]), ([pb['id']], [pa['id']])])
check('... two deaths for two: each its nearest',
      sorted((round(table([ev(5.5, 'c4', 1.0), ev(5.58, 'c4', 1.08)])['events'][i]['offset_h'], 2), d['id'])
             for i, d in D.resolve(table([ev(5.5, 'c4', 1.0), ev(5.58, 'c4', 1.08)]),
                                   [pb, pa])['by_event'].items()),
      sorted([(1.0, pa['id']), (1.08, pb['id'])]))
check('the death offered for one not applied: not one dismissed; none in another file; failing '
      'that, the nearest on the clock in any',
      (D.nearest(d0, [E0, ev(7.0, 'c4', 3.0)], taken={0}), D.nearest(d0, [ev(9.0, 'c5', 0.5)]),
       D.nearest_any(dict(d0, start_h=6.0), [ev(9.0, 'c5', 0.5), ev(6.4, 'c5b', 0.1), E0],
                     taken={2}),
       D.nearest_any(dict(d0, start_h=6.0), [ev(9.0, 'c5', 0.5), ev(6.4, 'c5b', 0.1), E0], taken={1})),
      (1, None, 1, 2))
r_ = rec_of(table([ev(6.1, 'c5', 0.1), ev(9.0, 'c5', 3.0)]), [d0])
D.history_add(r_, 'analysed', 'a')
x_ = D.view(r_)['dismissals'][0]
check('... so the view of one whose file has no death left offers the nearest in another',
      (x_['status'], x_['nearest']['i'], x_['nearest']['other_file']), ('review', 0, True))
r_ = rec_of(T, [dict(d0, offset_h=1.8)])
D.history_add(r_, 'analysed', 'a')
check('... one in its file, said so', D.view(r_)['dismissals'][0]['nearest']['other_file'], False)

# A table that keeps the first deaths of more (TABLE_EVENTS_MAX).
cap = table([ev(1.0 + k, 'c4', 0.1 + k) for k in range(12)], events_found=15)
rc = D.resolve(cap, [])
check('a table keeping the first deaths of more: the rest count, as later deaths',
      (rc['found'], rc['beyond'], rc['later_n'], len(rc['later']), D.ocv_death_lines(rc),
       D.ocv_lines(rc)[1]['label'], D.index_summary(rec_of(cap))['found'],
       D.analysis_text(rec_of(cap)).split(': ')[1]),
      (15, 3, 14, 11, ['OCV death = 1.00 h', 'Later OCV deaths = 14', 'First later OCV death = 2.00 h'],
       'OCV death rule met again (flagged), the first 10 of 14 drawn', 15,
       'OCV death at 1.00 h in hold 2 (c4), 14 later deaths'))
rc = D.resolve(cap, [dis(cap['events'][0])])
check('... one dismissed, one later death fewer; with no death found, none',
      (rc['later_n'], D.death_h(rc), D.resolve(table([]), [])['later_n'],
       D.resolve(table([], events_found=None), [])['found']), (13, 2.0, 0, 0))
check('two records\' dismissals as one sample\'s: each once, in the order first seen',
      [d['id'] for d in D.merge_dismissals([d0, d1], [dict(d1, note='seen again'), dis(E0, id='z')])],
      [d0['id'], d1['id'], 'z'])
# Each as the record an admin changed it on last has it, by the two histories.
T1, T2, T3, T4 = ('2026-11-0%dT00:00:00Z' % k for k in (1, 2, 3, 4))
m0 = dis(E0, id='m0', at=T1)
m0v = dict(m0, start_h=6.0, offset_h=2.0, moved_at=T3)


def hh(at, what, did='m0'):
    return {'at': at, 'what': what, 'text': what, 'id': did}


HA = [hh(T1, 'dismissed')]


def mg_(a, b, ha, hb):
    return [(d['id'], d['offset_h'], d.get('note'), d.get('kept', False))
            for d in D.merge_dismissals(a, b, ha, hb)]


check('... moved in the second record: as moved there',
      mg_([m0], [m0v], HA, HA + [hh(T3, 'move')]), [('m0', 2.0, 'H2 ran out', False)])
check('... moved in the first, after the second was made: as moved there',
      mg_([m0v], [m0], HA + [hh(T3, 'move')], HA + [hh(T2, 'analysed', None)]), [('m0', 2.0, 'H2 ran out', False)])
check('... restored, or removed, in either: gone',
      [mg_([m0], [], HA, HA + [hh(T3, 'restored')]), mg_([], [m0], HA + [hh(T3, 'remove')], HA),
       mg_([m0], [m0], HA + [hh(T4, 'restored')], [hh(T2, 'analysed', None)])], [[], [], []])
check('... restored in one, then the other changed later: as changed there',
      mg_([], [dict(m0, kept=True)], HA + [hh(T2, 'restored')], HA + [hh(T3, 'kept')]),
      [('m0', 1.5, 'H2 ran out', True)])
check('... the histories\' last change at the same time: the second\'s',
      mg_([m0], [dict(m0, note='b')], HA, HA), [('m0', 1.5, 'b', False)])
check('... named by neither history: the copy made or moved last, the second\'s on a tie',
      [mg_([m0], [m0v], None, None), mg_([m0v], [m0], [], []), mg_([m0], [dict(m0, note='b')], [], [])],
      [[('m0', 2.0, 'H2 ran out', False)], [('m0', 2.0, 'H2 ran out', False)], [('m0', 1.5, 'b', False)]])
check('... one whose history names a change it holds no copy of: the other\'s copy',
      mg_([m0], [], HA, HA + [hh(T3, 'kept')]), [('m0', 1.5, 'H2 ran out', False)])
check('... kept on file in the first, then applied again by an analysis in the second: kept no more',
      mg_([dict(m0, kept=True)], [m0], HA + [hh(T2, 'kept')],
          [{'at': T3, 'what': 'analysed', 'text': 'a', 'unkept': ['m0']}]),
      [('m0', 1.5, 'H2 ran out', False)])
check('... a change the second\'s history no longer holds: by its copy\'s own time, the move there '
      'stands', (mg_([m0], [m0v], HA, []), mg_([m0v], [m0], [], HA)),
      ([('m0', 2.0, 'H2 ran out', False)], [('m0', 2.0, 'H2 ran out', False)]))
check('... a change said at the second its other copy was made: the change',
      mg_([], [m0], HA + [hh(T1, 'restored')], []), [])
check('... a record\'s copy moved after its own history last named it (a merged record, the move '
      'carried from another): the copy stands', mg_([m0v], None, HA + [hh(T2, 'restored')], None),
      [('m0', 2.0, 'H2 ran out', False)])
check('... a move said like any other change: said at the same second as a change in the first, the '
      'second\'s; at the second the other\'s copy was moved, the move',
      (mg_([dict(m0, kept=True)], [m0v], HA + [hh(T3, 'kept')], HA + [hh(T3, 'move')]),
       mg_([m0v], [dict(m0, note='b', moved_at=T3)], HA + [hh(T3, 'move')], [])),
      ([('m0', 2.0, 'H2 ran out', False)], [('m0', 2.0, 'H2 ran out', False)]))
ex_ = {'metrics': {}, 'summary': [], 'job_id': 'a', 'timestamp': '1', 'ocv_dismissals': [m0],
       'ocv_death_history': [hh(T2, 'restored')]}
check('a push merges none: the stored dismissals stand as they are, whatever their history says',
      [d['id'] for d in REC.merge_detail_record(ex_, {'metrics': {}, 'summary': [], 'job_id': 'b',
                                                      'timestamp': '2'}).get('ocv_dismissals') or []],
      ['m0'])
r_ = dict(rec_of(T, [d0]), ocv_death_history=['junk', {'at': T1, 'what': 'analysed', 'text': 'a'}])
check('a history holding something not an entry (a record edited by hand): read past it',
      (D.analysed_at(r_), [h['text'] for h in D.view(r_)['history']], D.analysis_text(r_)[:11],
       D.kept_through(r_, d0)), (T1, ['a'], 'Re-analysed', False))
D.history_add(r_, 'analysed', 'b', at=T2)
check('the history keeps entries only: anything else dropped as it is written',
      ([h['text'] for h in r_['ocv_death_history']], D.analysed_at(r_), D.view(r_)['analysed_at']),
      (['a', 'b'], T2, T2))
check('... two such merged: each read past the rest, and the merged history its entries only',
      (mg_([m0], [m0v], ['junk'] + HA, [7, hh(T3, 'move')]),
       [h['what'] for h in D.merge_history(['junk'] + HA, [7, hh(T3, 'move')])]),
      ([('m0', 2.0, 'H2 ran out', False)], ['dismissed', 'move']))
r_ = rec_of(T, [d0])
D.history_add(r_, 'analysed', 'first', at=T1)
D.history_add(r_, 'dismissed', 'd', id=d0['id'], at=T2)
D.history_add(r_, 'analysed', 'again', at=T3, unkept=[d0['id']])
for k_ in range(D.HISTORY_MAX):
    D.history_add(r_, 'analysed', f'n{k_}', at=f'2026-11-05T10:{k_:02d}:00Z')
check('the history\'s cap drops the oldest analyses first: the changes a merge goes by outlast them',
      (len(r_['ocv_death_history']), [h['text'] for h in r_['ocv_death_history']][:3],
       r_['ocv_death_history'][-1]['text'], D.last_change(r_['ocv_death_history'], d0['id'])[2],
       len(D.merge_history(r_['ocv_death_history'], [hh(T4, 'kept', 'zz')]))),
      (D.HISTORY_MAX, ['d', 'again', 'n2'], f'n{D.HISTORY_MAX - 1}', 'unkept', D.HISTORY_MAX))
# The merged record judged as a push judges: one kept on file whose death the
# newest analysis has applies again, kept no more; the entry's deaths read off it.
r_ = dict(rec_of(T, [dict(d0, kept=True)]), summary=[{'Analysis': 'fc_durability', 'Verdict': 'dead'}],
          metrics={'fc_durability': {'X-FCDurability-fcdur_timeline': {
              'conditions': {}, 'values': {'Verdict': 'dead', 'Hours to death': 5.5}}}})
e_ = {'Data': [{'Analysis': 'other', 'key_values': {}},
               {'Analysis': 'fc_durability', 'key_values': {'Hours to death': 5.5}}]}
D.judge_merged(r_, e_, 'B1')
check('a merged record judged: kept no more where applied, its row, and the entry\'s units and deaths',
      ('kept' in r_['ocv_dismissals'][0], r_['summary'][0]['Hours to death'],
       [(u['Analysis'], u['key_values'].get('Hours to death')) for u in e_['Data']],
       e_['ocv_deaths']),
      (False, 18.0, [('other', None), ('fc_durability', 18.0)],
       {'death_h': 18.0, 'found': 2, 'dismissed': 1, 'dismissals': 1, 'review': 0, 'trigger': 'median'}))
r_, e_ = {'summary': []}, {'Data': [{'Analysis': 'other'}]}
D.judge_merged(r_, e_, 'B1')
check('... nothing for one without the table', (r_, e_), ({'summary': []}, {'Data': [{'Analysis': 'other'}]}))
H1 = [{'at': '2026-11-01T00:00:00Z', 'what': 'analysed', 'text': 'a'},
      {'at': '2026-11-03T00:00:00Z', 'what': 'dismissed', 'text': 'b', 'id': 'x'}]
H2 = [{'at': '2026-11-02T00:00:00Z', 'what': 'analysed', 'text': 'c'}, dict(H1[0])]
check('... their histories: each entry once, in time order, the last 60',
      ([h['text'] for h in D.merge_history(H1, H2)], [h['text'] for h in D.merge_history(None, H2)],
       len(D.merge_history([dict(H1[0], text=str(k)) for k in range(50)],
                           [dict(H1[1], text=str(k)) for k in range(50)]))),
      (['a', 'c', 'b'], ['a', 'c'], 60))

r_ = rec_of(T)
for k_ in range(D.HISTORY_MAX + 1):
    D.history_add(r_, 'analysed', f'n{k_}', at=f'2026-11-06T10:{k_:02d}:00Z')
check('the death history keeps the last 60',
      (len(r_['ocv_death_history']), r_['ocv_death_history'][0]['text'],
       r_['ocv_death_history'][-1]['text'], r_['ocv_death_history'][-1]['at']),
      (60, 'n1', 'n60', '2026-11-06T10:60:00Z'))
r_ = rec_of(T, [d0, dict(d1, offset_h=0.5)])
D.history_add(r_, 'dismissed', 'x', id='abc', left_out=None)
h_ = r_['ocv_death_history'][-1]
check('an entry: what, the death and verdict after it, the dismissals applied and waiting',
      {k: h_.get(k) for k in ('what', 'text', 'death_h', 'dead', 'verdict', 'applied', 'review', 'id')},
      {'what': 'dismissed', 'text': 'x', 'death_h': 18.0, 'dead': True,
       'verdict': 'dead at 18.00 h, OCV', 'applied': 1, 'review': 1, 'id': 'abc'})
true('... with the time, a None left out', h_['at'].endswith('Z') and 'left_out' not in h_)
D.history_add(r_, 'analysed', 'y', at='2026-11-07T09:00:00Z')
D.history_add(r_, 'analysed', 'y', at='2026-11-08T10:05:00Z')
D.history_add(r_, 'dismissed', 'z')
check('... the time an analysis was stored: the last analysed entry\'s',
      D.analysed_at(r_), '2026-11-08T10:05:00Z')
check('... or the record\'s own without one', D.analysed_at({'timestamp': '2026-11-01T00:00:00Z'}),
      '2026-11-01T00:00:00Z')

r_ = rec_of(T)
check('the history\'s line for an analysis', D.analysis_text(r_),
      'Analysed, 20.00 h tested: OCV death at 5.50 h in hold 2 (c4), 1 later death')
D.history_add(r_, 'analysed', 'first', at='2026-11-04T09:14:00Z')
r_['ocv_dismissals'] = [d0, dict(d1, id='u', offset_h=0.4), dict(d1, id='k', offset_h=0.3, kept=True)]
check('... the next: what became of each dismissal', D.analysis_text(r_),
      'Re-analysed, 20.00 h tested: OCV death at 5.50 h in hold 2 (c4), 1 later death; the death at '
      '5.50 h found again in hold 2 (c4), kept dismissed; no death at 18.00 h in hold 2 (c10): the '
      'dismissal not applied, to review; no death at 18.00 h in hold 2 (c10): the dismissal kept '
      'on file')
check('... none found; three',
      (D.analysis_text(rec_of(table([], hours_tested=12.5))),
       D.analysis_text(rec_of(table([E0, E1, ev(19.0, 'c10', 3.0)])))),
      ('Analysed, 12.50 h tested: no OCV death',
       'Analysed, 20.00 h tested: OCV death at 5.50 h in hold 2 (c4), 2 later deaths'))

r_ = rec_of(T, [dict(d0, at='2026-11-06T11:52:00Z'), dict(d1, id='u', offset_h=0.4),
                dict(d1, id='k', offset_h=0.3, kept=True)])
D.history_add(r_, 'analysed', 'a', at='2026-11-08T10:05:00Z')
v_ = D.view(r_)
check('the view: each death\'s status, each dismissal\'s',
      ([e['status'] for e in v_['events']], [(x['id'], x['status']) for x in v_['dismissals']],
       v_['events'][0]['dismissal']),
      (['dismissed', 'death'], [(d0['id'], 'applied'), ('u', 'review'), ('k', 'kept')], d0['id']))
check('... an applied one names its death; one not applied the nearest in its step',
      (v_['dismissals'][0]['event']['i'], v_['dismissals'][1]['nearest']['i'],
       v_['dismissals'][1]['nearest']['start_h'], v_['dismissals'][1]['nearest']['offset_h'],
       'event' in v_['dismissals'][1], 'nearest' in v_['dismissals'][0]),
      (0, 1, 18.0, 2.0, False, False))
check('... kept through an analysis stored after it', [x['kept_through'] for x in v_['dismissals']],
      [True, False, False])
r_['ocv_dismissals'][0]['moved_at'] = '2026-11-09T08:00:00Z'
check('... not if moved to its death since', D.view(r_)['dismissals'][0]['kept_through'], False)
r_['ocv_dismissals'][0].pop('moved_at')
r_['ocv_dismissals'][0]['at'] = '2026-11-08T10:06:00Z'
check('... nor if made since', D.view(r_)['dismissals'][0]['kept_through'], False)
r_ = rec_of(T, [dict(d0, at='2026-11-08T10:05:00Z')])
D.history_add(r_, 'analysed', 'a', at='2026-11-08T10:05:00Z')
D.history_add(r_, 'dismissed', 'x', at='2026-11-08T10:05:00Z', id=d0['id'])
D.history_add(r_, 'dismissed', 'other', at='2026-11-08T10:05:00Z', id='other')
check('... by the history\'s order where it holds the dismissal: made after the analysis in the '
      'same second, not kept through it', D.view(r_)['dismissals'][0]['kept_through'], False)
D.history_add(r_, 'analysed', 'b', at='2026-11-08T10:05:00Z')
check('... an analysis after it in the same second, kept through',
      D.view(r_)['dismissals'][0]['kept_through'], True)
D.history_add(r_, 'move', 'm', at='2026-11-08T10:05:00Z', id=d0['id'])
check('... moved after that, not', D.view(r_)['dismissals'][0]['kept_through'], False)
r_ = rec_of(T, [dict(d0, at='2026-11-08T10:05:00Z')])
D.history_add(r_, 'analysed', 'a', at='2026-11-08T10:05:00Z')
check('... by the times where it does not: the same second, not kept through',
      D.view(r_)['dismissals'][0]['kept_through'], False)
check('... the verdict, how many wait, the rule\'s settings',
      (v_['verdict'], v_['death_h'], v_['dead'], v_['cause'], v_['hours'], v_['review'],
       v_['threshold_v'], v_['persist_min'], v_['match_min'], v_['hours_tested'], v_['analysed_at']),
      ('dead at 18.00 h, OCV', 18.0, True, 'OCV', 18.0, 1, 0.7, 30.0, 10.0, 20.0,
       '2026-11-08T10:05:00Z'))
check('... nothing for a record without the table', D.view({}), None)

OCV_SC = {'plot_type': D.OCV_PLOT, 'data': {'axes': [
    {'axvlines': [
        {'x': 0.0, 'color': D._rgba('0.55'), 'linestyle': ':', 'label': 'characterization'},
        {'x': 5.5, 'color': D._rgba(D.DEATH_COLOR), 'linestyle': '-', 'label': 'OCV death at 5.50 h'},
        {'x': 18.0, 'color': D._rgba(D.DEATH_COLOR), 'linestyle': '-.', 'label': D.LATER_LABEL}],
     'texts': [{'text': 'OCV initial = 0.9500 V\nOCV decay rate = 550 μV/h\n'
                        'OCV min (10-min median) = 0.3000 V\nOCV death = 5.50 h\n'
                        'Later OCV deaths = 1\nFirst later OCV death = 18.00 h\n'
                        'Hours tested = 20.00 h\nOCV plot bin = 1 min', 'x': 1.0}]},
    {'axvlines': [{'x': 5.5, 'color': D._rgba(D.DEATH_COLOR), 'linestyle': '-', 'label': None}],
     'texts': [{'text': 'OCV plot bin = 1 min'}]}]}}
sc_ = copy.deepcopy(OCV_SC)
D._rewrite_ocv(sc_, D.resolve(T, [d0]))
ax_ = sc_['data']['axes'][0]
check('the OCV plot rewritten: its other lines kept, the death lines drawn anew',
      vlines(ax_),
      [(0.0, ':', rgba3('0.55'), 'characterization'),
       (18.0, '-', rgba3(D.DEATH_COLOR), 'OCV death at 18.00 h'),
       (5.5, ':', rgba3(D.DISMISSED_COLOR), 'OCV death dismissed by an admin')])
check('... its readout: the rate and the death lines in their places, the rest as it was',
      (ax_['texts'][0]['text'].split('\n'), ax_['texts'][0]['x']),
      (['OCV initial = 0.9500 V', 'OCV decay rate = 1800 μV/h', 'OCV min (10-min median) = 0.3000 V',
        'OCV death = 18.00 h', 'Later OCV deaths = 0', 'Dismissed OCV deaths = 1',
        'Hours tested = 20.00 h', 'OCV plot bin = 1 min'], 1.0))
check('... only the upper axis', sc_['data']['axes'][1], OCV_SC['data']['axes'][1])
sc_ = copy.deepcopy(OCV_SC)
D._rewrite_ocv(sc_, D.resolve(table([E0, E1], rate_uncut=None), [d0, d1]))
check('... alive with no rate: no rate line, no death', sc_['data']['axes'][0]['texts'][0]['text'].split('\n'),
      ['OCV initial = 0.9500 V', 'OCV min (10-min median) = 0.3000 V', 'OCV death: none',
       'Dismissed OCV deaths = 2', 'Hours tested = 20.00 h', 'OCV plot bin = 1 min'])
sc_ = {'plot_type': D.OCV_PLOT, 'data': {'axes': [{'axvlines': [], 'texts': [
    {'text': 'OCV death = 5.50 h\nHours tested = 20.00 h\nOCV plot bin = 1 min'}]}]}}
D._rewrite_ocv(sc_, D.resolve(T, []))
check('... a readout without the minimum: the rate before the death',
      sc_['data']['axes'][0]['texts'][0]['text'].split('\n'),
      ['OCV decay rate = 550 μV/h', 'OCV death = 5.50 h', 'Later OCV deaths = 1',
       'First later OCV death = 18.00 h', 'Hours tested = 20.00 h', 'OCV plot bin = 1 min'])
TL_SC = {'plot_type': D.TIMELINE_PLOT, 'data': {'axes': [{'axvlines': [
    {'x': 3.0, 'color': D._rgba('k'), 'linestyle': ':', 'label': None},
    {'x': 5.5, 'color': D._rgba(D.DEATH_COLOR), 'linestyle': '-', 'label': 'death at 5.50 h'}],
    'texts': [{'text': 'Hours tested = 20.00 h\nHold segments = 5\nVerdict: dead\n'
                       'Hours to death = 5.50 h\nDeath cause: OCV'}]}]}}
sc_ = copy.deepcopy(TL_SC)
D._rewrite_timeline(sc_, D.resolve(T, [d0, d1]))
check('the timeline rewritten: its hold lines kept, no death line for a sample alive; the verdict',
      ([(l['x'], l['label']) for l in sc_['data']['axes'][0]['axvlines']],
       sc_['data']['axes'][0]['texts'][0]['text'].split('\n')),
      ([(3.0, None)], ['Hours tested = 20.00 h', 'Hold segments = 5', 'Verdict: alive']))
sc_ = copy.deepcopy(TL_SC)
D._rewrite_timeline(sc_, D.resolve(T, [d0]))
check('... dead later: its line, its lines',
      (vlines(sc_['data']['axes'][0])[1:], sc_['data']['axes'][0]['texts'][0]['text'].split('\n')[2:]),
      ([(18.0, '-', rgba3(D.DEATH_COLOR), 'death at 18.00 h')],
       ['Verdict: dead', 'Hours to death = 18.00 h', 'Death cause: OCV']))
RH_SC = {'plot_type': D.RH_PLOT, 'data': {'axes': [{'axvlines': [
    {'x': 5.5, 'color': D._rgba(D.DEATH_COLOR), 'linestyle': '-', 'label': 'death at 5.50 h, RH cycle 7'}],
    'texts': [{'text': 'RH cycles tested = 40\nRH cycles to death = 7'}]}]}}
sc_ = copy.deepcopy(RH_SC)
D._rewrite_rh(sc_, D.resolve(tc, [dis(tc['events'][0])]))
check('the RH cycles plot rewritten: the death line and its cycles',
      ([(l['x'], l['label']) for l in sc_['data']['axes'][0]['axvlines']],
       sc_['data']['axes'][0]['texts'][0]['text'].split('\n')),
      ([(18.0, 'death at 18.00 h, RH cycle 30')], ['RH cycles tested = 40', 'RH cycles to death = 30']))
sc_ = copy.deepcopy(RH_SC)
D._rewrite_rh(sc_, D.resolve(tc, [dis(e) for e in tc['events']]))
check('... alive: neither', (sc_['data']['axes'][0]['axvlines'], sc_['data']['axes'][0]['texts'][0]['text']),
      ([], 'RH cycles tested = 40'))

ROW = {'Analysis': 'fc_durability', 'Label': 'L', 'Hours tested': 20.0, 'OCV decay rate': 550.0,
       'OCV death': 5.5, 'Verdict': 'dead', 'Hours to death': 5.5, 'Death cause': 'OCV', 'zz': 1}
rec_ = {'fc_deaths': T, 'ocv_dismissals': [d0],
        'summary': [{'Analysis': 'other', 'Verdict': 'x'}, dict(ROW)],
        'metrics': {'fc_durability': {'ocv': {'values': {'stale': 1}}, 'kept': {'values': {'k': 2}}}}}
scs_ = {'ocv': copy.deepcopy(OCV_SC), 'kept': {'plot_type': 'fcdur_other', 'data': {}}}
res_ = D.apply_to_record(rec_, sidecars=scs_)
check('a record brought to its dismissals: the row\'s death fields, the rest kept, in order',
      list(rec_['summary'][1].items()),
      [('Analysis', 'fc_durability'), ('Label', 'L'), ('Hours tested', 20.0), ('OCV decay rate', 1800.0),
       ('OCV death', 18.0), ('Verdict', 'dead'), ('Hours to death', 18.0), ('Death cause', 'OCV'),
       ('OCV deaths dismissed', 1), ('zz', 1)])
check('... another analysis\'s row untouched; the resolution returned',
      (rec_['summary'][0], res_['death']), ({'Analysis': 'other', 'Verdict': 'x'}, 1))
check('... the values read off its rewritten plots, the others\' kept',
      ({k: v for k, v in rec_['metrics']['fc_durability']['ocv']['values'].items()
        if 'death' in k.lower()}, rec_['metrics']['fc_durability']['kept']['values']),
      ({'OCV death': {'value': 18.0, 'unit': 'h'}, 'Later OCV deaths': 0.0, 'Dismissed OCV deaths': 1.0},
       {'k': 2}))
check('... the sidecars it was given, rewritten in place', vlines(scs_['ocv']['data']['axes'][0])[1:],
      [(18.0, '-', rgba3(D.DEATH_COLOR), 'OCV death at 18.00 h'),
       (5.5, ':', rgba3(D.DISMISSED_COLOR), 'OCV death dismissed by an admin')])
rec_ = {'fc_deaths': T, 'ocv_dismissals': [d0], 'summary': dict(ROW)}
REC.attach_sidecars(rec_, {'ocv': copy.deepcopy(OCV_SC)})
D.apply_to_record(rec_)
check('... its own sidecars, decoded and attached again; a lone row',
      (vlines(REC.decode_sidecars(rec_)['ocv']['data']['axes'][0])[1:], rec_['summary']['OCV death']),
      ([(18.0, '-', rgba3(D.DEATH_COLOR), 'OCV death at 18.00 h'),
        (5.5, ':', rgba3(D.DISMISSED_COLOR), 'OCV death dismissed by an admin')], 18.0))
check('... none for a record without the table', D.apply_to_record({'summary': [dict(ROW)]}), None)
ROWS = [dict(ROW)]
rec_ = {'fc_deaths': T, 'ocv_dismissals': [d0], 'summary': ROWS}
D.apply_to_record(rec_, sidecars={})
check('... the list it was given left as it was (a push\'s record shares its job\'s)',
      (ROWS[0]['OCV death'], rec_['summary'][0]['OCV death'], rec_['summary'] is ROWS),
      (5.5, 18.0, False))
PLOTS = (('x-FCDurability-fcdur_ocv_vs_time', OCV_SC, D._rewrite_ocv),
         ('x-FCDurability-fcdur_timeline', TL_SC, D._rewrite_timeline),
         ('x-FCDurability-fcdur_rh_cycles', RH_SC, D._rewrite_rh))
for t_, S_ in ((T, [d0]), (T, [d0, d1]), (tc, [dis(tc['events'][0])]), (TX, [dx0])):
    rec_ = {'fc_deaths': t_, 'ocv_dismissals': S_, 'metrics': {'fc_durability': {
        n: {'values': REC.extract_values(sc)} for n, sc, _ in PLOTS}}}
    D.apply_to_record(rec_, sidecars={})
    want = {}
    for n, sc, fn in PLOTS:
        sc = copy.deepcopy(sc)
        fn(sc, D.resolve(t_, S_))
        want[n] = REC.extract_values(sc)
    check(f'... its plots\' data not stored ({len(S_)} dismissed): their values as their readouts '
          f'rewritten give them', {n: e['values'] for n, e in rec_['metrics']['fc_durability'].items()},
          want)

# ── 2. the analysis judged with dismissals ──────────────────────────
print('2. the analysis judged with dismissals')
tl = FD.build_timeline(str(FDIR))
ocv0 = FD.analyze_ocv(tl)
ca = FD.analyze_characterizations(tl)
tab = FD.deaths_table(tl, ocv0, ca, FD.decide(tl, ocv0, ca))
check('the table: each death and later death, named by where it starts',
      [(round(e['start_h'], 4), e['folder'], e['step'], round(e['offset_h'], 4), e['hold'],
        e['hold_label'], e['trigger']) for e in tab['events']],
      [(5.4972, '.', 'c4', 1.4972, 2, 'hold 2 (c4)', 'median'),
       (18.0, '.', 'c10', 2.0, 5, 'hold 5 (c10)', 'median')])
check('... how long and how low, when the OCV was back, the rate ending by each',
      [(round(e['minutes']), round(e['lowest_v'], 3), e['back_h'] and round(e['back_h'], 2),
        e['rate']['window'], round(e['rate']['uV_h'], 1)) for e in tab['events']],
      [(60, 0.3, 6.5, [0.0, 4.0], 400.0), (120, 0.55, None, [12.0, 16.0], 400.0)])
check('... what was tested, the rule\'s settings; no crossover judged (no CV in these stubs)',
      (tab['hours_tested'], tab['cycles_tested'], tab['ocv_judged'], tab['xover_judged'],
       tab['threshold_v'], tab['persist_min'], tab['xover'], tab['v']),
      (20.0, None, True, False, 0.7, 30.0, None, 1))
e0, e1 = tab['events']
dz0 = D.new_dismissal(e0, 'death', 'gas', 'H2 cylinder ran empty')
dz1 = D.new_dismissal(e1, 'later death', 'station', 'controller fault')
DAY = D.day(dz0['at'])
o1 = FD.analyze_ocv(tl, None, [dz0])
check('the death dismissed: the later death counts; the dip still a dip',
      ([(r.verdict, round(r.start_h, 2)) for r in o1.runs], round(o1.death_h, 2), o1.later_deaths,
       [round(r.start_h, 2) for r in o1.dismissed_runs], o1.recovered_h,
       [d['id'] for d in o1.dismissals], o1.unmatched),
      ([('dismissed', 5.5), ('dip', 9.0), ('death', 18.0)], 18.0, [], [5.5], None, [dz0['id']], []))
check('... judged as it was, kept', [r.judged_as for r in o1.dismissed_runs], ['death'])
check('... the decay rate from the latest window ending by it',
      (round(o1.decay_uV_h, 1), o1.decay_window_h), (400.0, (12.0, 16.0)))
check('... the notes: the dismissal, then the death that counts now',
      [n for n in o1.notes if 'OCV death' in n or 'dismiss' in n],
      [f'OCV death at 5.50 h in hold 2 (c4) dismissed by an admin on {DAY}: gas supply ran out '
       f'(“H2 cylinder ran empty”). The OCV death is now the next one, at 18.00 h',
       'OCV death at 18.00 h in hold 5 (c10), step c10: 10-min median under 0.7 V for 120 min',
       'OCV decay rate is from the latest window ending by the OCV death at 18.00 h (12.00–16.00 h)'])
o2 = FD.analyze_ocv(tl, None, [dz1])
n2 = [n for n in o2.notes if 'OCV death' in n]
check('the later death dismissed: the death stands, its recovery said; the dismissal after them',
      (round(o2.death_h, 2), o2.later_deaths, n2[0].startswith('OCV death at 5.50 h in hold 2 (c4)'),
       n2[1].startswith('OCV recovered'), n2[2:]),
      (5.5, [], True, True,
       [f'Later OCV death at 18.00 h in hold 5 (c10) dismissed by an admin on {DAY}: station or '
        f'controller fault (“controller fault”)',
        'OCV decay rate is from the latest window ending by the OCV death at 5.50 h (0.00–4.00 h)']))
o3 = FD.analyze_ocv(tl, None, [dz1, dz0])
check('both dismissed: no OCV death; the last note says none follows them',
      (o3.death, [n for n in o3.notes if 'dismissed by an admin' in n]),
      (None, [f'OCV death at 5.50 h in hold 2 (c4) dismissed by an admin on {DAY}: gas supply ran '
              f'out (“H2 cylinder ran empty”)',
              f'OCV death at 18.00 h in hold 5 (c10) dismissed by an admin on {DAY}: station or '
              f'controller fault (“controller fault”). No OCV death follows them']))
o3b = FD.analyze_ocv(tl, None, [dict(dz0, note='  ')])
check('... the only one dismissed, no note: none follows it',
      [n for n in o3b.notes if 'dismissed by an admin' in n],
      [f'OCV death at 5.50 h in hold 2 (c4) dismissed by an admin on {DAY}: gas supply ran out. '
       f'The OCV death is now the next one, at 18.00 h'])
far = dict(dz0, id='far', offset_h=e0['offset_h'] + 0.5)
o4 = FD.analyze_ocv(tl, None, [far])
check('a dismissal 30 min from any death: not applied, said so with the nearest',
      (round(o4.death_h, 2), o4.dismissed_runs, [d['id'] for d in o4.unmatched],
       [n for n in o4.notes if n.startswith('A dismissal')]),
      (5.5, [], ['far'],
       [f'A dismissal of the OCV death at 5.50 h in hold 2 (c4) (gas supply ran out, {DAY}) found '
        f'no death starting in c4 within 10 min of it in this analysis, so it is not applied; an '
        f'admin should review it. The nearest death there starts at 5.50 h']))
o5 = FD.analyze_ocv(tl, None, [dict(far, kept=True), dict(dz0, id='gone', step='c99')])
check('... kept on file, not to review; one in a step the run lacks, no nearest',
      [n for n in o5.notes if n.startswith('A dismissal')],
      [f'A dismissal of the OCV death at 5.50 h in hold 2 (c4) (gas supply ran out, {DAY}) found '
       f'no death starting in c4 within 10 min of it in this analysis, so it is not applied (kept '
       f'on file). The nearest death there starts at 5.50 h',
       f'A dismissal of the OCV death at 5.50 h in hold 2 (c4) (gas supply ran out, {DAY}) found '
       f'no death starting in c99 within 10 min of it in this analysis, so it is not applied; an '
       f'admin should review it'])
o6 = FD.analyze_ocv(tl, None, [{'step': 'c4'}, 'x', {'offset_h': 1.5}])
check('dismissals that name no place: none', (round(o6.death_h, 2), o6.dismissals, o6.unmatched),
      (5.5, [], []))

out1 = TMP / 'out_dismissed'
res1 = quiet(FD.run, str(FDIR), str(out1), {'image_format': 'png', 'ocv_dismissals': [dz0]})
row1 = res1['summary'][0]
check('run(): the verdict and the summary row from the death that counts',
      (res1['verdict'], round(res1['hours_to_death'], 4), round(res1['ocv_death_h'], 4),
       {k: row1.get(k, 'absent') for k in D.SUMMARY_DEATH_KEYS}),
      ('dead at 18.00 h, OCV', 18.0, 18.0,
       {'OCV death': 18.0, 'OCV decay rate': 400.0, 'Verdict': 'dead', 'Hours to death': 18.0,
        'RH cycles to death': 'absent', 'Death cause': 'OCV', 'Death bracket': 'absent',
        'OCV deaths dismissed': 1}))
check('... the table handed to the store: each death as judged, the dismissals judged with',
      ([round(e['start_h'], 4) for e in res1['fc_deaths']['events']],
       [d['id'] for d in res1['fc_deaths']['dismissals']]), ([5.4972, 18.0], [dz0['id']]))
sc1 = json.loads((out1 / '_plot_data' / 'fcdur_ocv_vs_time.json').read_text())
check('... the OCV plot: the dismissed death grey and dotted, the death solid red',
      vlines(sc1['data']['axes'][0], 1.0, 19.5),
      [(18.0, '-', rgba3(D.DEATH_COLOR), 'OCV death at 18.00 h'),
       (5.4972, ':', rgba3(D.DISMISSED_COLOR), D.DISMISSED_LABEL)])
vals1 = REC.extract_values(sc1)
check('... its readout',
      (vals1.get('OCV death'), vals1.get('Later OCV deaths'), vals1.get('Dismissed OCV deaths'),
       vals1.get('First later OCV death'), vals1.get('OCV decay rate')),
      ({'value': 18.0, 'unit': 'h'}, 0.0, 1.0, None, {'value': 400.0, 'unit': 'μV/h'}))
tl1 = json.loads((out1 / '_plot_data' / 'fcdur_timeline.json').read_text())
check('... the timeline\'s death line',
      [v for v in vlines(tl1['data']['axes'][0]) if v[2] == rgba3(D.DEATH_COLOR)],
      [(18.0, '-', rgba3(D.DEATH_COLOR), 'death at 18.00 h')])
wb_ = openpyxl.load_workbook(out1 / 'fc_durability.xlsx', read_only=True)
summ1 = {r[0]: r[1] for r in wb_['Summary'].iter_rows(min_row=2, values_only=True)}
ev1 = [(r[0], r[6]) for r in wb_['OCV events'].iter_rows(min_row=2, values_only=True)]
wb_.close()
check('... the workbook: the dismissed death in its Summary and OCV events sheets',
      (summ1.get('OCV deaths dismissed by an admin (h)'), summ1.get('OCV death (h)'),
       summ1.get('OCV death dismissals not applied', 'no row'), ev1),
      ('5.4972 (gas supply ran out)', 18, 'none',
       [(2, 'death, dismissed by an admin (gas supply ran out)'), (3, 'dip, shorter than 30 min'),
        (5, 'death')]))
res1b = quiet(FD.run, str(FDIR), str(TMP / 'out_far'), {'image_format': 'none',
                                                      'ocv_dismissals': [far]})
wb_ = openpyxl.load_workbook(TMP / 'out_far' / 'fc_durability.xlsx', read_only=True)
summ1b = {r[0]: r[1] for r in wb_['Summary'].iter_rows(min_row=2, values_only=True)}
wb_.close()
check('... one not applied: in the Summary sheet, the death as found',
      (summ1b.get('OCV death dismissals not applied'), summ1b.get('OCV deaths dismissed by an admin (h)',
                                                              'no row'), res1b['verdict']),
      ('5.4972 h in hold 2 (c4) (to review)', 'none', 'dead at 5.50 h, OCV'))
res1c = quiet(FD.run, str(FDIR), str(TMP / 'out_junk'), {'image_format': 'none',
                                                       'ocv_dismissals': 'nonsense'})
check('... dismissals that are not a list: none', (res1c['verdict'], res1c['fc_deaths']['dismissals']),
      ('dead at 5.50 h, OCV', []))

# ── 3. one rule: the store's rewrite is the analysis's own ───────────
# A record stored before a dismissal, rewritten by deaths.py, is what the
# analysis gives when judged with it: every sidecar, the summary row, the
# verdict. For each set of dismissals of the false death's two deaths, and
# of the crossover run's.
print('3. one rule')
res0 = quiet(FD.run, str(FDIR), str(TMP / 'out_plain'), {'image_format': 'png'})
sc0 = REC.load_sidecars(TMP / 'out_plain')
true('the plots a rewrite touches are among those stored',
     {D.OCV_PLOT, D.TIMELINE_PLOT} <= {s.get('plot_type') for s in sc0.values()})
for n_, S in enumerate(([], [dz0], [dz1], [dz1, dz0])):
    tag = ['none', 'the death', 'the later death', 'both'][n_]
    outS = TMP / f'out_set{n_}'
    resS = res1 if n_ == 1 else quiet(FD.run, str(FDIR), str(outS),
                                      {'image_format': 'png', 'ocv_dismissals': S})
    want = REC.load_sidecars(out1 if n_ == 1 else outS)
    rec_ = {'fc_deaths': copy.deepcopy(res0['fc_deaths']), 'ocv_dismissals': S,
            'summary': copy.deepcopy(res0['summary'])}
    got = copy.deepcopy(sc0)
    D.apply_to_record(rec_, sidecars=got)
    check(f'{tag} dismissed: the stored row rewritten is the analysis\'s',
          list(rec_['summary'][0].items()), list(resS['summary'][0].items()))
    check(f'... its plots, every one', sorted(n for n in want if not same_json(got.get(n), want[n])), [])
    check(f'... the verdict', D.verdict_text(D.resolve(res0['fc_deaths'], S)['verdict']),
          resS['verdict'])


def vnorm(v):
    return {k: (round(x, 6) if isinstance(x, float) else
                [round(y, 6) for y in x] if isinstance(x, (list, tuple)) else x) for k, x in v.items()}


tlx = FD.build_timeline(str(XDIR))
cax = FD.analyze_characterizations(tlx)
ox0 = FD.analyze_ocv(tlx)
vx0 = FD.decide(tlx, ox0, cax)
tabx = FD.deaths_table(tlx, ox0, cax, vx0)
check('the crossover run: dead on crossover between 2 and 4 h, an OCV death inside it, a later one',
      (vx0.text, [round(e['start_h'], 2) for e in tabx['events']], tabx['xover']['H'],
       tabx['xover']['low'], tabx['xover']['rate']['window']),
      ('dead at 2.99 h, both', [2.99, 4.5], 4.0, 2.0, [2.0, 4.0]))
dxs = [D.new_dismissal(e, 'death', 'gas', 'n') for e in tabx['events']]
for S in itertools.chain.from_iterable(itertools.combinations(dxs, k) for k in range(3)):
    oS = FD.analyze_ocv(tlx, None, list(S))
    check(f'... {len(S)} dismissed ({", ".join(str(round(d["start_h"], 2)) for d in S) or "none"}): '
          f'the verdict decide() gives is resolve()\'s',
          vnorm(D.resolve(tabx, list(S))['verdict']), vnorm(FD.verdict_dict(FD.decide(tlx, oS, cax))))
resx0 = quiet(FD.run, str(XDIR), str(TMP / 'out_x0'), {'image_format': 'png'})
resx1 = quiet(FD.run, str(XDIR), str(TMP / 'out_x1'), {'image_format': 'png',
                                                     'ocv_dismissals': dxs[:1]})
rec_ = {'fc_deaths': resx0['fc_deaths'], 'ocv_dismissals': dxs[:1],
        'summary': copy.deepcopy(resx0['summary'])}
got = REC.load_sidecars(TMP / 'out_x0')
D.apply_to_record(rec_, sidecars=got)
want = REC.load_sidecars(TMP / 'out_x1')
check('... the OCV death inside the bracket dismissed: dead on crossover, the row and plots as the '
      'analysis has them',
      (resx1['verdict'], list(rec_['summary'][0].items()) == list(resx1['summary'][0].items()),
       sorted(n for n in want if not same_json(got.get(n), want[n]))),
      ('dead at 4.00 h, crossover, between 2.00 and 4.00 h', True, []))

# A Scribner FuelCell RH-cycling hold whose station limit trips twice: a
# death and a later death, each a station's act; the RH cycles plot too.
FCD_COLS = ['Time (Sec)', 'I (A)', 'I (mA/cm²)', 'Power (Watts)', 'Power (mW/cm²)', 'E_Stack (V)',
            'E_Avg (V)', 'E_Comp_Stack (V)', 'E_Comp_Avg (V)', 'E_iR_Stack (V)', 'E_iR_Avg (V)',
            'E_iR_Stack (mOhm)', 'E_iR_Avg (mOhm*cm²)', 'Cell (C)', 'Temp_Anode (C)',
            'Flow_Anode  (l/min)', 'Wet_Anode', 'Temp_Cathode (C)', 'Flow_Cathode  (l/min)',
            'Wet_Cathode', 'HFR (mOhm)', 'HFR (mOhm*cm²)', 'Z_Freq (Hz)', 'Z_Real (Ohm)',
            'Z_Imag (Ohm)', 'Z_Mag (Ohm)', 'Z_Angle (Deg)', 'Ctrl_Status', 'Ctrl_Mode', 'Ctrl_Value',
            'Ctrl_Repeat', 'Ctrl_Temp_Cell (C)', 'Ctrl_Temp_Anode (C)', 'Ctrl_Temp_Cathode (C)',
            'Ctrl_Flow_Anode  (l/min)', 'Ctrl_Flow_Cathode  (l/min)']


def write_fcd(path, t, data, desc):
    """A FuelCell5 step file: its header, then a row per time."""
    t = np.asarray(t, float)
    cols = {c: np.zeros(len(t)) for c in FCD_COLS}
    cols['Time (Sec)'] = t
    for c, v in data.items():
        cols[c] = np.broadcast_to(np.asarray(v, float), t.shape)
    head = ['FUELCELL DATA', 'Format: FuelCell5', 'Time: "10/1/2026 9:00:00 AM"',
            'DateTime: 46296.375', 'comment 1: ""', 'comment 2: ""',
            f'Description: "{desc}; Single File: {Path(path).name}(Append)"', 'SurfaceArea: 5',
            'StackNumber: 1', f'DataColumnCount: {len(FCD_COLS)}']
    head += [f'DataColumn{k}: {k}' for k in range(1, len(FCD_COLS) + 1)]
    head += ['\t'.join(FCD_COLS), 'End Comments']
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        fh.write('\r\n'.join(head) + '\r\n')
        fh.writelines('\t'.join(f'{cols[c][k]:.10g}' for c in FCD_COLS) + '\r\n' for k in range(len(t)))


TRIPS = {40: 10, 200: 10}       # cycle -> the wet half's second the limit trips at
ts_, vs_, ws_, t_ = [], [], [], 0.0
for c_ in range(300):
    for wet_, n_ in ((1, TRIPS.get(c_, 30)), (0, 45)):
        for k_ in range(1, n_ + 1):
            t_ += 1.0
            ts_.append(t_)
            vs_.append(0.65 if wet_ and TRIPS.get(c_) == k_ else 0.95)
            ws_.append(float(wet_))
RDIR = TMP / 'rh_station'
write_fcd(RDIR / 'c4_OCV-RHC_90C-100RH(90C)-0RH_0o1H2-0o1Air_0kPa.fcd', 100.0 + np.asarray(ts_),
          {'E_Stack (V)': np.asarray(vs_), 'Wet_Anode': np.asarray(ws_), 'Wet_Cathode': np.asarray(ws_)},
          'Open Circuit; 30 Seconds; Fixed 1 Sec/Pt; Term:E<0.7 NextStep')
resr0 = quiet(FD.run, str(RDIR), str(TMP / 'out_r0'), {'image_format': 'png'})
tabr = resr0['fc_deaths']
check('an RH-cycling hold whose station limit trips twice: a death and a later death, each the '
      'station\'s, with RH cycles; the RH cycles plot among its own',
      ([(e['trigger'], e['cycles']) for e in tabr['events']], resr0['verdict'],
       D.RH_PLOT in {sc_.get('plot_type') for sc_ in REC.load_sidecars(TMP / 'out_r0').values()}),
      ([('station', 40), ('station', 200)], 'dead at 0.84 h, OCV (station limit)', True))
drs = [D.new_dismissal(e, 'death', 'station', 'n') for e in tabr['events']]
for n_, S in enumerate(([drs[0]], [drs[1]], drs)):
    resS = quiet(FD.run, str(RDIR), str(TMP / f'out_r{n_ + 1}'),
                 {'image_format': 'png', 'ocv_dismissals': S})
    rec_ = {'fc_deaths': copy.deepcopy(tabr), 'ocv_dismissals': S,
            'summary': copy.deepcopy(resr0['summary'])}
    got = REC.load_sidecars(TMP / 'out_r0')
    D.apply_to_record(rec_, sidecars=got)
    want = REC.load_sidecars(TMP / f'out_r{n_ + 1}')
    check(f'... {["the death", "the later death", "both"][n_]} dismissed ({resS["verdict"]}): the '
          f'row and every plot, the RH cycles plot\'s too, as the analysis has them',
          (list(rec_['summary'][0].items()) == list(resS['summary'][0].items()),
           sorted(n for n in want if not same_json(got.get(n), want[n]))), (True, []))

# A run with more deaths than the table keeps: the rest counted as later
# deaths, and a dismissal's rewrite still the analysis's own.
MDIR = TMP / 'many_deaths'


def many_ocv(H):
    """Under 0.7 V for 35 min of every 55 from 1 h on: a death, then a later death each time."""
    return np.where((H >= 3600.0) & (((H - 3600.0) % 3300.0) < 2100.0), 0.30,
                    0.95 - (H / 3600.0) * 400e-6)


xover(MDIR, 'b20')
for j_, step_ in enumerate(['c2', 'c4', 'c6', 'c8', 'c10']):
    if j_:
        gas_step(MDIR, f'c{2 * j_ + 1}')
    write_ocv(MDIR / HOLD.format(step_), 14400, 10, j_ * 14400.0, many_ocv)
real_cap = D.TABLE_EVENTS_MAX
D.TABLE_EVENTS_MAX = 12
try:
    resm0 = quiet(FD.run, str(MDIR), str(TMP / 'out_m0'), {'image_format': 'png'})
    tabm = resm0['fc_deaths']
    dm = D.new_dismissal(tabm['events'][0], 'death', 'gas', 'n')
    resm1 = quiet(FD.run, str(MDIR), str(TMP / 'out_m1'), {'image_format': 'png', 'ocv_dismissals': [dm]})
finally:
    D.TABLE_EVENTS_MAX = real_cap
valm = REC.extract_values(REC.load_sidecars(TMP / 'out_m0')['fcdur_ocv_vs_time'])
check('a run of more deaths than the table keeps (12 here): the first kept, all counted',
      (len(tabm['events']), tabm['events_found'] > 13, valm.get('Later OCV deaths'),
       D.resolve(tabm, [])['later_n']),
      (12, True, float(tabm['events_found'] - 1), tabm['events_found'] - 1))
rec_ = {'fc_deaths': copy.deepcopy(tabm), 'ocv_dismissals': [dm], 'summary': copy.deepcopy(resm0['summary'])}
got = REC.load_sidecars(TMP / 'out_m0')
D.apply_to_record(rec_, sidecars=got)
want = REC.load_sidecars(TMP / 'out_m1')
check('... its death dismissed: the rewrite is the analysis\'s, the later deaths past the table counted',
      (list(rec_['summary'][0].items()) == list(resm1['summary'][0].items()),
       sorted(n for n in want if not same_json(got.get(n), want[n])),
       REC.extract_values(got['fcdur_ocv_vs_time']).get('Later OCV deaths')),
      (True, [], float(tabm['events_found'] - 2)))
# A later death past the table dismissed (made when an earlier one kept it):
# the table keeps it too, so the store applies the dismissal as the analysis.
tab_all = quiet(FD.run, str(MDIR), str(TMP / 'out_m_all'), {'image_format': 'png'})['fc_deaths']
d15 = D.new_dismissal(tab_all['events'][14], 'later death', 'station', 'n')
D.TABLE_EVENTS_MAX = 12
try:
    resm2 = quiet(FD.run, str(MDIR), str(TMP / 'out_m2'), {'image_format': 'png', 'ocv_dismissals': [dm, d15]})
finally:
    D.TABLE_EVENTS_MAX = real_cap
tabm2 = resm2['fc_deaths']
rm2 = D.resolve(tabm2, [dm, d15])
valm2 = REC.extract_values(REC.load_sidecars(TMP / 'out_m2')['fcdur_ocv_vs_time'])
check('a later death past the table dismissed: kept with the first, the analysis\'s row and plot alike',
      (len(tab_all['events']), [round(e['start_h'], 4) for e in tabm2['events']][12:], D.kept_first(tabm2),
       tabm2['events_found'], rm2['dismissed'], len(rm2['unmatched']), rm2['later_n'],
       resm2['summary'][0].get('OCV deaths dismissed'), valm2.get('Later OCV deaths'),
       valm2.get('Dismissed OCV deaths')),
      (tabm['events_found'], [round(tab_all['events'][14]['start_h'], 4)], 12, tabm['events_found'],
       [0, 12], 0, tabm['events_found'] - 3, 2, float(tabm['events_found'] - 3), 2.0))
rec_ = {'fc_deaths': copy.deepcopy(tabm2), 'ocv_dismissals': [dm, d15],
        'summary': copy.deepcopy(resm2['summary'])}
got = REC.load_sidecars(TMP / 'out_m2')
D.apply_to_record(rec_, sidecars=got)
want = REC.load_sidecars(TMP / 'out_m2')
check('... stored so, the rewrite changes nothing: the store\'s rule is the analysis\'s',
      (list(rec_['summary'][0].items()) == list(resm2['summary'][0].items()),
       sorted(n for n in want if not same_json(got.get(n), want[n]))), (True, []))
rec_['ocv_dismissals'] = [dm]
D.apply_to_record(rec_, sidecars=got)
want = REC.load_sidecars(TMP / 'out_m1')
check('... restored there: a later death again, counted with the rest; the row and every plot as the '
      'analysis judged without it',
      (rec_['summary'][0].get('OCV deaths dismissed', 'absent'),
       REC.extract_values(got['fcdur_ocv_vs_time']).get('Later OCV deaths'),
       list(rec_['summary'][0].items()) == list(resm1['summary'][0].items()),
       sorted(n for n in want if not same_json(got.get(n), want[n]))),
      (1, float(tabm['events_found'] - 2), True, []))

# ── 4. the store: dismiss, restore, settle, reruns ───────────────────
print('4. the store')


class Store:
    """JSONBin in memory: bins and the index persist across pushes."""

    def __init__(self):
        self.index, self.bins = {'schema': 2, 'runs': []}, {}

    def __call__(self, url, method='GET', body=None, extra_headers=None):
        cp = (lambda x: json.loads(json.dumps(x)))
        if method == 'POST':
            bid = f'BIN{len(self.bins) + 1}'
            self.bins[bid] = cp(body)
            return {'record': body, 'metadata': {'id': bid}}
        if method == 'GET':
            bid = next((b for b in self.bins if f'/{b}/' in url), None)
            return {'record': cp(self.bins[bid] if bid else self.index)}
        if method == 'PUT':
            bid = next((b for b in self.bins if url.endswith(f'/{b}')), None)
            if bid:
                self.bins[bid] = cp(body)
            else:
                self.index = cp(body)
            return {'record': body}
        raise AssertionError(method)


store = Store()
jsonbin._request = store


def submit(job_id, src, params):
    """One job through the portal's own runner and completion callback."""
    d = portal.JOBS_DIR / job_id
    (d / 'output').mkdir(parents=True)
    shutil.copytree(src, d / 'input')
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
        portal.push_executor.submit(lambda: None).result()
    return portal.jobs[job_id]


def entry_of(name):
    return next(e for e in store.index['runs'] if e['sample_name'] == name)


def row_of(rec):
    row = next(r for r in rec['summary'] if r.get('Analysis') == 'fc_durability')
    return {k: row.get(k, 'absent') for k in ('OCV death', 'Verdict', 'Hours to death',
                                              'OCV deaths dismissed')}


def r4(d):
    """Hours to four places, a rate to one."""
    return {k: (round(v, 1 if k == 'rate' else 4) if isinstance(v, float) else v)
            for k, v in (d or {}).items()}


def frozen():
    return json.dumps([store.index, store.bins], sort_keys=True)


SAMPLE, OTHER = '261021_SYN-FD2', '261104_SYN-FD1'
P = {'sample_name': SAMPLE, 'image_format': 'png', 'stand': 'FCTS 1'}
j1 = submit('j1', FDIR, P)
submit('j0', FDIR, dict(P, sample_name=OTHER))
entry = entry_of(SAMPLE)
key = entry['bin_id']
rec = store.bins[key]
check('pushed: the record keeps the table, the index the deaths',
      (j1['metrics_push'].get('pushed'), [round(e['start_h'], 4) for e in rec['fc_deaths']['events']],
       r4(entry.get('ocv_deaths'))),
      (True, [5.4972, 18.0],
       {'death_h': 5.4972, 'found': 2, 'dismissed': 0, 'dismissals': 0, 'review': 0,
        'trigger': 'median'}))
check('... the history: the analysis, when it was stored',
      [(h['what'], h['text'], h['at'] >= entry['timestamp'], h['job_id']) for h in rec['ocv_death_history']],
      [('analysed', 'Analysed, 20.00 h tested: OCV death at 5.50 h in hold 2 (c4), 1 later death',
        True, 'j1')])
L = admin.list_deaths()
check('listed for the admin, newest first, with the reasons and the match window',
      ([(t['sample_name'], t['found'], t['review']) for t in L['tests']], L['older'], L['reasons'],
       L['match_min']),
      ([(OTHER, 2, 0), (SAMPLE, 2, 0)], [], [list(r) for r in D.REASONS], 10.0))
det = admin.death_detail(key)
check('a test\'s deaths for the admin, and its OCV plot',
      ([(e['i'], e['status'], round(e['start_h'], 4), e['where']) for e in det['view']['events']],
       det['ocv_plot'], det['sample_name']),
      ([(0, 'death', 5.4972, 'hold 2 (c4)'), (1, 'later death', 18.0, 'hold 5 (c10)')],
       f'{SAMPLE}-FCDurability-fcdur_ocv_vs_time', SAMPLE))
raises('... none for a key not stored', admin.death_detail, 'NOPE', want='no stored sample')
ev0 = det['view']['events'][0]
before = frozen()
pv = admin.dismiss_death(key, 0, ev0['start_h'], '', '')
check('a preview needs no reason or note, and writes nothing',
      (pv['applied'], frozen() == before, pv['event']['i'], pv['event']['status']),
      (False, True, 0, 'death'))
check('... what changes',
      (r4(pv['before']), r4(pv['after'])),
      ({'death_h': 5.4972, 'verdict': 'dead at 5.50 h, OCV', 'dead': True, 'hours_to_death': 5.4972,
        'cycles_to_death': None, 'cause': 'OCV', 'rate': 400.0, 'window': [0.0, 4.0], 'later': 1,
        'dismissed': 0},
       {'death_h': 18.0, 'verdict': 'dead at 18.00 h, OCV', 'dead': True, 'hours_to_death': 18.0,
        'cycles_to_death': None, 'cause': 'OCV', 'rate': 400.0, 'window': [12.0, 16.0], 'later': 0,
        'dismissed': 1}))
raises('applying needs a reason', admin.dismiss_death, key, 0, ev0['start_h'], '', 'note',
       apply=True, want="choose why")
raises('... one of the four', admin.dismiss_death, key, 0, ev0['start_h'], 'weather', 'note',
       apply=True, want="choose why")
raises('... a reason given is checked in a preview too', admin.dismiss_death, key, 0,
       ev0['start_h'], 'weather', '', want="choose why")
raises('... and a note', admin.dismiss_death, key, 0, ev0['start_h'], 'gas', ' \t\n ', apply=True,
       want='a note is required')
raises('... of 500 characters at most', admin.dismiss_death, key, 0, ev0['start_h'], 'gas',
       'x' * 501, apply=True, want='over 500')
raises('a death the stored analysis does not hold', admin.dismiss_death, key, 2, 18.0, 'gas', 'n',
       apply=True, want='not in the stored analysis')
raises('... or one since moved', admin.dismiss_death, key, 0, ev0['start_h'] + 0.01, 'gas', 'n',
       apply=True, want='has changed since')
raises('... or none named', admin.dismiss_death, key, 'x', ev0['start_h'], 'gas', 'n',
       want='no death given')
check('nothing written by a refusal', frozen() == before, True)
ts0 = entry['timestamp']
ap = admin.dismiss_death(key, 0, ev0['start_h'], 'gas', 'H2 cylinder\x07 ran empty', apply=True)
rec, entry = store.bins[key], entry_of(SAMPLE)
check('applied, with a backup of the record as it was',
      (ap['applied'], ap['backup']['kind'], [p['bin_id'] for p in ap['backup']['purged']],
       'ocv_dismissals' in ap['backup']['purged'][0]['record'],
       ap['backup']['purged'][0]['record']['summary'][0]['OCV death']),
      (True, 'analysis-portal-backup', [key], False, 5.4972))
dzs = rec['ocv_dismissals'][0]
check('the dismissal kept with the record: where, why, the note cleaned',
      {k: dzs[k] for k in ('id', 'step', 'kind', 'reason', 'note', 'by', 'trigger')},
      {'id': ap['id'], 'step': 'c4', 'kind': 'death', 'reason': 'gas', 'note': 'H2 cylinder  ran empty',
       'by': 'admin', 'trigger': 'median'})
check('... the stored row as the analysis judged with it gives it',
      rec['summary'][0], res1['summary'][0])
stored_sc = {n.replace(f'{SAMPLE}-FCDurability-', ''): sc_
             for n, sc_ in REC.decode_sidecars(rec).items()}
job_sc = REC.load_sidecars(out1)
check('... the stored plots as the analysis judged with it draws them, every one',
      (sorted(stored_sc), sorted(n for n in job_sc if not same_json(stored_sc.get(n), job_sc[n]))),
      (sorted(job_sc), []))
kv = entry['Data'][0]['key_values']
check('... the index entry: its values and deaths, its time and job count as they were',
      (kv.get('Hours to death'), kv.get('OCV decay rate'), r4(entry['ocv_deaths']),
       entry['timestamp'] == ts0, entry.get('n_jobs') == entry_of(OTHER).get('n_jobs')),
      (18.0, 400.0, {'death_h': 18.0, 'found': 2, 'dismissed': 1, 'dismissals': 1, 'review': 0,
                     'trigger': 'median'},
       True, True))
h_ = rec['ocv_death_history'][-1]
check('... its history',
      (h_['what'], h_['text'], h_['death_h'], h_['applied'], h_['review'], h_['id']),
      ('dismissed', 'Admin dismissed the death at 5.50 h in hold 2 (c4): gas supply ran out', 18.0, 1,
       0, ap['id']))
check('... the other test untouched', r4(entry_of(OTHER)['ocv_deaths'])['dismissed'], 0)
raises('a death dismissed already', admin.dismiss_death, key, 0, ev0['start_h'], 'gas', 'again',
       apply=True, want='already dismissed')
ev1 = admin.death_detail(key)['view']['events'][1]
pv = admin.dismiss_death(key, 1, ev1['start_h'], 'station', '')
check('the later death too, previewed: alive',
      (pv['event']['status'], pv['after']['verdict'], pv['after']['dead'], pv['after']['hours_to_death'],
       pv['after']['death_h']), ('death', 'alive at 20.00 h', False, None, None))

# A rerun of the same files: the job reads the dismissal in its worker and is
# judged with it; the push has nothing to apply.
# Made after the first analysis, before the rerun, whatever the clock says.
store.bins[key]['ocv_death_history'][0]['at'] = '2000-01-01T00:00:00Z'
store.bins[key]['ocv_dismissals'][0]['at'] = '2000-01-02T00:00:00Z'
j2 = submit('j2', FDIR, P)
rec = store.bins[key]
check('a rerun: judged with the stored dismissal, read in the worker; nothing left to apply',
      (j2['script_result']['verdict'], j2['metrics_push'].get('dismissals_reapplied'),
       'fc_deaths' in j2['script_result'], j2['message']),
      ('dead at 18.00 h, OCV', None, False, 'Analysis complete'))
check('... the stored table without the job\'s copy of them', 'dismissals' in rec['fc_deaths'], False)
check('... the dismissal kept, the row as dismissed',
      ([d['id'] for d in rec['ocv_dismissals']], row_of(rec)),
      ([ap['id']], {'OCV death': 18.0, 'Verdict': 'dead', 'Hours to death': 18.0,
                    'OCV deaths dismissed': 1}))
h_ = rec['ocv_death_history'][-1]
check('... the history: found again, kept dismissed',
      (h_['what'], h_['text'], h_['applied'], h_['review'], h_['job_id']),
      ('analysed', 'Re-analysed, 20.00 h tested: OCV death at 5.50 h in hold 2 (c4), 1 later death; '
                   'the death at 5.50 h found again in hold 2 (c4), kept dismissed', 1, 0, 'j2'))
check('... and Recall: kept through it', [(x['status'], x['kept_through']) for x in D.view(rec)['dismissals']],
      [('applied', True)])

# A rerun that could not read the dismissals: judged without; the push
# applies the stored ones itself.
real_read = D.stored_dismissals


def down(name):
    raise RuntimeError('the store is down')


D.stored_dismissals = down
try:
    j3 = submit('j3', FDIR, P)
finally:
    D.stored_dismissals = real_read
rec = store.bins[key]
ocv_name = f'{SAMPLE}-FCDurability-fcdur_ocv_vs_time'
check('a rerun that could not read them: judged without them, the push applies them',
      (j3['script_result']['verdict'], j3['metrics_push'].get('dismissals_reapplied'), row_of(rec),
       vlines(REC.decode_sidecars(rec)[ocv_name]['data']['axes'][0], 1.0, 19.5)),
      ('dead at 5.50 h, OCV', True,
       {'OCV death': 18.0, 'Verdict': 'dead', 'Hours to death': 18.0, 'OCV deaths dismissed': 1},
       [(18.0, '-', rgba3(D.DEATH_COLOR), 'OCV death at 18.00 h'),
        (5.4972, ':', rgba3(D.DISMISSED_COLOR), D.DISMISSED_LABEL)]))
check('... its index entry from the record as rewritten',
      (entry_of(SAMPLE)['Data'][0]['key_values'].get('Hours to death'), r4(entry_of(SAMPLE)['ocv_deaths'])),
      (18.0, {'death_h': 18.0, 'found': 2, 'dismissed': 1, 'dismissals': 1, 'review': 0,
              'trigger': 'median'}))
check('... its job says its downloads were not judged with them',
      'stored with the sample\'s dismissed OCV deaths' in j3['message'], True)

# A rerun with a 2 h settle: the death in hold 2 now starts at 6.00 h, 30 min
# from the dismissed one -- not taken for it without an admin.
j4 = submit('j4', FDIR, dict(P, settle_min='120'))
rec, entry = store.bins[key], entry_of(SAMPLE)
v_ = D.view(rec)
check('a rerun that moves the death 30 min: the dismissal not applied, to review',
      (j4['script_result']['verdict'], r4(entry['ocv_deaths']),
       [(x['status'], round(x['nearest']['start_h'], 2)) for x in v_['dismissals']]),
      ('dead at 6.00 h, OCV', {'death_h': 6.0, 'found': 2, 'dismissed': 0, 'dismissals': 1,
                                'review': 1, 'trigger': 'median'}, [('review', 6.0)]))
h_ = rec['ocv_death_history'][-1]
check('... said in the history',
      (h_['text'], h_['review'], h_['applied']),
      ('Re-analysed, 20.00 h tested: OCV death at 6.00 h in hold 2 (c4), 1 later death; no death at '
       '5.50 h in hold 2 (c4): the dismissal not applied, to review', 1, 0))
check('... listed first for the admin', [t['sample_name'] for t in admin.list_deaths()['tests']],
      [SAMPLE, OTHER])
did = rec['ocv_dismissals'][0]['id']
at0 = rec['ocv_dismissals'][0]['at']
raises('settling: an unknown action', admin.settle_dismissal, key, did, 'drop', want='unknown action')
raises('... a dismissal not on the record', admin.settle_dismissal, key, 'nope', 'keep',
       want='no such dismissal')
before = frozen()
pv = admin.settle_dismissal(key, did, 'keep')
check('... kept on file, previewed: the verdict as it is; nothing written',
      (pv['before']['verdict'], pv['after']['verdict'], pv['applied'], frozen() == before),
      ('dead at 6.00 h, OCV', 'dead at 6.00 h, OCV', False, True))
kp = admin.settle_dismissal(key, did, 'keep', apply=True)
rec, entry = store.bins[key], entry_of(SAMPLE)
check('... kept: it waits no more',
      (kp['applied'], bool(kp.get('backup')), rec['ocv_dismissals'][0].get('kept'),
       entry['ocv_deaths']['review'], D.view(rec)['dismissals'][0]['status'],
       rec['ocv_death_history'][-1]['what'], rec['ocv_death_history'][-1]['text']),
      (True, True, True, 0, 'kept', 'kept', 'Admin kept the dismissal of 5.50 h on file, unapplied'))
raises('a dismissal not applied is settled, not restored', admin.restore_death, key, did,
       want='not applied')
j4b = submit('j4b', FDIR, P)
rec = store.bins[key]
check('kept on file, then an analysis that finds its death: applied again, and kept no more',
      (j4b['script_result']['verdict'], D.view(rec)['dismissals'][0]['status'],
       'kept' in rec['ocv_dismissals'][0], row_of(rec)['Hours to death']),
      ('dead at 18.00 h, OCV', 'applied', False, 18.0))
j4c = submit('j4c', FDIR, dict(P, settle_min='120'))
rec = store.bins[key]
check('... so the next that loses it flags it to review again',
      (D.view(rec)['dismissals'][0]['status'], entry_of(SAMPLE)['ocv_deaths']['review']), ('review', 1))
ev0 = D.view(rec)['events'][0]
raises('... moved to a death since moved', admin.settle_dismissal, key, did, 'move', 0,
       ev0['start_h'] + 1, apply=True, want='has changed since')
raises('... or to none', admin.settle_dismissal, key, did, 'move', None, None, want='no death given')
admin.settle_dismissal(key, did, 'keep', apply=True)
mv = admin.settle_dismissal(key, did, 'move', 0, ev0['start_h'], apply=True)
rec = store.bins[key]
dzs = rec['ocv_dismissals'][0]
check('... kept on file again, then moved to the death the rerun found: kept no more; the reason and '
      'note carried over',
      (round(dzs['start_h'], 4), round(dzs['offset_h'], 4), dzs['kind'], dzs['reason'], dzs['note'],
       dzs['id'] == did, 'kept' in dzs, bool(dzs.get('moved_at')), dzs['at'] == at0),
      (6.0, 2.0, 'death', 'gas', 'H2 cylinder  ran empty', True, False, True, True))
check('... applied: the next death counts',
      (mv['after']['death_h'], r4(entry_of(SAMPLE)['ocv_deaths']), row_of(rec)['Hours to death']),
      (18.0, {'death_h': 18.0, 'found': 2, 'dismissed': 1, 'dismissals': 1, 'review': 0,
              'trigger': 'median'}, 18.0))
check('... not kept through an analysis: moved since', D.view(rec)['dismissals'][0]['kept_through'],
      False)
check('... in the history', rec['ocv_death_history'][-1]['text'],
      'Admin moved the dismissal of 5.50 h to the death at 6.00 h in hold 2 (c4): gas supply ran out')
raises('an applied dismissal is restored, not settled', admin.settle_dismissal, key, did, 'remove',
       apply=True, want='restore it instead')
pv = admin.restore_death(key, did)
check('restoring, previewed', (pv['applied'], pv['before']['death_h'], pv['after']['death_h']),
      (False, 18.0, 6.0))
rs = admin.restore_death(key, did, apply=True)
rec = store.bins[key]
check('restored: no dismissal, the death counts again',
      (rec['ocv_dismissals'], rs['applied'], bool(rs.get('backup')), row_of(rec),
       r4(entry_of(SAMPLE)['ocv_deaths'])['dismissed']),
      ([], True, True, {'OCV death': 6.0, 'Verdict': 'dead', 'Hours to death': 6.0,
                        'OCV deaths dismissed': 'absent'}, 0))
check('... in the history', (rec['ocv_death_history'][-1]['what'], rec['ocv_death_history'][-1]['text']),
      ('restored', 'Admin restored the death at 6.00 h in hold 2 (c4)'))
check('... its plot: the grey line gone, the death red again, the later death dash-dot',
      vlines(REC.decode_sidecars(rec)[ocv_name]['data']['axes'][0], 1.0, 19.5),
      [(6.0, '-', rgba3(D.DEATH_COLOR), 'OCV death at 6.00 h'),
       (18.0, '-.', rgba3(D.DEATH_COLOR), D.LATER_LABEL)])
plain = {n.replace(f'{SAMPLE}-FCDurability-', ''): sc_ for n, sc_ in REC.decode_sidecars(rec).items()}
want = {n.replace(f'{SAMPLE}-FCDurability-', ''): sc_
        for n, sc_ in REC.load_sidecars(portal.JOBS_DIR / 'j4' / 'output').items()}
check('... dismissed, kept, moved and restored: the record as the analysis gave it, row and plots',
      (rec['summary'][0] == j4['script_result']['summary'][0], sorted(plain),
       sorted(n for n in want if not same_json(plain.get(n), want[n]))),
      (True, sorted(want), []))
raises('... not twice', admin.restore_death, key, did, want='no such dismissal')

# Dismissed under the 2 h settle, then rerun with the usual one: 30 min off
# again, to review, and removed.
ev0 = D.view(store.bins[key])['events'][0]
admin.dismiss_death(key, 0, ev0['start_h'], 'operator', 'Stopped by hand to swap a fitting',
                    apply=True)
j5 = submit('j5', FDIR, P)
rec = store.bins[key]
did = rec['ocv_dismissals'][0]['id']
check('back to the usual settle: that dismissal 30 min off, to review',
      (j5['script_result']['verdict'], D.view(rec)['dismissals'][0]['status']),
      ('dead at 5.50 h, OCV', 'review'))
rm = admin.settle_dismissal(key, did, 'remove', apply=True)
rec = store.bins[key]
check('... removed: gone, said in the history; the analysis as it was',
      (rec['ocv_dismissals'], rec['ocv_death_history'][-1]['what'],
       rec['ocv_death_history'][-1]['text'], rm['before']['verdict'], rm['after']['verdict'],
       entry_of(SAMPLE)['ocv_deaths']['review']),
      ([], 'remove', 'Admin removed the dismissal of 6.00 h', 'dead at 5.50 h, OCV',
       'dead at 5.50 h, OCV', 0))
check('the history, in order', [h['what'] for h in rec['ocv_death_history']],
      ['analysed', 'dismissed', 'analysed', 'analysed', 'analysed', 'kept', 'analysed', 'analysed',
       'kept', 'move', 'restored', 'dismissed', 'analysed', 'remove'])
check('... dated in that order', [h['at'] for h in rec['ocv_death_history']],
      sorted(h['at'] for h in rec['ocv_death_history']))

# Two entries for one sample, as a push whose lookup failed leaves them, and
# the admin's merge of them.
real_find, real_read = jsonbin.find_sample_entry, D.stored_dismissals


def lost(index, name):
    raise RuntimeError('lookup down')


def lookup_down(job_id, params, read=True):
    """A push that cannot look its sample up (the worker's read works, or
    not), so makes a new bin."""
    def worker_read(name):
        jsonbin.find_sample_entry = real_find
        try:
            return real_read(name)
        finally:
            jsonbin.find_sample_entry = lost
    jsonbin.find_sample_entry, D.stored_dismissals = lost, (worker_read if read else down)
    try:
        return submit(job_id, FDIR, params)
    finally:
        jsonbin.find_sample_entry, D.stored_dismissals = real_find, real_read


def entries_of(name):
    return [e for e in store.index['runs'] if e['sample_name'] == name]


def dismiss_at(k, i=0, note='H2 ran out'):
    e = admin.death_detail(k)['view']['events'][i]
    return admin.dismiss_death(k, e['i'], e['start_h'], 'gas', note, apply=True)


DUP = '261020_SYN-DUP'
submit('d1', FDIR, dict(P, sample_name=DUP))
kd1 = entry_of(DUP)['bin_id']
dd = dismiss_at(kd1)
jd = lookup_down('d2', dict(P, sample_name=DUP))
kd2 = entries_of(DUP)[-1]['bin_id']
check('a push that cannot look the sample up: a new bin, keeping the dismissals its job was judged '
      'with', (len(entries_of(DUP)), jd['script_result']['verdict'],
               jd['metrics_push'].get('dismissals_reapplied'),
               [d['id'] for d in store.bins[kd2].get('ocv_dismissals') or []],
               row_of(store.bins[kd2]), r4(entries_of(DUP)[-1]['ocv_deaths'])['dismissed']),
      (2, 'dead at 18.00 h, OCV', None, [dd['id']],
       {'OCV death': 18.0, 'Verdict': 'dead', 'Hours to death': 18.0, 'OCV deaths dismissed': 1}, 1))
check('... its job\'s own row as it was judged', jd['script_result']['summary'][0]['OCV death'], 18.0)
mg = admin.merge_duplicate_group([kd1, kd2], DUP, apply=True)
mrec = store.bins[mg['survivor_bin']]
check('... merged: the dismissal once, applied', ([d['id'] for d in mrec['ocv_dismissals']],
                                                  row_of(mrec), len(entries_of(DUP))),
      ([dd['id']], {'OCV death': 18.0, 'Verdict': 'dead', 'Hours to death': 18.0,
                    'OCV deaths dismissed': 1}, 1))
DUP2 = '261019_SYN-DUP'
submit('e1', FDIR, dict(P, sample_name=DUP2))
ke1 = entry_of(DUP2)['bin_id']
lookup_down('e2', dict(P, sample_name=DUP2))
ke2 = entries_of(DUP2)[-1]['bin_id']
de = dismiss_at(ke2)
mg = admin.merge_duplicate_group([ke1, ke2], DUP2, apply=True)
mrec, ment = store.bins[mg['survivor_bin']], entry_of(DUP2)
check('a merge where the newer entry holds the dismissal: kept, applied; the index from it',
      ([d['id'] for d in mrec['ocv_dismissals']], row_of(mrec), r4(ment['ocv_deaths']),
       ment['Data'][0]['key_values'].get('Hours to death'), [d['id'] for d in D.stored_dismissals(DUP2)]),
      ([de['id']], {'OCV death': 18.0, 'Verdict': 'dead', 'Hours to death': 18.0,
                    'OCV deaths dismissed': 1},
       {'death_h': 18.0, 'found': 2, 'dismissed': 1, 'dismissals': 1, 'review': 0, 'trigger': 'median'},
       18.0, [de['id']]))
check('... the histories of both, in time order',
      ([h['what'] for h in mrec['ocv_death_history']],
       [h['at'] for h in mrec['ocv_death_history']] == sorted(h['at'] for h in mrec['ocv_death_history'])),
      (['analysed', 'analysed', 'dismissed'], True))
DUP3 = '261018_SYN-DUP'
submit('f1', FDIR, dict(P, sample_name=DUP3))
kf1 = entry_of(DUP3)['bin_id']
df = dismiss_at(kf1)
lookup_down('f2', dict(P, sample_name=DUP3), read=False)
kf2 = entries_of(DUP3)[-1]['bin_id']
check('a push that could neither read nor find them: a new bin without', (
    store.bins[kf2].get('ocv_dismissals'), row_of(store.bins[kf2])['Hours to death']), (None, 5.4972))
mg = admin.merge_duplicate_group([kf1, kf2], DUP3, apply=True)
mrec, ment = store.bins[mg['survivor_bin']], entry_of(DUP3)
ocv_m = next(sc_ for sc_ in REC.decode_sidecars(mrec).values() if sc_.get('plot_type') == D.OCV_PLOT)
check('... merged with the older that holds it: applied to the newer analysis, plots and index too',
      ([d['id'] for d in mrec['ocv_dismissals']], row_of(mrec), vlines(ocv_m['data']['axes'][0], 1.0, 19.5),
       r4(ment['ocv_deaths'])['dismissed'], ment['Data'][0]['key_values'].get('Hours to death')),
      ([df['id']], {'OCV death': 18.0, 'Verdict': 'dead', 'Hours to death': 18.0,
                    'OCV deaths dismissed': 1},
       [(18.0, '-', rgba3(D.DEATH_COLOR), 'OCV death at 18.00 h'),
        (5.4972, ':', rgba3(D.DISMISSED_COLOR), D.DISMISSED_LABEL)], 1, 18.0))

# What an admin did in either of two duplicates stands through their merge.
def dup_pair(n, name, settle=None):
    """A sample stored and its death dismissed, then a push that could not
    look it up (its worker read the dismissal, with a 2 h settle or not): two
    bins, the dismissal in both, as the second's job was judged."""
    submit(f'q{n}a', FDIR, dict(P, sample_name=name))
    k1 = entry_of(name)['bin_id']
    did_ = dismiss_at(k1)['id']
    lookup_down(f'q{n}b', dict(P, sample_name=name, **({'settle_min': settle} if settle else {})))
    return k1, entries_of(name)[-1]['bin_id'], did_


def merged_state(name, ks):
    mg_ = admin.merge_duplicate_group(ks, name, apply=True)
    m_, e_ = store.bins[mg_['survivor_bin']], entry_of(name)
    return ([(round(d['start_h'], 4), d.get('moved_at') is not None) for d in m_.get('ocv_dismissals') or []],
            row_of(m_)['OCV death'], r4(e_['ocv_deaths']), len(entries_of(name)),
            [h['what'] for h in m_['ocv_death_history']][-1])


DUP4 = '261012_SYN-DUP'
ka, kb, did = dup_pair(4, DUP4)
admin.restore_death(kb, did, apply=True)
check('restored in the newer of two duplicates: gone once merged, the death counting again',
      merged_state(DUP4, [ka, kb]),
      ([], 5.4972, {'death_h': 5.4972, 'found': 2, 'dismissed': 0, 'dismissals': 0, 'review': 0,
                    'trigger': 'median'}, 1, 'restored'))
DUP5 = '261011_SYN-DUP'
ka, kb, did = dup_pair(5, DUP5)
admin.restore_death(ka, did, apply=True)
check('... restored in the older, after the newer was made: gone too',
      merged_state(DUP5, [ka, kb])[:3],
      ([], 5.4972, {'death_h': 5.4972, 'found': 2, 'dismissed': 0, 'dismissals': 0, 'review': 0,
                    'trigger': 'median'}))
DUP6 = '261010_SYN-DUP'
ka, kb, did = dup_pair(6, DUP6, settle='120')
check('the newer judged with a 2 h settle: its copy of the dismissal to review',
      [x['status'] for x in D.view(store.bins[kb])['dismissals']], ['review'])
evb = D.view(store.bins[kb])['events'][0]
admin.settle_dismissal(kb, did, 'move', 0, evb['start_h'], apply=True)
check('... moved there to the death it found: moved once merged, and applied',
      merged_state(DUP6, [ka, kb]),
      ([(6.0, True)], 18.0, {'death_h': 18.0, 'found': 2, 'dismissed': 1, 'dismissals': 1, 'review': 0,
                             'trigger': 'median'}, 1, 'move'))
DUP8 = '261006_SYN-DUP'
submit('q8a', FDIR, dict(P, sample_name=DUP8))
ka = entry_of(DUP8)['bin_id']
did = dismiss_at(ka)['id']
submit('q8b', FDIR, dict(P, sample_name=DUP8, settle_min='120'))
admin.settle_dismissal(ka, did, 'keep', apply=True)
lookup_down('q8c', dict(P, sample_name=DUP8))
kb = entries_of(DUP8)[-1]['bin_id']
check('kept on file, then a push that could not look the sample up: applied again in its new bin, '
      'kept no more, said in its history',
      ('kept' in store.bins[kb]['ocv_dismissals'][0], store.bins[kb]['ocv_death_history'][-1].get('unkept')),
      (False, [did]))
submit('q8d', FDIR, dict(P, sample_name=DUP8, settle_min='120'))
check('... a rerun into it that loses the death: to review there',
      [x['status'] for x in D.view(store.bins[kb])['dismissals']], ['review'])
check('... merged with the older, where it was kept on file before: to review still',
      merged_state(DUP8, [ka, kb])[2]['review'], 1)
DUP7 = '261009_SYN-DUP'
ka, kb, did = dup_pair(7, DUP7, settle='120')
admin.settle_dismissal(kb, did, 'remove', apply=True)
check('... removed there instead: gone once merged, nothing to review',
      merged_state(DUP7, [ka, kb])[:3],
      ([], 6.0, {'death_h': 6.0, 'found': 2, 'dismissed': 0, 'dismissals': 0, 'review': 0,
                 'trigger': 'median'}))


# The command-line merges judge as the admin's: find_duplicates (one sample
# under two spellings) and migrate_samples (one name, two entries), each run
# as on the server over a store of the two alone, the older's bin named first.
def cli(script, pair, *args):
    st = Store()
    for b_, k_ in zip(('CLIA', 'CLIB'), pair):
        st.bins[b_] = copy.deepcopy(store.bins[k_])
        st.index['runs'].append(dict(copy.deepcopy(next(e for e in store.index['runs']
                                                        if e['bin_id'] == k_)), bin_id=b_))
    saved = sys.argv, jsonbin._request, os.getcwd()
    sys.argv, jsonbin._request = [script, *args], st
    os.chdir(TMP)
    try:
        with contextlib.redirect_stdout(io.StringIO()) as out_:
            try:
                runpy.run_path(str(Path(__file__).resolve().parent / script), run_name='__main__')
                code = 0
            except SystemExit as e:
                code = e.code
    finally:
        sys.argv, jsonbin._request = saved[0], saved[1]
        os.chdir(saved[2])
    return st, code, out_.getvalue()


def cli_state(st):
    (e_,) = st.index['runs']
    m_ = st.bins[e_['bin_id']]
    ocv_ = next(sc_ for sc_ in REC.decode_sidecars(m_).values() if sc_.get('plot_type') == D.OCV_PLOT)
    fc_ = [u for u in e_['Data'] if u.get('Analysis') == 'fc_durability']
    return (e_['bin_id'], len(m_.get('ocv_dismissals') or []), row_of(m_), vlines(ocv_['data']['axes'][0], 1.0, 19.5),
            r4(e_['ocv_deaths'])['dismissed'], [u['key_values'].get('Hours to death') for u in fc_])


DEAD18 = ({'OCV death': 18.0, 'Verdict': 'dead', 'Hours to death': 18.0, 'OCV deaths dismissed': 1},
          [(18.0, '-', rgba3(D.DEATH_COLOR), 'OCV death at 18.00 h'),
           (5.4972, ':', rgba3(D.DISMISSED_COLOR), D.DISMISSED_LABEL)], 1, [18.0])
CA, CB = '261008_SYN-CLI1', '261008_SYN_CLI1'
submit('w1', FDIR, dict(P, sample_name=CA))
kca = entry_of(CA)['bin_id']
dismiss_at(kca)
submit('w2', FDIR, dict(P, sample_name=CB))
st, code, out = cli('find_duplicates.py', [kca, entry_of(CB)['bin_id']], '--apply', '--group', '1',
                    '--keep-name', CA)
check('find_duplicates: one sample under two spellings, the older\'s death dismissed: merged into the '
      'newer, the dismissal applied to its analysis, plots and index too',
      (code, '----- END INDEX BACKUP -----' in out) + cli_state(st), (0, True, 'CLIB', 1) + DEAD18)
CM = '261007_SYN-MIG'
submit('w3', FDIR, dict(P, sample_name=CM))
kcm = entry_of(CM)['bin_id']
dismiss_at(kcm)
lookup_down('w4', dict(P, sample_name=CM), read=False)
st, code, out = cli('migrate_samples.py', [kcm, entries_of(CM)[-1]['bin_id']], '--apply')
check('migrate_samples: one name\'s two entries, the older\'s death dismissed: as find_duplicates',
      (code, 'index rewritten' in out) + cli_state(st), (0, True, 'CLIB', 1) + DEAD18)


# A record whose timeline data did not fit: its values brought to the dismissals.
BIG = '261017_SYN-BIG'
submit('g1', FDIR, dict(P, sample_name=BIG))
kg = entry_of(BIG)['bin_id']
rg = store.bins[kg]
REC.attach_sidecars(rg, {n: sc_ for n, sc_ in REC.decode_sidecars(rg).items()
                         if sc_.get('plot_type') != D.TIMELINE_PLOT})
dismiss_at(kg)
dismiss_at(kg, 1)
rg, eg = store.bins[kg], entry_of(BIG)
tl_vals = next(e['values'] for n, e in rg['metrics']['fc_durability'].items() if n.endswith(D.TIMELINE_PLOT))
check('a record whose timeline data was not stored: both dismissed, its values and the index alive',
      (row_of(rg)['Verdict'], {k: tl_vals.get(k, 'absent') for k in ('Verdict', 'Hours to death')},
       eg['Data'][0]['key_values'].get('Hours to death', 'absent')),
      ('alive', {'Verdict': 'alive', 'Hours to death': 'absent'}, 'absent'))
submit('g2', FDIR, dict(P, sample_name=BIG))
check('... as a rerun judged with them has it',
      entry_of(BIG)['Data'][0]['key_values'].get('Hours to death', 'absent'), 'absent')

# Dead on crossover only: no OCV death to dismiss, nor analysis to redo.
XONLY = TMP / 'xover_only'
xo(XONLY, 'a1', 3.0)
write_ocv(XONLY / HOLD.format('c1'), 7200, 30, 0.0, lambda H: 0.95 - 5e-3 * H / 3600)
xo(XONLY, 'c2', 5.0)
write_ocv(XONLY / HOLD.format('c3'), 7200, 30, 7200.0, lambda H: 0.95 - 5e-3 * H / 3600)
xo(XONLY, 'c4', 16.0)
XO = '261016_SYN-XO'
jx = submit('x1', XONLY, dict(P, sample_name=XO))
L = admin.list_deaths()
check('a test dead on crossover only: neither listed nor said to need analysing again',
      (jx['script_result']['verdict'], XO in [t['sample_name'] for t in L['tests']],
       XO in [t['sample_name'] for t in L['older']], r4(entry_of(XO)['ocv_deaths'])),
      ('dead at 4.00 h, crossover, between 2.00 and 4.00 h', False, False,
       {'death_h': None, 'found': 0, 'dismissed': 0, 'dismissals': 0, 'review': 0, 'trigger': ''}))

# An upload's own parameters do not dismiss.
OWN = '261015_SYN-OWN'
jo = submit('o1', FDIR, dict(P, sample_name=OWN, ocv_dismissals=[
    D.new_dismissal(tab['events'][0], 'death', 'gas', 'from the upload')]))
check('dismissals in an upload\'s parameters are not taken',
      (jo['script_result']['verdict'], store.bins[entry_of(OWN)['bin_id']].get('ocv_dismissals')),
      ('dead at 5.50 h, OCV', None))

# A test with more deaths than its table keeps: dismissals that leave the
# death, or the later deaths drawn, past it are refused.
MANY = '261013_SYN-MANY'
D.TABLE_EVENTS_MAX = 12
try:
    submit('m1', MDIR, dict(P, sample_name=MANY))
finally:
    D.TABLE_EVENTS_MAX = real_cap
km = entry_of(MANY)['bin_id']
dismiss_at(km)
raises('more deaths than the table keeps: a second dismissal would draw later deaths past it',
       dismiss_at, km, 1, want='past the first 12')
check('... the first one dismissed stands', (D.view(store.bins[km])['death'], r4(entry_of(MANY)['ocv_deaths'])
                                             ['found']), (1, tabm['events_found']))
# One a rerun found no death for, moved to a death: refused the same.
far = dict(D.new_dismissal(tab_all['events'][5], 'later death', 'gas', 'n'), step='c99')
store.bins[km]['ocv_dismissals'].append(far)
ev1 = D.view(store.bins[km])['events'][1]
before = frozen()
raises('... nor one to review moved to a death, if that draws later deaths past it',
       admin.settle_dismissal, km, far['id'], 'move', 1, ev1['start_h'], apply=True, want='past the first 12')
check('... nothing written', frozen() == before, True)
admin.settle_dismissal(km, far['id'], 'remove', apply=True)
# A later death past the table dismissed when an earlier analysis kept it: a
# rerun judged with it keeps it, applied by the store as by the analysis.
store.bins[km]['ocv_dismissals'].append(d15)
D.TABLE_EVENTS_MAX = 12
try:
    jm2 = submit('m2', MDIR, dict(P, sample_name=MANY))
finally:
    D.TABLE_EVENTS_MAX = real_cap
vm = D.view(store.bins[km])
check('... rerun: the dismissal past the table applied, by the analysis and the store alike',
      (jm2['metrics_push'].get('dismissals_reapplied'), [x['status'] for x in vm['dismissals']],
       len(vm['events']), vm['kept_first'], [e['status'] for e in vm['events']][12:],
       r4(entry_of(MANY)['ocv_deaths'])['dismissed'], row_of(store.bins[km])['OCV deaths dismissed']),
      (None, ['applied', 'applied'], 13, 12, ['dismissed'], 2, 2))
raises('... a refusal there names the first it keeps, not those past them too', dismiss_at, km, 1,
       want='past the first 12 deaths')
check('... each numbered among all found', [e['n'] for e in vm['events']][10:], [11, 12, 15])
admin.restore_death(km, d15['id'], apply=True)
check('... that one restored: listed still, a later death again',
      ([e['status'] for e in D.view(store.bins[km])['events']][11:], r4(entry_of(MANY)['ocv_deaths'])['dismissed']),
      (['later death', 'later death'], 1))
raises('... a dismissal that would draw it among the first ten later deaths: refused', dismiss_at, km, 1,
       want='past the first 12 deaths')

# A test analysed before deaths could be dismissed: no table in its record.
OLDER = '260930_SYN-FD3'
submit('j6', FDIR, dict(P, sample_name=OLDER))
k3 = entry_of(OLDER)['bin_id']
del store.bins[k3]['fc_deaths']
entry_of(OLDER).pop('ocv_deaths')
L = admin.list_deaths()
check('one analysed before: listed apart, to analyse again',
      ([(t['key'], round(t['hours_to_death'], 4)) for t in L['older']],
       k3 in [t['key'] for t in L['tests']]), ([(k3, 5.4972)], False))
raises('... its deaths cannot be dismissed until then', admin.death_detail, k3,
       want='analysed before deaths could be dismissed')
raises('... nor any change made', admin.dismiss_death, k3, 0, 5.4972, 'gas', 'n',
       want='analysed before deaths could be dismissed')
check('a rerun\'s read of the dismissals: none for a sample not stored, or none dismissed',
      (D.stored_dismissals('nope'), D.stored_dismissals(SAMPLE)), ([], []))
real_fetch = jsonbin.fetch_detail_bin


def no_read(bin_id):
    raise AssertionError('read')


jsonbin.fetch_detail_bin = no_read
try:
    check('... none, without reading its record, for one whose index has no deaths, or none found '
          'nor dismissed', (D.stored_dismissals(OLDER), D.stored_dismissals(XO)), ([], []))
finally:
    jsonbin.fetch_detail_bin = real_fetch
key_o = entry_of(OTHER)['bin_id']
ev0 = admin.death_detail(key_o)['view']['events'][0]
admin.dismiss_death(key_o, 0, ev0['start_h'], 'gas', 'n', apply=True)
check('... the stored ones for one with a dismissal',
      [d['id'] for d in D.stored_dismissals(OTHER)], [store.bins[key_o]['ocv_dismissals'][0]['id']])
summ_o = entry_of(OTHER)['ocv_deaths']
entry_of(OTHER)['ocv_deaths'] = dict(summ_o, dismissed=0, dismissals=0)
try:
    check('... read from the record for one with a death found, whatever its index entry says of '
          'dismissals (an admin change whose index write failed)',
          [d['id'] for d in D.stored_dismissals(OTHER)], [store.bins[key_o]['ocv_dismissals'][0]['id']])
finally:
    entry_of(OTHER)['ocv_deaths'] = summ_o
saved_key = os.environ.pop('JSONBIN_API_KEY')
try:
    check('... none with the store not set up', D.stored_dismissals(OTHER), [])
finally:
    os.environ['JSONBIN_API_KEY'] = saved_key

# A dismissal named by its folder and step: kept through an earlier folder
# added, and the folders' order turned round; the same step in another folder
# never taken for it.
FOLD = TMP / 'folders'


def gas_out(t):
    return np.where((t >= 5400) & (t < 9000), 0.30, 0.95)


for fo_ in ('40H_1', '40H_2'):
    write_ocv(FOLD / fo_ / HOLD.format('c2'), 14400, 10, 0.0, lambda t: 0.95 + 0 * t)
    gas_step(FOLD / fo_, 'c3')
    write_ocv(FOLD / fo_ / HOLD.format('c4'), 14400, 10, 0.0, gas_out)
FOLD2 = TMP / 'folders_more'
shutil.copytree(FOLD, FOLD2)
write_ocv(FOLD2 / '20H_0' / HOLD.format('c2'), 14400, 10, 0.0, lambda t: 0.95 + 0 * t)
FO = '261014_SYN-FOLD'
submit('h1', FOLD, dict(P, sample_name=FO))
kh = entry_of(FO)['bin_id']
vh = D.view(store.bins[kh])
check('two folders: a death in each, named by folder and step',
      [(e['folder'], e['step'], round(e['offset_h'], 4), e['status']) for e in vh['events']],
      [('40H_1', 'c4', 1.4972, 'death'), ('40H_2', 'c4', 1.4972, 'later death')])
dismiss_at(kh)


def fold_state():
    v = D.view(store.bins[kh])
    return ([(e['folder'], round(e['start_h'], 2), e['status']) for e in v['events']],
            [(x['status'], x['event']['folder']) for x in v['dismissals'] if x['status'] == 'applied'])


submit('h2', FOLD2, dict(P, sample_name=FO))
check('... an earlier folder added: the clock moves, the dismissal stays with its folder\'s death',
      fold_state(), ([('40H_1', 9.5, 'dismissed'), ('40H_2', 17.5, 'death')], [('applied', '40H_1')]))
submit('h3', FOLD, dict(P, sample_name=FO, folder_order=['40H_2', '40H_1']))
check('... the folders turned round: still that death, never the other folder\'s at the same place',
      fold_state(), ([('40H_2', 5.5, 'death'), ('40H_1', 13.5, 'dismissed')], [('applied', '40H_1')]))

# ── 5. the server ────────────────────────────────────────────────────
print('5. the server')


def http(fn, payload):
    try:
        return fn(payload), None
    except HTTPException as e:
        return None, (e.status_code, e.detail)


check('the routes ask for the code',
      [http(fn, p)[1][0] for fn in (portal.admin_deaths, portal.admin_death_detail,
                                    portal.admin_death_dismiss, portal.admin_death_restore,
                                    portal.admin_death_settle)
       for p in ({}, {'code': 'nope', 'key': key})], [403] * 10)
got_, err_ = http(portal.admin_deaths, {'code': 'code'})
check('... and with it, answer', (err_, [t['key'] for t in got_['tests']]),
      (None, [t['key'] for t in admin.list_deaths()['tests']]))
check('a refusal is a 400 saying why',
      http(portal.admin_death_dismiss, {'code': 'code', 'key': key, 'event': 0, 'start_h': 5.4972,
                                        'reason': 'gas', 'note': '', 'apply': True})[1],
      (400, 'a note is required: say what happened, for whoever reads it next'))
viewstore.clear_cache()
dv = asyncio.run(portal.view_run_detail(key))['ocv_death_view']
check('Recall is shown the deaths', [e['status'] for e in dv['events']], ['death', 'later death'])
ev0 = dv['events'][0]
got_, err_ = http(portal.admin_death_dismiss, {'code': 'code', 'key': key, 'event': 0,
                                               'start_h': ev0['start_h'], 'reason': 'gas',
                                               'note': 'through the route', 'apply': True})
dv = asyncio.run(portal.view_run_detail(key))['ocv_death_view']
check('... and after a change through the route, read afresh',
      (err_, got_['applied'], [e['status'] for e in dv['events']], dv['dismissals'][0]['note']),
      (None, True, ['dismissed', 'death'], 'through the route'))
did = dv['dismissals'][0]['id']
got_, err_ = http(portal.admin_death_restore, {'code': 'code', 'key': key, 'id': did})
check('... the restore route previews', (err_, got_['applied'], got_['after']['death_h'] is not None
                                         and round(got_['after']['death_h'], 4)), (None, False, 5.4972))
check('... the settle route passes its refusal on',
      http(portal.admin_death_settle, {'code': 'code', 'key': key, 'id': did, 'action': 'keep'})[1],
      (400, 'that dismissal is applied to a death: restore it instead'))
check('... the detail route', http(portal.admin_death_detail, {'code': 'code', 'key': key})[0]['key'],
      key)
viewstore.clear_cache()
true('a test without the table: no deaths for Recall',
     'ocv_death_view' not in asyncio.run(portal.view_run_detail(k3)))
dv = asyncio.run(portal.view_run_detail(key))
check('Recall is sent the deaths as their view, not the table or the dismissals as stored',
      ('ocv_death_view' in dv, [k for k in ('fc_deaths', 'ocv_dismissals', 'ocv_death_history') if k in dv]),
      (True, []))

# ── 6. Recall's redraw ───────────────────────────────────────────────
print('6. Recall\'s redraw')
real_render = plot_compare.render_overlay_comparison
seen = []


def spy(items, **kw):
    seen.append(bool(items[0].get('ref_colors')))
    return real_render(items, **kw)


plot_compare.render_overlay_comparison = spy
try:
    asyncio.run(portal.view_render({'key': key, 'plot': ocv_name}))
finally:
    plot_compare.render_overlay_comparison = real_render
check('the server\'s redraw of an FC Durability plot keeps its lines\' colours', seen, [True])
sc_ = REC.decode_sidecars(store.bins[key])[ocv_name]


def drawn(fig):
    """The vertical lines on the upper axis (x, colour) and the legend keys."""
    lines = sorted({(round(float(l.get_xdata()[0]), 4), to_hex(l.get_color()))
                    for l in fig.axes[0].get_lines()
                    if len(l.get_xdata()) == 2 and l.get_xdata()[0] == l.get_xdata()[1]
                    and 1 < l.get_xdata()[0] < 19.5})
    return lines, [t.get_text() for t in fig.legends[0].get_texts()]


fig_ = quiet(real_render, [{'label': ocv_name, 'sidecar': sc_, 'ref_colors': True}],
             save_path=str(TMP / 'redraw.png'), title=ocv_name)
check('... the dismissed death grey and dotted, the death red, each named in the legend',
      drawn(fig_),
      ([(5.4972, D.DISMISSED_COLOR), (18.0, D.DEATH_COLOR)],
       [ocv_name, 'high–low of each bin', 'characterization', 'OCV death at 18.00 h',
        'OCV death dismissed by an admin']))
fig_ = quiet(real_render, [{'label': ocv_name, 'sidecar': sc_}], save_path=str(TMP / 'r2.png'),
             title=ocv_name)
check('... any other plot as before: its lines grey, unnamed',
      drawn(fig_), ([(5.4972, '#808080'), (18.0, '#808080')], [ocv_name, 'high–low of each bin']))
plt.close('all')

shutil.rmtree(TMP, ignore_errors=True)
print(f'\n{passed} passed, {failed} failed')
raise SystemExit(1 if failed else 0)
