"""
Dismissing a false OCV death: the rules the analysis and the store share.

FC Durability judges every stretch of the hold OCV (`judge_runs` in
fc_durability_analysis): the death, later deaths, dips, and parts of an
episode. An admin may dismiss a death the cell did not cause -- a gas supply
that ran out, a station fault, a test stopped by hand -- and the time of
death moves to the next death the rule found, or the sample is alive. The
data, the plots and the death itself stay: the dismissed death is drawn grey
and dotted, and can be restored.

The analysis stores what that takes in the record, as `fc_deaths` (the
table), because the raw files are gone a day after a job (JOB_TTL_HOURS):

- every death and later death, one per episode (`events`): where it starts
  -- its folder, its step, and the time into that step's file, which more
  data, or an earlier folder added, leaves where it was -- how long and how
  low, when the OCV was back, the RH cycles by then and the decay rate ending
  by it;
- the crossover death with the rate ending by it, the rate uncut (a sample
  left alive), the hours and RH cycles tested.

Dismissals belong to the sample, not to an analysis: they are kept in the
record (`ocv_dismissals`) through every rerun, and with them a history
(`ocv_death_history`). A rerun passes them to the analysis, which applies
each to the death that starts in the same step within MATCH_MIN of it. One
it cannot match is not applied -- never moved to another death -- and waits
for an admin (`review`) unless one has kept it on file (`kept`).

`resolve(table, dismissals)` is the one rule for which death counts: the
analysis draws and reports from it, and the store rewrites a record from it
(`apply_to_record`) when an admin dismisses or restores a death, so the two
cannot disagree. The functions here take and return plain dicts, as stored.
"""

from __future__ import annotations

import copy
import json
import math
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

# A rerun applies a dismissal to the death starting in the same step file
# within this many minutes of where the dismissed one started. A death the
# same data gives again starts at the same sample; one that settings moved
# further is not taken for it without an admin.
MATCH_MIN = 10.0
# Hold hours within this are the same moment (as CLOCK_TOL_H in the analysis).
CLOCK_TOL_H = 1e-9
# Later deaths named in the notes and drawn on the plot; the rest are counted.
LATER_LISTED = 10
# Deaths the stored table keeps, the first in clock order, and past them those
# a dismissal takes, so the store applies it as the analysis did; the rest are
# counted (`events_found`). A station's limit tripping all through an RH loop
# can give thousands, and the record has 900 KB for everything.
TABLE_EVENTS_MAX = 100

REASONS: Tuple[Tuple[str, str], ...] = (
    ('gas', 'gas supply ran out'),
    ('station', 'station or controller fault'),
    ('operator', 'operator stop or test setup'),
    ('other', 'other'),
)
REASON_TEXT = dict(REASONS)
NOTE_MAX = 500
HISTORY_MAX = 60

# How the plots draw them, here so that the analysis and a store rewrite
# draw the same.
DEATH_COLOR = '#d62728'
DISMISSED_COLOR = '#6a7686'
LATER_LABEL = 'OCV death rule met again (flagged)'
DISMISSED_LABEL = 'OCV death dismissed by an admin'

# The summary row in the order the analysis writes it; a rewrite keeps it.
SUMMARY_ORDER = (
    'Analysis', 'Label', 'Hours tested', 'Hold files', 'Hold segments', 'Step files',
    'geo_area', 'loading', 'Folders left out', 'OCV initial', 'OCV final', 'OCV loss',
    'OCV decay rate', 'OCV min', 'OCV death', 'j_xover initial', 'j_xover final',
    'R_short final', 'RH cycles tested', 'Verdict', 'Hours to death',
    'RH cycles to death', 'Death cause', 'Death bracket', 'OCV deaths dismissed',
)
# The summary fields a dismissal can change.
SUMMARY_DEATH_KEYS = ('OCV death', 'OCV decay rate', 'Verdict', 'Hours to death',
                      'RH cycles to death', 'Death cause', 'Death bracket',
                      'OCV deaths dismissed')

OCV_PLOT = 'fcdur_ocv_vs_time'
TIMELINE_PLOT = 'fcdur_timeline'
RH_PLOT = 'fcdur_rh_cycles'
# The readout values of each plot a rewrite changes.
DEATH_VALUE_KEYS = {
    OCV_PLOT: ('OCV decay rate', 'OCV death', 'Later OCV deaths', 'First later OCV death',
               'Dismissed OCV deaths'),
    TIMELINE_PLOT: ('Verdict', 'Hours to death', 'RH cycles to death', 'Death cause',
                    'Death bracket'),
    RH_PLOT: ('RH cycles to death',),
}


def _num(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


def kv(key: str, x: Any, spec: str, unit: str) -> Optional[str]:
    """A readout line, as the analysis writes them: none for no number."""
    v = _num(x)
    return f'{key} = {v:{spec}} {unit}' if v is not None else None


# ─────────────────────────────────────────────────────────────────────
#  Dismissals as given
# ─────────────────────────────────────────────────────────────────────

def clean_dismissals(given: Any) -> List[Dict[str, Any]]:
    """The dismissals a job was given, those that name a place: a list of
    dicts with a step and a time into it. Anything else is left out."""
    out: List[Dict[str, Any]] = []
    for d in given if isinstance(given, list) else []:
        if not isinstance(d, dict):
            continue
        off = _num(d.get('offset_h'))
        step = d.get('step')
        if off is None or not isinstance(step, str) or not step:
            continue
        out.append(d)
    return out


def canon(dismissals: Iterable[Dict[str, Any]]) -> List[Tuple[Any, ...]]:
    """What decides how dismissals judge: the same for two lists that judge alike."""
    return sorted((str(d.get('id') or ''), str(d.get('folder') or ''), str(d.get('step') or ''),
                   round(_num(d.get('offset_h')) or 0.0, 9)) for d in dismissals or [])


def reason_text(d: Dict[str, Any]) -> str:
    return REASON_TEXT.get(str(d.get('reason') or ''), str(d.get('reason') or 'other'))


def day(at: Any) -> str:
    """'6 Nov 2026' from an ISO time, or '' if it is not one."""
    m = re.match(r'(\d{4})-(\d{2})-(\d{2})', str(at or ''))
    if not m:
        return ''
    months = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return f'{d} {months[mo - 1]} {y}' if 1 <= mo <= 12 else ''


# ─────────────────────────────────────────────────────────────────────
#  Which death counts
# ─────────────────────────────────────────────────────────────────────

def gap_h(d: Dict[str, Any], ev: Dict[str, Any]) -> Optional[float]:
    """How far apart a dismissal and a death start in their step file (h):
    None for a death in another file (folder and step), or where either has
    no time into it."""
    if (str(ev.get('folder') or ''), str(ev.get('step') or '')) != \
            (str(d.get('folder') or ''), str(d.get('step') or '')):
        return None
    off, e_off = _num(d.get('offset_h')), _num(ev.get('offset_h'))
    return abs(e_off - off) if off is not None and e_off is not None else None


def match(d: Dict[str, Any], events: List[Dict[str, Any]], taken: Iterable[int] = ()
          ) -> Optional[int]:
    """The event a dismissal names: the one starting in its step file (folder
    and step) nearest to where it started, within MATCH_MIN; None if none."""
    tol = MATCH_MIN / 60.0 + 1e-9
    taken = set(taken)
    best: Optional[Tuple[float, int]] = None
    for i, ev in enumerate(events):
        gap = None if i in taken else gap_h(d, ev)
        if gap is not None and gap <= tol and (best is None or gap < best[0]):
            best = (gap, i)
    return best[1] if best else None


def nearest(d: Dict[str, Any], events: List[Dict[str, Any]], taken: Iterable[int] = ()
            ) -> Optional[int]:
    """The death in a dismissal's step file nearest to it, however far, but
    one already dismissed (`taken`): what an admin is offered when the
    dismissal matches none."""
    taken = set(taken)
    best: Optional[Tuple[float, int]] = None
    for i, ev in enumerate(events):
        gap = None if i in taken else gap_h(d, ev)
        if gap is not None and (best is None or gap < best[0]):
            best = (gap, i)
    return best[1] if best else None


def nearest_any(d: Dict[str, Any], events: List[Dict[str, Any]], taken: Iterable[int] = ()
                ) -> Optional[int]:
    """The death nearest a dismissal on the clock, in any file, but one
    already dismissed: what an admin is offered when none is in its file
    (a rerun that moved it into the next file of the hold, say)."""
    t = _num(d.get('start_h'))
    taken = set(taken)
    best: Optional[Tuple[float, int]] = None
    for i, ev in enumerate(events):
        e_t = _num(ev.get('start_h'))
        if i not in taken and t is not None and e_t is not None \
                and (best is None or abs(e_t - t) < best[0]):
            best = (abs(e_t - t), i)
    return best[1] if best else None


def verdict(table: Dict[str, Any], ev: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The verdict with `ev` as the OCV death (None: no OCV death), by the
    analysis's rule (decide): the earlier of it and the crossover death
    counts; 'both' when the OCV death falls inside the crossover bracket;
    neither leaves the sample alive at its hours tested."""
    t = _num(ev.get('start_h')) if ev else None
    xd = table.get('xover') if isinstance(table.get('xover'), dict) else None
    cycles_tested = table.get('cycles_tested') or None
    base = {'ocv_judged': bool(table.get('ocv_judged', True)),
            'xover_judged': bool(table.get('xover_judged', True))}
    if t is None and xd is None:
        return dict(base, dead=False, hours=float(table.get('hours_tested') or 0.0), cause='',
                    bracket=None, at_bol=False, trigger='', cycles=cycles_tested)
    if xd is not None and (t is None or float(xd['H']) < t - CLOCK_TOL_H):
        return dict(base, dead=True, hours=float(xd['H']), cause='crossover',
                    bracket=[float(xd['low']), float(xd['H'])], at_bol=bool(xd.get('at_bol')),
                    trigger='', cycles=xd.get('cycles') if cycles_tested else None)
    both = xd is not None and float(xd['low']) < t <= float(xd['H']) + CLOCK_TOL_H
    return dict(base, dead=True, hours=t, cause='both' if both else 'OCV',
                bracket=[float(xd['low']), float(xd['H'])] if both else None, at_bol=False,
                trigger=str(ev.get('trigger') or ''),
                cycles=ev.get('cycles') if cycles_tested else None)


def cause_text(v: Dict[str, Any]) -> str:
    if v.get('trigger') == 'station' and v.get('cause') in ('OCV', 'both'):
        return f"{v['cause']} (station limit)"
    return str(v.get('cause') or '')


def bracket_text(v: Dict[str, Any]) -> str:
    if not v.get('dead') or v.get('cause') != 'crossover' or not v.get('bracket'):
        return ''
    lo, hi = v['bracket']
    return 'at BOL' if v.get('at_bol') else f'({lo:.2f}, {hi:.2f}] h'


def verdict_text(v: Dict[str, Any]) -> str:
    if not v.get('dead'):
        return f"alive at {v['hours']:.2f} h"
    if v.get('at_bol'):
        return 'dead at BOL (0 h), crossover'
    if v.get('bracket') and v.get('cause') == 'crossover':
        lo, hi = v['bracket']
        return f"dead at {v['hours']:.2f} h, crossover, between {lo:.2f} and {hi:.2f} h"
    return f"dead at {v['hours']:.2f} h, {cause_text(v)}"


def resolve(table: Dict[str, Any], dismissals: Iterable[Dict[str, Any]] = ()) -> Dict[str, Any]:
    """Which death counts, given the dismissals.

    Each dismissal takes the event it matches (in the order given, one event
    each); the first event left is the death, the rest later deaths. The
    verdict and the decay rate follow: the rate ending by the death the
    verdict names (the crossover's, if it came first), or uncut for a sample
    left alive.
    """
    events = list(table.get('events') or [])
    given = list(dismissals or [])
    # The nearest pairs first, each death and each dismissal taken once: of two
    # dismissed deaths a rerun finds one of, the dismissal nearest it takes it.
    tol = MATCH_MIN / 60.0 + 1e-9
    pairs = sorted((gap, j, i) for j, d in enumerate(given) for i, ev in enumerate(events)
                   for gap in (gap_h(d, ev),) if gap is not None and gap <= tol)
    by_event: Dict[int, Dict[str, Any]] = {}
    used = set()
    for _, j, i in pairs:
        if i not in by_event and j not in used:
            by_event[i] = given[j]
            used.add(j)
    unmatched = [d for j, d in enumerate(given) if j not in used]
    heads = [i for i in range(len(events)) if i not in by_event]
    death = heads[0] if heads else None
    # Deaths past the stored table (TABLE_EVENTS_MAX) count, as later deaths.
    found = max(len(events), int(_num(table.get('events_found')) or 0))
    beyond = found - len(events)
    v = verdict(table, events[death] if death is not None else None)
    if not v['dead']:
        rate = table.get('rate_uncut')
    elif v['cause'] == 'crossover':
        rate = (table.get('xover') or {}).get('rate')
    else:
        rate = events[death].get('rate')
    return {'events': events, 'death': death, 'later': heads[1:],
            'later_n': len(heads) - 1 + beyond if death is not None else 0,
            'found': found, 'beyond': beyond,
            'dismissed': sorted(by_event), 'by_event': by_event, 'unmatched': unmatched,
            'verdict': v, 'rate': rate if isinstance(rate, dict) else None,
            'judged': bool(table.get('ocv_judged', True))}


def death_h(res: Dict[str, Any]) -> Optional[float]:
    return _num(res['events'][res['death']]['start_h']) if res['death'] is not None else None


def kept_first(table: Dict[str, Any]) -> int:
    """How many of the first deaths the table keeps (past them it keeps only
    those a dismissal takes)."""
    n = len(table.get('events') or [])
    cap = _num(table.get('events_max'))
    return min(n, int(cap)) if cap is not None else n


# ─────────────────────────────────────────────────────────────────────
#  What the outputs show of it
# ─────────────────────────────────────────────────────────────────────

def summary_fields(res: Dict[str, Any]) -> Dict[str, Any]:
    """The summary row's death fields, as the analysis writes them."""
    out: Dict[str, Any] = {}
    rate = _num((res.get('rate') or {}).get('uV_h'))
    if rate is not None:
        out['OCV decay rate'] = round(rate, 1)
    dh = death_h(res)
    if dh is not None:
        out['OCV death'] = round(dh, 4)
    v = res['verdict']
    out['Verdict'] = 'dead' if v['dead'] else 'alive'
    if v['dead']:
        out['Hours to death'] = round(v['hours'], 4)
        if v['cycles'] is not None:
            out['RH cycles to death'] = int(v['cycles'])
        out['Death cause'] = cause_text(v)
        if bracket_text(v):
            out['Death bracket'] = bracket_text(v)
    if res['dismissed']:
        out['OCV deaths dismissed'] = len(res['dismissed'])
    return out


def ordered_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """A summary row in the analysis's order, keys it does not know after."""
    out = {k: row[k] for k in SUMMARY_ORDER if k in row}
    out.update({k: v for k, v in row.items() if k not in out})
    return out


def verdict_lines(v: Dict[str, Any]) -> List[str]:
    """The timeline readout's verdict lines. Hours to death is the only
    number; the verdict, cause and bracket are text, as in the summary row."""
    if not v['dead']:
        lines = ['Verdict: alive']
        if not v['ocv_judged'] or not v['xover_judged']:
            lines.append(' and '.join(
                n for n, ok in (('OCV rule', v['ocv_judged']), ('crossover rule', v['xover_judged']))
                if not ok) + ' not applied (no data)')
        return lines
    lines = ['Verdict: dead', f"Hours to death = {v['hours']:.2f} h"]
    if v['cycles'] is not None:
        lines.append(f"RH cycles to death = {v['cycles']}")
    lines.append(f'Death cause: {cause_text(v)}')
    if bracket_text(v):
        lines.append(f'Death bracket: {bracket_text(v)}')
    return lines


def verdict_line(v: Dict[str, Any], rh: bool = False) -> Optional[Dict[str, Any]]:
    """The death line of the timeline (or, `rh`, the RH cycles plot)."""
    if not v['dead']:
        return None
    label = f"death at {v['hours']:.2f} h" + (f", RH cycle {v['cycles']}" if rh else '')
    return {'x': round(v['hours'], 4), 'color': DEATH_COLOR, 'ls': '-', 'lw': 1.6, 'label': label}


def ocv_lines(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The OCV plot's death lines, in the order drawn: the death solid, the
    first LATER_LISTED later deaths dash-dot (apart from the threshold's
    dashes), each dismissed death grey and dotted."""
    ev, out = res['events'], []
    if res['death'] is not None:
        d = ev[res['death']]
        how = ", the station's own limit" if d.get('trigger') == 'station' else ''
        out.append({'x': round(d['start_h'], 4), 'color': DEATH_COLOR, 'ls': '-', 'lw': 1.6,
                    'label': f"OCV death at {d['start_h']:.2f} h{how}"})
    later, n = res['later'], res.get('later_n', len(res['later']))
    for j, i in enumerate(later[:LATER_LISTED]):
        out.append({'x': round(ev[i]['start_h'], 4), 'color': DEATH_COLOR, 'ls': '-.', 'lw': 1.3,
                    'label': (LATER_LABEL + (f', the first {LATER_LISTED} of {n} drawn'
                                             if n > LATER_LISTED else '')) if j == 0 else '_nolegend_'})
    for j, i in enumerate(res['dismissed']):
        out.append({'x': round(ev[i]['start_h'], 4), 'color': DISMISSED_COLOR, 'ls': ':',
                    'lw': 1.6, 'label': DISMISSED_LABEL if j == 0 else '_nolegend_'})
    return out


def ocv_rate_line(res: Dict[str, Any]) -> Optional[str]:
    return kv('OCV decay rate', (res.get('rate') or {}).get('uV_h'), '.0f', 'μV/h')


def ocv_death_lines(res: Dict[str, Any]) -> List[str]:
    """The OCV readout's death lines: the death, and once there is one how
    many later deaths (0 too) and when the first was; how many dismissed.
    Numbers the record keeps as such."""
    dh = death_h(res)
    lines = [kv('OCV death', dh, '.2f', 'h') if dh is not None
             else 'OCV death: none' if res['judged'] else 'OCV death: not judged']
    if dh is not None:
        lines.append(f"Later OCV deaths = {res.get('later_n', len(res['later']))}")
        if res['later']:
            lines.append(kv('First later OCV death', res['events'][res['later'][0]]['start_h'],
                            '.2f', 'h'))
    if res['dismissed']:
        lines.append(f"Dismissed OCV deaths = {len(res['dismissed'])}")
    return [ln for ln in lines if ln]


# ─────────────────────────────────────────────────────────────────────
#  Rewriting a stored record
# ─────────────────────────────────────────────────────────────────────

def _rgba(hex_color: str) -> List[float]:
    from matplotlib.colors import to_rgba
    return [float(x) for x in to_rgba(hex_color)]


def _is(color: Any, hex_color: str) -> bool:
    try:
        return all(abs(float(a) - b) < 1e-6 for a, b in zip(color, _rgba(hex_color)))
    except (TypeError, ValueError):
        return False


def _vline(spec: Dict[str, Any]) -> Dict[str, Any]:
    """A line spec as the sidecar writer stores an axvline."""
    label = spec['label']
    return {'x': float(spec['x']), 'color': _rgba(spec['color']), 'linestyle': spec['ls'],
            'label': None if str(label).startswith('_') else label}


def _death_vline(vl: Dict[str, Any]) -> bool:
    """A death line, as these plots draw them: the only red or dismissed-grey
    vertical lines they have (characterizations are grey, hold boundaries
    black)."""
    c = vl.get('color')
    return _is(c, DEATH_COLOR) or _is(c, DISMISSED_COLOR)


def _readout(ax: Dict[str, Any], anchor: str) -> Optional[Dict[str, Any]]:
    return next((t for t in ax.get('texts') or [] if anchor in str(t.get('text') or '')), None)


def _rewrite_ocv(sc: Dict[str, Any], res: Dict[str, Any]) -> None:
    axes = (sc.get('data') or {}).get('axes') or []
    if not axes:
        return
    top = axes[0]
    top['axvlines'] = ([v for v in top.get('axvlines') or [] if not _death_vline(v)]
                       + [_vline(s) for s in ocv_lines(res)])
    t = _readout(top, 'OCV plot bin')
    if t is None:
        return
    prefixes = ('OCV decay rate = ', 'OCV death', 'Later OCV deaths', 'First later OCV death',
                'Dismissed OCV deaths')
    lines = [ln for ln in str(t['text']).split('\n') if not ln.startswith(prefixes)]
    at = next((i for i, ln in enumerate(lines) if ln.startswith('Hours tested = ')), len(lines))
    lines[at:at] = ocv_death_lines(res)
    rate = ocv_rate_line(res)
    if rate:
        at = next((i for i, ln in enumerate(lines)
                   if ln.startswith(('OCV min (10-min median)', 'OCV death'))), len(lines))
        lines.insert(at, rate)
    t['text'] = '\n'.join(lines)


def _rewrite_timeline(sc: Dict[str, Any], res: Dict[str, Any]) -> None:
    axes = (sc.get('data') or {}).get('axes') or []
    if not axes:
        return
    ax = axes[0]
    line = verdict_line(res['verdict'])
    ax['axvlines'] = ([v for v in ax.get('axvlines') or [] if not _death_vline(v)]
                      + ([_vline(line)] if line else []))
    t = _readout(ax, 'Hold segments = ')
    if t is None:
        return
    lines = str(t['text']).split('\n')
    cut = next((i for i, ln in enumerate(lines) if ln.startswith('Verdict:')), len(lines))
    t['text'] = '\n'.join(lines[:cut] + verdict_lines(res['verdict']))


def _rewrite_rh(sc: Dict[str, Any], res: Dict[str, Any]) -> None:
    axes = (sc.get('data') or {}).get('axes') or []
    if not axes:
        return
    ax, v = axes[0], res['verdict']
    line = verdict_line(v, rh=True)
    ax['axvlines'] = ([vl for vl in ax.get('axvlines') or [] if not _death_vline(vl)]
                      + ([_vline(line)] if line else []))
    t = _readout(ax, 'RH cycles tested = ')
    if t is None:
        return
    lines = [ln for ln in str(t['text']).split('\n') if not ln.startswith('RH cycles to death = ')]
    if v['dead'] and v['cycles'] is not None:
        lines.append(f"RH cycles to death = {v['cycles']}")
    t['text'] = '\n'.join(lines)


REWRITERS = {OCV_PLOT: _rewrite_ocv, TIMELINE_PLOT: _rewrite_timeline, RH_PLOT: _rewrite_rh}


def apply_to_record(record: Dict[str, Any], sidecars: Optional[Dict[str, Any]] = None
                    ) -> Optional[Dict[str, Any]]:
    """Bring a stored FC Durability record to its dismissals, in place: the
    summary row's death fields, the death lines and readouts of the OCV,
    timeline and RH cycles plots, and the values the record reads off them.

    `sidecars`, when given, is the decoded set to rewrite (a push attaches it
    after); otherwise the record's own are decoded, rewritten and attached.
    Returns the resolution, or None for a record without the table.
    """
    from scripts.helpers import record as rec_mod

    table = record.get('fc_deaths')
    if not isinstance(table, dict):
        return None
    res = resolve(table, record.get('ocv_dismissals') or [])

    given = record.get('summary')
    # A new list: a push's record shares its job's.
    rows = list(given) if isinstance(given, list) else ([given] if isinstance(given, dict) else [])
    for i, row in enumerate(rows):
        if isinstance(row, dict) and row.get('Analysis') == 'fc_durability':
            kept = {k: v for k, v in row.items() if k not in SUMMARY_DEATH_KEYS}
            kept.update(summary_fields(res))
            rows[i] = ordered_row(kept)
    if isinstance(given, list):
        record['summary'] = rows
    elif isinstance(given, dict):           # a lone row, kept as one
        record['summary'] = rows[0]

    own = sidecars is None
    sc_all = rec_mod.decode_sidecars(record) if own else sidecars
    rewritten = []
    for name, sc in sc_all.items():
        fn = REWRITERS.get(str((sc or {}).get('plot_type') or '')) if isinstance(sc, dict) else None
        if fn:
            fn(sc, res)
            rewritten.append(name)
    if own and rewritten:
        rec_mod.attach_sidecars(record, sc_all)
    for plots in (record.get('metrics') or {}).values():
        for name, entry in plots.items():
            if not isinstance(entry, dict):
                continue
            if name in rewritten:
                entry['values'] = rec_mod.extract_values(sc_all[name])
                continue
            # A plot whose data the record could not keep keeps its values,
            # which the index reads when the row has none (hours to death of
            # a sample alive): those of its readout's death lines, rewritten.
            kind = next((t for t in REWRITERS if str(name).endswith(t)), None)
            if kind and name not in sc_all:
                vals = {k: v for k, v in (entry.get('values') or {}).items()
                        if k not in DEATH_VALUE_KEYS[kind]}
                vals.update(_death_values(kind, res, rec_mod))
                entry['values'] = vals
    return res


def _death_values(kind: str, res: Dict[str, Any], rec_mod: Any) -> Dict[str, Any]:
    """The values a plot's readout gives of the death, read as the record
    reads a readout."""
    v = res['verdict']
    if kind == OCV_PLOT:
        lines = ocv_death_lines(res) + [ocv_rate_line(res)]
    elif kind == TIMELINE_PLOT:
        lines = verdict_lines(v)
    else:
        lines = [f"RH cycles to death = {v['cycles']}"] if v['dead'] and v['cycles'] is not None else []
    text = '\n'.join(ln for ln in lines if ln)
    return rec_mod.extract_values({'data': {'axes': [{'texts': [{'text': text}]}]}}) if text else {}


# ─────────────────────────────────────────────────────────────────────
#  What the page and the admin panel are shown
# ─────────────────────────────────────────────────────────────────────

def event_view(ev: Dict[str, Any], status: str, i: int) -> Dict[str, Any]:
    """A death as the page shows it: `i` its place in the table, `n` among
    all found (they differ past the first TABLE_EVENTS_MAX)."""
    return {'i': i, 'n': int(_num(ev.get('n')) or i + 1), 'status': status,
            'start_h': ev.get('start_h'),
            'where': ev.get('hold_label') or '', 'folder': ev.get('folder') or '',
            'step': ev.get('step') or '', 'offset_h': ev.get('offset_h'),
            'minutes': ev.get('minutes'),
            'lowest_v': ev.get('lowest_v'), 'back_h': ev.get('back_h'),
            'trigger': ev.get('trigger') or '', 'acted': ev.get('acted') or '',
            'cycles': ev.get('cycles')}


def view(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The deaths of a stored test and their dismissals, for Recall and the
    admin panel; None for a record without the table."""
    table = record.get('fc_deaths')
    if not isinstance(table, dict):
        return None
    dis = list(record.get('ocv_dismissals') or [])
    res = resolve(table, dis)
    events = res['events']
    rows = []
    for i, ev in enumerate(events):
        status = ('dismissed' if i in res['by_event'] else
                  'death' if i == res['death'] else 'later death')
        row = event_view(ev, status, i)
        if i in res['by_event']:
            row['dismissal'] = res['by_event'][i].get('id')
        rows.append(row)
    unmatched = {id(d) for d in res['unmatched']}
    dviews = []
    for d in dis:
        out = {k: d.get(k) for k in ('id', 'start_h', 'hold_label', 'folder', 'step', 'offset_h',
                                     'kind', 'trigger', 'reason', 'note', 'at', 'by', 'kept',
                                     'moved_at')}
        out['reason_text'] = reason_text(d)
        if id(d) in unmatched:
            out['status'] = 'kept' if d.get('kept') else 'review'
            j, other = nearest(d, events, res['by_event']), False
            if j is None:
                j, other = nearest_any(d, events, res['by_event']), True
            if j is not None:
                out['nearest'] = dict(event_view(events[j], 'nearest', j), other_file=other)
        else:
            i = next(k for k, x in res['by_event'].items() if x is d)
            out['status'] = 'applied'
            out['event'] = rows[i]
        dviews.append(out)
    v = res['verdict']
    last = analysed_at(record)
    for out, d in zip(dviews, dis):
        out['kept_through'] = out['status'] == 'applied' and kept_through(record, d)
    return {'events': rows, 'dismissals': dviews, 'death': res['death'],
            'death_h': death_h(res), 'verdict': verdict_text(v), 'dead': v['dead'],
            'cause': cause_text(v), 'hours': v['hours'],
            'hours_tested': table.get('hours_tested'),
            'threshold_v': table.get('threshold_v'), 'persist_min': table.get('persist_min'),
            'found': res['found'], 'later_n': res['later_n'], 'kept_first': kept_first(table),
            'rate': res['rate'], 'analysed_at': last,
            'review': sum(1 for x in dviews if x['status'] == 'review'),
            'history': history_of(record)[-HISTORY_MAX:],
            'match_min': MATCH_MIN}


def history_of(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """A record's death history, its entries only: a record edited by hand
    may hold anything."""
    return [h for h in record.get('ocv_death_history') or [] if isinstance(h, dict)]


def kept_through(record: Dict[str, Any], d: Dict[str, Any]) -> bool:
    """Whether an analysis was stored since the dismissal was made or last
    moved: by the history's order, which a second does not blur (an analysis
    is stored seconds after a rerun starts); by the times when the history
    no longer holds the entry."""
    hist = history_of(record)
    marks = [i for i, h in enumerate(hist)
             if h.get('id') == d.get('id') and h.get('what') in ('dismissed', 'move')]
    if marks:
        return any(h.get('what') == 'analysed' for h in hist[marks[-1] + 1:])
    since = max(str(d.get('at') or ''), str(d.get('moved_at') or ''))
    last = analysed_at(record)
    return bool(last) and str(last) > since


def analysed_at(record: Dict[str, Any]) -> str:
    """When the test's FC Durability analysis was last stored."""
    hist = [h for h in history_of(record) if h.get('what') == 'analysed']
    if hist:
        return str(hist[-1].get('at') or '')
    return str((record.get('fc_deaths') or {}).get('at') or record.get('timestamp') or '')


def index_summary(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """What the index keeps of a test's deaths, for the admin list and
    Recall: the time of death, how many deaths were found and dismissed, and
    how many dismissals wait for an admin. None when there is nothing."""
    table = record.get('fc_deaths')
    dis = record.get('ocv_dismissals') or []
    if not isinstance(table, dict):
        return None
    res = resolve(table, dis)
    return {'death_h': death_h(res), 'found': res['found'],
            'dismissed': len(res['dismissed']), 'dismissals': len(dis),
            'review': sum(1 for d in res['unmatched'] if not d.get('kept')),
            'trigger': str(res['events'][res['death']].get('trigger') or '')
            if res['death'] is not None else ''}


# ─────────────────────────────────────────────────────────────────────
#  History
# ─────────────────────────────────────────────────────────────────────

def _where(ev_or_d: Dict[str, Any]) -> str:
    w = ev_or_d.get('hold_label') or ''
    return f' in {w}' if w else ''


def history_add(record: Dict[str, Any], what: str, text: str, **extra: Any) -> None:
    res = resolve(record['fc_deaths'], record.get('ocv_dismissals') or []) \
        if isinstance(record.get('fc_deaths'), dict) else None
    entry = {'at': extra.pop('at', None) or now(), 'what': what, 'text': text,
             'death_h': death_h(res) if res else None,
             'dead': bool(res and res['verdict']['dead']),
             'verdict': verdict_text(res['verdict']) if res else '',
             # The dismissals applied then, and those waiting for an admin.
             'applied': len(res['by_event']) if res else 0,
             'review': sum(1 for d in res['unmatched'] if not d.get('kept')) if res else 0}
    entry.update({k: v for k, v in extra.items() if v is not None})
    record['ocv_death_history'] = capped(history_of(record) + [entry])


def analysis_text(record: Dict[str, Any]) -> str:
    """The history's line for an analysis stored: what it found, and what
    became of each dismissal."""
    table = record['fc_deaths']
    res = resolve(table, record.get('ocv_dismissals') or [])
    ev = res['events']
    again = any(h.get('what') == 'analysed' for h in history_of(record))
    parts = [f"{'Re-analysed' if again else 'Analysed'}, "
             f"{float(table.get('hours_tested') or 0):.2f} h tested"]
    found = res['found']
    if not found:
        parts.append('no OCV death')
    else:
        first = ev[0]
        parts.append(f"OCV death at {first['start_h']:.2f} h{_where(first)}"
                     + (f", {found - 1} later death{'s' if found != 2 else ''}" if found > 1 else ''))
    for i, d in sorted(res['by_event'].items()):
        parts.append(f"the death at {ev[i]['start_h']:.2f} h found again{_where(ev[i])}, "
                     f"kept dismissed")
    for d in res['unmatched']:
        parts.append(f"no death at {float(d.get('start_h') or 0):.2f} h{_where(d)}: "
                     + ('the dismissal kept on file' if d.get('kept') else
                        'the dismissal not applied, to review'))
    return '; '.join(parts[:1]) + (': ' + '; '.join(parts[1:]) if len(parts) > 1 else '')


def new_dismissal(ev: Dict[str, Any], kind: str, reason: str, note: str) -> Dict[str, Any]:
    """A dismissal of an event, naming it by where it starts."""
    return {'id': uuid.uuid4().hex[:12], 'start_h': ev.get('start_h'),
            'folder': ev.get('folder') or '', 'step': ev.get('step') or '',
            'offset_h': ev.get('offset_h'), 'hold_label': ev.get('hold_label') or '',
            'kind': kind, 'trigger': ev.get('trigger') or '', 'reason': reason,
            'note': note, 'at': now(), 'by': 'admin'}


# What an admin does to a dismissal, as the death history names it; with an
# analysis that applied one kept on file again (`unkept`), the last of these
# for its id is what became of it on that record.
CHANGES = ('dismissed', 'move', 'kept', 'remove', 'restored')
TAKEN_BACK = ('remove', 'restored')


def _dkey(d: Dict[str, Any]) -> str:
    return str(d.get('id') or json.dumps(d, sort_keys=True))


def last_change(hist: Any, did: str) -> Optional[Tuple[str, int, str]]:
    """When a record's history last says the dismissal changed -- an admin's
    change, or an analysis that applied it again, kept on file till then:
    (at, place in the history, what), or None."""
    last = None
    for n, h in enumerate(hist or []):
        if not isinstance(h, dict):
            continue
        if h.get('id') == did and h.get('what') in CHANGES:
            last = (str(h.get('at') or ''), n, str(h['what']))
        elif isinstance(h.get('unkept'), list) and did in h['unkept']:
            last = (str(h.get('at') or ''), n, 'unkept')
    return last


def merge_dismissals(a: Any, b: Any, ha: Any = None, hb: Any = None) -> List[Dict[str, Any]]:
    """Two records' dismissals as one sample's, each once, by its id, as the
    record it last changed on has it: moved or kept on file there, so;
    restored or removed there, gone. When it last changed on a record is the
    later of when its history (`ha`, `hb`) last says so and when its copy
    there was made or moved -- a change the history no longer holds, or a copy
    a push or a merge carried from another record -- a change said at the
    same second following the copy. Otherwise the second's on a tie. A change
    made in either of two duplicates stands."""
    def by_id(xs: Any) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for d in xs or []:
            if isinstance(d, dict):
                out.setdefault(_dkey(d), d)
        return out

    def when(last: Optional[Tuple[str, int, str]], d: Optional[Dict[str, Any]]
             ) -> Optional[Tuple[str, int, str]]:
        said = (last[0], 1, last[2]) if last else None
        own = (max(str(d.get('at') or ''), str(d.get('moved_at') or '')), 0, 'made') if d else None
        return max((w for w in (said, own) if w), default=None)

    da, db = by_id(a), by_id(b)
    out: List[Dict[str, Any]] = []
    for k in list(da) + [k for k in db if k not in da]:
        x, y = da.get(k), db.get(k)
        wx, wy = when(last_change(ha, k), x), when(last_change(hb, k), y)
        mine, other, w = (y, x, wy) if wy and (not wx or wy[:2] >= wx[:2]) else (x, y, wx)
        if w[2] in TAKEN_BACK:
            continue
        out.append(mine or other)
    return out


def capped(hist: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """A death history's entries (as history_of and merge_history give them)
    at most HISTORY_MAX long: the oldest analyses go first, so the changes a
    merge of duplicates goes by (last_change) outlast them; then the oldest of
    the rest."""
    over = len(hist) - HISTORY_MAX
    if over <= 0:
        return hist
    drop = set([n for n, h in enumerate(hist)
                if h.get('what') == 'analysed' and not h.get('unkept')][:over])
    return [h for n, h in enumerate(hist) if n not in drop][-HISTORY_MAX:]


def merge_history(a: Any, b: Any) -> List[Dict[str, Any]]:
    """Two records' death histories as one: each entry once, in time order
    (a stable sort, so one record's entries of a second keep theirs), capped."""
    out: List[Dict[str, Any]] = []
    seen = set()
    for h in list(a or []) + list(b or []):
        if isinstance(h, dict):
            k = json.dumps(h, sort_keys=True)
            if k not in seen:
                seen.add(k)
                out.append(h)
    out.sort(key=lambda h: str(h.get('at') or ''))
    return capped(out)


def copy_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(record)


def applied_again(record: Dict[str, Any]) -> List[str]:
    """A dismissal kept on file whose death the record's table has applies
    again, and waits again if a later analysis loses it: its `kept` cleared.
    The ids of those, for the history a push writes (`unkept`), which a merge
    of duplicates goes by."""
    stored = record.get('ocv_dismissals') or []
    again = {d.get('id') for d in resolve(record['fc_deaths'], stored)['by_event'].values()
             if d.get('kept')}
    if again:
        record['ocv_dismissals'] = [
            {k: v for k, v in d.items() if k != 'kept'} if d.get('id') in again else d
            for d in stored]
    return sorted(str(x) for x in again if x)


def judge_merged(detail: Dict[str, Any], entry: Dict[str, Any], bin_id: Any,
                 sidecars: Optional[Dict[str, Any]] = None) -> None:
    """Stored records of one sample merged into one (record.merge_detail_record:
    the admin's merge of duplicates, find_duplicates and migrate_samples
    alike): their dismissals judged against the newest analysis's table, as a
    push judges them -- its row, plots and values brought to them -- and the
    merged index entry's FC Durability units and deaths read off the record.
    `sidecars`, when given, is the decoded set the caller attaches after.
    Nothing for a record without the table."""
    from scripts.helpers import record as rec_mod
    if not isinstance(detail.get('fc_deaths'), dict):
        return
    applied_again(detail)
    apply_to_record(detail, sidecars=sidecars)
    rebuilt = rec_mod.build_index_entry(detail, bin_id)
    entry['Data'] = ([u for u in entry.get('Data') or [] if u.get('Analysis') != 'fc_durability']
                     + [u for u in rebuilt.get('Data') or [] if u.get('Analysis') == 'fc_durability'])
    entry['ocv_deaths'] = index_summary(detail)


def stored_dismissals(sample_name: str) -> List[Dict[str, Any]]:
    """The dismissals the store keeps for a sample, for a rerun to be judged
    with: [] for a sample not stored, or a store not set up. Raises if the
    store cannot be read (the push then applies them itself). The record is
    read for any test with a death found or a dismissal: an index entry an
    admin change could not rewrite says too little."""
    from scripts.helpers import jsonbin
    if not sample_name or not jsonbin.is_configured():
        return []
    index = jsonbin.fetch_index()
    _, entry = jsonbin.find_sample_entry(index, sample_name)
    summ = (entry or {}).get('ocv_deaths') or {}
    if not entry or not entry.get('bin_id') or not (summ.get('found') or summ.get('dismissals')):
        return []
    record = jsonbin.fetch_detail_bin(entry['bin_id'])
    return clean_dismissals(record.get('ocv_dismissals') or [])
