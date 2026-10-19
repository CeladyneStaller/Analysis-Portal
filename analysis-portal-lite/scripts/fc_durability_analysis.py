#!/usr/bin/env python3
"""
Fuel Cell OCV-Hold Durability Analyzer
======================================

Reads a durability run as the test stations write it -- one CSV per protocol
step, either in a single folder (Scribner) or as a break-in/BOL folder plus
one folder per 40 h segment (FCTS) -- and builds the timeline that sample
death is measured on. File names carry everything the timeline needs:

    {step}_{sample}_{kind}_{T}Cell-{RH}RH_{anode}-{cathode}.csv
    c5_Gore-A_OCV_95Cell-30RH_0.1WH2-0.2WA.csv

  * Which files are OCV holds. Most files named OCV are not: H2/N2 steps sit
    near 0.1 V for hours and purges fall through 0.7 V in minutes, so a hold
    is an OCV file with air or O2 on the cathode that runs at least 2 h.
    Consecutive hold files at the same conditions in the same folder form
    one hold (Gore-A's 150 h is fifteen 10 h files).
  * The order of every step. No file carries a date and every log restarts
    near 0 s, so folders are ordered by their earliest step letter and then
    by natural name, and steps by letter, number and any letter after the
    number (f3, f3b, f4) within a folder.
  * The hold-hour clock. Each hold file adds its last time stamp plus the
    30 s that every station step runs past the end of its log.

The Scribner stand's own software (FuelCell) writes a .fcd file per step
instead, all in one folder, with no sample in the name and 'o' for the
decimal point:

    {step}_{kind}[_{range}]_{T}C-{RH}RH_{anode}-{cathode}_{P}kPa.fcd
    c4_OCV-RHC-1000cyc_90C-100RH(90C)-0RH_0o1H2-0o1Air_0kPa.fcd

CVs are named by scan rate (CV-2mVs crossover, CV-50mVs ECSA, CV-500mVs
cleaning), the pol curve is IV and EIS is GEIS. A .fcd header runs to its
column line and 'End Comments'. Its Time (Sec) runs on from the start of the
station session, so a step's own clock starts one interval before its first
sample and the step ends with its last: no tail is added. Its columns are
read under the station CSV names, so everything below reads both formats.

The hold OCV is then judged on that clock. A hold's files are stitched end
to end; samples drawing current, and the first 10 min of each hold while the
OCV settles, are not judged. Death on OCV is the start of the first stretch
in which the centred 10-min rolling median of the judged samples stays below
0.7 V for 30 min. A .fcd hold whose header carries its own voltage limit
(Term:E<0.7) also dies when the station acts on it, however briefly: when
it ends the step, or cuts a wet half of the RH-cycling loop short and the
loop runs on. Each hold also gives a settled start and end OCV (30-min
medians) and a rolling 10 h decay rate; the test's OCV loss runs from the
first hold's settled start to the last hold's end at the first hold's
conditions.

Every crossover, ECSA, EIS and pol-curve file is analyzed by the portal's
own functions on columns read by header name, and stamped with the hold
hours before it. The last good repeat of each measurement represents its
characterization block. Death on crossover is the first characterization
whose 100% RH crossover -- the window mean of the averaged CV current over
0.35-0.50 V -- exceeds 15 mA/cm², bracketed back to the previous crossover.
The verdict takes the earlier of the two rules ("both" when the OCV trip
falls inside the crossover bracket); a sample neither rule kills is alive at
the hours tested. DOE AST targets are reported beside it, never judged.

RH cycles count in holds. A .fcd hold logs the humidifiers' Wet flags, and
every fall of one from wet to dry ends a cycle. A hold that logs none -- every
station CSV -- has its cycles read off its OCV, which wet gas lowers: where
the OCV steps at a steady pace between a low (wet) level and a high (dry) one,
every rise from the one to the other ends a cycle. An FCTS 40 h folder cycles
RH so in its two 0% RH steps, about 490 times in each. The OCV is read so
only off a log of a reading every 5 s or finer. Where it cannot be read
(its log too coarse or too sparse, or its rises no steady cycle), or swings
so in under half of a hold whose name gives a humidified RH, the hold's
cycles are not counted, and a note says why.

Only OCV hold time counts: hours tested, hours to death, RH cycles and the
x axis of every plot, the timeline's included, leave out characterizations
and every other step between holds. The portal runs this as "FC Durability".
"""

import contextlib
import io
import math
import os
import re
from dataclasses import dataclass, field, replace as dc_replace
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
# The OCV plot's high–low band is labelled by its edges as the portal's redraw
# and Comparisons know them, and shaded alike.
from scripts.helpers.plot_compare import save_with_sidecar, BAND_LOW, BAND_HIGH, BAND_ALPHA


# ═══════════════════════════════════════════════════════════════════
#  Defaults (each one is also a run() parameter)
# ═══════════════════════════════════════════════════════════════════

# An air/O2 OCV step running at least this long is a hold (plan decision 9).
# Characterization OCV steps under air last 30 s to 1 h; holds last 10 h.
HOLD_MIN_H = 2.0

# Every station step stops logging this long before it ends: a 1 h step's
# last stamp is 3,570 s and a 10 h step's 35,970 s, on both stands. Adding it
# back makes the clock agree with the hand-built Gore-A workbook, which steps
# 36,000 s per hold file (plan decision 10).
STEP_TAIL_S = 30.0

# A log whose first stamp is later than this did not start with its step,
# so its last stamp is not the step's length.
FIRST_SAMPLE_MAX_S = 60.0

DEFAULT_GEO_AREA = 5.0     # cm²; both stands
DEFAULT_LOADING = 0.40     # mg Pt/cm²; both durability MEAs on hand

OXIDANTS = ('Air', 'O2')


# ═══════════════════════════════════════════════════════════════════
#  File names
# ═══════════════════════════════════════════════════════════════════

# The step: a letter, its number, and an optional letter after it (f3b, a
# step added after f3), as the index reads steps (record._STEP_RE).
_STEP_RE = re.compile(r'^(?P<letter>[A-Za-z])(?P<number>\d+)(?P<trail>[A-Za-z]?)$')
_NUM = r'\d+(?:\.\d+)?'
_TRH_RE = re.compile(rf'^(?P<T>{_NUM})Cell-(?P<RH>{_NUM})RH$', re.IGNORECASE)
# 'Air' is tried before 'A' so 0.4WAir does not read as A plus a stray 'ir'.
# The B<n> tail (0.5DN2B3) carries no information and is dropped.
_FLOW_RE = re.compile(
    rf'^(?P<fa>{_NUM})(?P<wa>[WD]?)(?P<ga>H2|N2|O2|Air|A)'
    rf'-(?P<fc>{_NUM})(?P<wc>[WD]?)(?P<gc>H2|N2|O2|Air|A)(?:B\d+)?$',
    re.IGNORECASE)
_KINDS = (
    ('ocv', re.compile(r'^OCV$', re.IGNORECASE)),
    ('crossover', re.compile(r'^H2X$', re.IGNORECASE)),
    ('ecsa', re.compile(r'^ECSA$', re.IGNORECASE)),
    ('eis', re.compile(r'^G?EIS$', re.IGNORECASE)),
    ('polcurve', re.compile(r'^PolarizationCurve$', re.IGNORECASE)),
    ('eclean', re.compile(r'^EClean\d*x?$', re.IGNORECASE)),
    ('breakin', re.compile(r'^Break-?in$', re.IGNORECASE)),
)
_GAS = {'h2': 'H2', 'n2': 'N2', 'o2': 'O2', 'air': 'Air', 'a': 'Air'}
_HUMID = {'W': 'wet', 'D': 'dry', '': ''}


def parse_step_name(filename: str) -> Optional[Dict[str, Any]]:
    """Fields encoded in a station step-file name, or None if it is not one.

    The sample may contain underscores (01M12umKolon_2), so the name is read
    by shape rather than by position: the step token comes first, the
    temperature/RH token is found by its form searching from the right, the
    kind sits just before it, and the gas flows just after it. Anything after
    the flows is kept as `suffix` (FILTERDATA, IVCurve, ...).
    """
    stem = re.sub(r'\.csv$', '', os.path.basename(filename), flags=re.IGNORECASE)
    tokens = stem.split('_')
    if len(tokens) < 4:
        return None
    m = _STEP_RE.match(tokens[0])
    if not m:
        return None
    trh_i = next((i for i in range(len(tokens) - 1, 2, -1)
                  if _TRH_RE.match(tokens[i])), None)
    if trh_i is None:
        return None
    trh = _TRH_RE.match(tokens[trh_i])
    kind_token = tokens[trh_i - 1]
    flows = _FLOW_RE.match(tokens[trh_i + 1]) if trh_i + 1 < len(tokens) else None
    rest = tokens[trh_i + 2:] if flows else tokens[trh_i + 1:]
    rec: Dict[str, Any] = {
        'step': f"{m['letter'].lower()}{int(m['number'])}{m['trail'].lower()}",
        'letter': m['letter'].lower(),
        'number': int(m['number']),
        'trail': m['trail'].lower(),
        'sample': '_'.join(tokens[1:trh_i - 1]),
        'kind': next((k for k, rx in _KINDS if rx.match(kind_token)), 'other'),
        'kind_token': kind_token,
        'T_C': float(trh['T']),
        'RH_pct': float(trh['RH']),
        'anode_gas': None, 'anode_slpm': None, 'anode_humid': '',
        'cathode_gas': None, 'cathode_slpm': None, 'cathode_humid': '',
        'suffix': '_'.join(rest),
    }
    if flows:
        rec.update(
            anode_gas=_GAS[flows['ga'].lower()],
            anode_slpm=float(flows['fa']),
            anode_humid=_HUMID[flows['wa'].upper()],
            cathode_gas=_GAS[flows['gc'].lower()],
            cathode_slpm=float(flows['fc']),
            cathode_humid=_HUMID[flows['wc'].upper()],
        )
    return rec


def natural_key(text: str) -> List[Any]:
    """Sort key that orders 40H_2 before 40H_10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', text)]


# ═══════════════════════════════════════════════════════════════════
#  Scribner FuelCell (.fcd) step files
# ═══════════════════════════════════════════════════════════════════

# As the KM8-S6 run (October 2026) names them:
#   b20_CV-2mVs_0o12to0o8V_80C-100RH_0o4H2-0o4N2_0kPa.fcd
# The block is one or two letters (the shut-down steps are zz1, zz2), and a
# letter may follow the number (f3b).
_FCD_STEP_RE = re.compile(r'^(?P<letter>[A-Za-z]{1,2})(?P<number>\d+)(?P<trail>[A-Za-z]?)$')
_ONUM = r'\d+(?:[o.]\d+)?'
# 80C-100RH; 90C-100RH(90C)-0RH, RH cycled between the two; 93C-(90C)RH, a
# dew point; 25C-Dry; or the temperature alone (a0_T-cell-ramp-up_83C_0kPa).
_FCD_TRH_RE = re.compile(
    rf'^(?P<T>{_ONUM})C(?:-(?:(?P<RH>{_ONUM})RH(?:\((?P<dp_wet>{_ONUM})C\)-(?P<RH2>{_ONUM})RH)?'
    rf'|\((?P<dp>{_ONUM})C\)RH|(?P<dry>Dry)))?$', re.IGNORECASE)
_FCD_FLOW_RE = re.compile(
    rf'^(?P<fa>{_ONUM})(?P<ga>H2|N2|O2|Air)-(?P<fc>{_ONUM})(?P<gc>H2|N2|O2|Air)$', re.IGNORECASE)
_FCD_P_RE = re.compile(rf'^(?P<P>{_ONUM})kPa$', re.IGNORECASE)
_FCD_CV_RE = re.compile(rf'^CV-(?P<rate>{_ONUM})mVs(?:-.*)?$', re.IGNORECASE)
# CVs are named by scan rate: 2 mV/s crossover, 50 mV/s ECSA, 500 mV/s cleaning.
FCD_CV_XOVER_MAX_MVS = 10.0
FCD_CV_ECSA_MAX_MVS = 200.0
# The step's own end condition, as the header's Description gives it:
# 'Open Circuit; 30 Seconds; Fixed 1 Sec/Pt; Term:E<0.7 NextStep; ...'. A
# hold the station ends on it is a hold however short, and an OCV death; so
# is a wet half of an RH-cycling loop it cuts short (KM8-S6's limit is on the
# loop's 30 s wet sub-step), the loop running on (station_runs).
_FCD_TERM_RE = re.compile(r'Term:\s*E\s*<\s*(?P<v>\d+(?:\.\d+)?)', re.IGNORECASE)
# How long that (sub-)step runs, the Description's second field: '30 Seconds'.
_FCD_DURATION_RE = re.compile(r'(?:^|;)\s*(?P<n>\d+(?:\.\d+)?)\s*(?P<u>second|minute|hour)s?\s*(?=;|$)',
                              re.IGNORECASE)
_FCD_UNIT_S = {'second': 1.0, 'minute': 60.0, 'hour': 3600.0}
# A step named for a change of gas ('N2-to-Air', KM8-S6's c3 and d3).
_GAS_CHANGE_RE = re.compile(r'(?:^|[_-])[A-Za-z0-9]+-to-[A-Za-z0-9]+(?:$|[_-])', re.IGNORECASE)
# Rows Full Analysis's Scribner presets leave out of crossover and pol-curve
# files (Ctrl_Mode, mode_exclude).
FCD_SKIP_MODES = (5,)
FCD_HEADER_MAX_LINES = 500
FCD_TIME = 'Time (Sec)'
FCD_MODE = 'Ctrl_Mode'
# Where each station CSV column comes from in a .fcd file.
_FCD_SOURCE = {
    'Time Stamp': FCD_TIME, 's': FCD_TIME,
    'Working Electrode Voltage': 'E_Stack (V)', 'Current': 'I (A)',
    'mV': 'E_Stack (V)', 'mA': 'I (A)',
    'Frequency': 'Z_Freq (Hz)', 'Real Impedance': 'Z_Real (Ohm)',
    'Imaginary Impedance': 'Z_Imag (Ohm)',
}


def _onum(text: str) -> float:
    return float(text.replace('o', '.').replace('O', '.'))


def _rh_from_dew_point(T_C: float, dew_C: float) -> float:
    """RH (%) of gas at T_C with the given dew point (Magnus, over water)."""
    def ps(t: float) -> float:
        return 6.112 * math.exp(17.62 * t / (243.12 + t))
    return 100.0 * ps(dew_C) / ps(T_C)


def _fcd_kind(token: str) -> str:
    """The kind a .fcd step token names: a CV by its scan rate, the rest by
    the token's first word (GEIS-150mAcm2 is EIS, IV-H2Air and IV-OCV-0o3V
    pol curves, OCV-RHC-1000cyc and Purge OCV-type steps). Only a token whose
    first word names nothing is an OCV-type step for 'OCV' among its later
    words (RHC-OCV-1000cyc)."""
    m = _FCD_CV_RE.match(token)
    if m:
        rate = _onum(m['rate'])
        return ('crossover' if rate <= FCD_CV_XOVER_MAX_MVS
                else 'ecsa' if rate <= FCD_CV_ECSA_MAX_MVS else 'eclean')
    words = token.lower().split('-')
    first = words[0]
    if first.startswith(('ocv', 'purge')):
        return 'ocv'
    if first.startswith(('geis', 'peis', 'eis')):
        return 'eis'
    if first == 'iv' or first.startswith(('polarization', 'polcurve')):
        return 'polcurve'
    if first.startswith('activation') or token.lower().replace('-', '').startswith('breakin'):
        return 'breakin'
    if 'ocv' in words:
        return 'ocv'
    return 'other'


def parse_fcd_step_name(filename: str) -> Optional[Dict[str, Any]]:
    """Fields encoded in a Scribner .fcd step-file name, or None if it is not one.

    The kind is the token after the step (OCV-RHC-1000cyc, CV-2mVs, IV, Purge,
    ...); the temperature/RH token is found by its form, and the flows and the
    back pressure by theirs, wherever they stand after the kind. Other tokens,
    such as a CV's range (0o12to0o8V) or an IV's (OCV-0o3V), are kept as
    `suffix`.
    """
    stem = re.sub(r'\.fcd$', '', os.path.basename(filename), flags=re.IGNORECASE)
    tokens = stem.split('_')
    m = _FCD_STEP_RE.match(tokens[0])
    if not m or len(tokens) < 2:
        return None
    trh_i = next((i for i in range(1, len(tokens)) if _FCD_TRH_RE.match(tokens[i])), None)
    if trh_i is None:
        return None
    trh = _FCD_TRH_RE.match(tokens[trh_i])
    T = _onum(trh['T'])
    rh_text, humid = '', ''
    if trh['RH'] is not None:
        RH = _onum(trh['RH'])
        if trh['RH2'] is not None:
            rh_text = f"{RH:g}↔{_onum(trh['RH2']):g}% RH cycling"
    elif trh['dp'] is not None:
        RH = round(_rh_from_dew_point(T, _onum(trh['dp'])), 1)
        rh_text = f"dew point {_onum(trh['dp']):g} °C"
    elif trh['dry'] is not None:
        RH, humid = 0.0, 'dry'
    else:
        RH = math.nan
    kind_i = 1 if trh_i > 1 else None
    extra: List[str] = []
    flows = press = None
    for i, t in enumerate(tokens[1:], 1):
        if i in (kind_i, trh_i):
            continue
        if flows is None and _FCD_FLOW_RE.match(t):
            flows = _FCD_FLOW_RE.match(t)
        elif press is None and _FCD_P_RE.match(t):
            press = _FCD_P_RE.match(t)
        else:
            extra.append(t)
    kind_token = tokens[1] if kind_i else ''
    rec: Dict[str, Any] = {
        'step': f"{m['letter'].lower()}{int(m['number'])}{m['trail'].lower()}",
        'letter': m['letter'].lower(),
        'number': int(m['number']),
        'trail': m['trail'].lower(),
        'sample': '',
        'kind': _fcd_kind(kind_token),
        'kind_token': kind_token,
        'T_C': T,
        'RH_pct': RH,
        'anode_gas': None, 'anode_slpm': None, 'anode_humid': humid,
        'cathode_gas': None, 'cathode_slpm': None, 'cathode_humid': humid,
        'suffix': '_'.join(extra),
        'source': 'fcd',
        'rh_text': rh_text,
        'P_kPa': _onum(press['P']) if press else None,
    }
    if flows:
        rec.update(anode_gas=_GAS[flows['ga'].lower()], anode_slpm=_onum(flows['fa']),
                   cathode_gas=_GAS[flows['gc'].lower()], cathode_slpm=_onum(flows['fc']))
    return rec


def is_fcd(path: str) -> bool:
    return str(path).lower().endswith('.fcd')


def fcd_layout(path: str) -> Tuple[Dict[str, str], List[str], int]:
    """A .fcd file's header fields, its column names, and how many lines come
    before its data. The header ends with the column line, then 'End
    Comments'; its other lines are 'Key: value'."""
    head: Dict[str, str] = {}
    prev: Optional[str] = None
    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        for n, line in enumerate(fh):
            if line.strip() == 'End Comments':
                if prev is None:
                    break
                return head, [c.strip() for c in prev.rstrip('\r\n').split('\t')], n + 1
            if prev is not None:
                key, sep, value = prev.partition(':')
                if sep:
                    head.setdefault(key.strip(), value.strip().strip('"'))
            prev = line
            if n >= FCD_HEADER_MAX_LINES:
                break
    raise ValueError("no 'End Comments' line, so not a FuelCell data file")


def read_fcd(path: str, names: Tuple[str, ...]) -> pd.DataFrame:
    """The named columns of a .fcd file, as numbers; names it lacks are left out."""
    _, cols, skip = fcd_layout(path)
    seen: Dict[str, int] = {}
    unique = []
    for c in cols:                        # read_csv refuses repeated names
        k = seen.get(c, 0)
        seen[c] = k + 1
        unique.append(c if k == 0 else f'{c}.{k}')
    want = [c for c in dict.fromkeys(names) if c in unique]
    if not want:
        return pd.DataFrame()
    # index_col=False: a row ending in a tab has one field more than the
    # column line, which read_csv would otherwise take for an index.
    df = pd.read_csv(path, sep='\t', header=None, names=unique, usecols=want, skiprows=skip,
                     index_col=False, encoding='utf-8', encoding_errors='replace',
                     on_bad_lines='skip', dtype=str)
    return df.apply(pd.to_numeric, errors='coerce')


def fcd_elapsed(t: np.ndarray) -> np.ndarray:
    """A .fcd step's own clock. Time (Sec) runs on from the start of the
    station session, so the step is timed from one sampling interval before
    its first sample: the station CSVs' convention, whose first stamp is one
    interval in, and the step's length is its last value.

    FuelCell appends a step run again after a restart to the same file, its
    Time (Sec) starting over; each run after the first is joined on one
    interval after the last sample of the one before, so the step's length
    is the time it logged."""
    t = np.asarray(t, float)
    out = np.full(len(t), np.nan)
    idx = np.flatnonzero(np.isfinite(t))
    if not len(idx):
        return out
    ok = t[idx]
    dt, starts = _fcd_runs(ok)
    bounds = np.r_[0, starts, len(ok)]
    end = 0.0
    for a, b in zip(bounds[:-1], bounds[1:]):
        out[idx[a:b]] = end + dt + ok[a:b] - ok[a]
        end = out[idx[b - 1]]
    return out


def _fcd_runs(ok: np.ndarray) -> Tuple[float, np.ndarray]:
    """The sampling interval of finite Time (Sec) values, and where each run
    after the first starts: wherever the time falls back by more than an
    interval (a stamp a hair out of order is not a restart)."""
    d = np.diff(ok)
    dt = float(np.median(d[d > 0])) if (d > 0).any() else 0.0
    return dt, np.flatnonzero(d < (-dt if dt > 0 else 0.0)) + 1


def fcd_run_index(t: np.ndarray) -> np.ndarray:
    """Each row's run of a .fcd file: 0 in the first, one more after each
    restart (fcd_elapsed); a row with no time is in the run of the row
    before it."""
    t = np.asarray(t, float)
    idx = np.flatnonzero(np.isfinite(t))
    out = np.full(len(t), -1)
    if len(idx):
        run_at = np.zeros(len(idx), dtype=int)
        for start in _fcd_runs(t[idx])[1]:
            run_at[start:] += 1
        out[idx] = run_at
    return np.maximum(np.maximum.accumulate(out), 0) if len(out) else out


def fcd_restarts(t: np.ndarray) -> int:
    """How many times a .fcd file's Time (Sec) starts over: the runs of the
    step appended to it after the first."""
    t = np.asarray(t, float)
    return len(_fcd_runs(t[np.isfinite(t)])[1])


def fcd_columns(path: str, names: Tuple[str, ...],
                skip_modes: Tuple[int, ...] = ()) -> pd.DataFrame:
    """Station CSV columns, by their station names, from a .fcd file: time
    on the step's own clock, E_Stack as the cell voltage, I (A) as the
    current, mV and mA for a pol curve, Z_* for EIS. A .fcd EIS file logs a
    row or two at 0 Hz before its sweep; those rows have no impedance. Rows
    in a control mode of `skip_modes` are blanked. Two names are no station
    column: 'Run', each row's run of the step (fcd_run_index), and 'Wet', the
    Wet flag RH cycles are read from (_rh_flag)."""
    want = tuple(_FCD_SOURCE[n] for n in names if n in _FCD_SOURCE)
    if 'Run' in names or 'Wet' in names:   # each row's run (fcd_run_index), from the time
        want += (FCD_TIME,)
    if 'Wet' in names:                 # the Wet flag RH cycles are read from (_rh_flag)
        want += FCD_WET
    df = read_fcd(path, want + ((FCD_MODE,) if skip_modes else ()))
    out: Dict[str, np.ndarray] = {}
    for n in names:
        src = _FCD_SOURCE.get(n)
        if src is None or src not in df.columns:
            continue
        a = df[src].to_numpy(float)
        if src == FCD_TIME:
            a = fcd_elapsed(a)
        elif n in ('mV', 'mA'):
            a = a * 1000.0
        out[n] = a
    if 'Z_Freq (Hz)' in df.columns:
        off = ~(df['Z_Freq (Hz)'].to_numpy(float) > 0)
        for n in ('Frequency', 'Real Impedance', 'Imaginary Impedance'):
            if n in out:
                out[n] = np.where(off, np.nan, out[n])
    if skip_modes and FCD_MODE in df.columns:
        off = np.isin(df[FCD_MODE].to_numpy(float), skip_modes)
        out = {n: np.where(off, np.nan, a) for n, a in out.items()}
    if 'Run' in names and FCD_TIME in df.columns:
        out['Run'] = fcd_run_index(df[FCD_TIME].to_numpy(float))
    if 'Wet' in names and FCD_TIME in df.columns:
        flag = _rh_flag(df, fcd_elapsed(df[FCD_TIME].to_numpy(float)))[0]
        out['Wet'] = df[flag].to_numpy(float) if flag else np.full(len(df), np.nan)
    return pd.DataFrame(out)


def fcd_time_info(path: str) -> Tuple[int, float, float, int]:
    """Row count and first and last time (s) of a .fcd file on the step's
    own clock, and how many times its Time (Sec) starts over."""
    df = read_fcd(path, (FCD_TIME,))
    if FCD_TIME not in df.columns:
        raise ValueError(f"no '{FCD_TIME}' column")
    raw = df[FCD_TIME].to_numpy(float)
    raw = raw[np.isfinite(raw)]
    if len(raw) == 0:
        return 0, math.nan, math.nan, 0
    t = fcd_elapsed(raw)
    return len(t), float(t[0]), float(t[-1]), fcd_restarts(raw)


def fcd_run_end_v(path: str) -> List[float]:
    """The last voltage of each run a .fcd file logged (its re-runs after
    restarts, fcd_run_index), for the runs that logged one."""
    df = fcd_columns(path, ('Working Electrode Voltage', 'Run'))
    if 'Working Electrode Voltage' not in df.columns or 'Run' not in df.columns:
        return []
    v, run = df['Working Electrode Voltage'].to_numpy(float), df['Run'].to_numpy(int)
    ok = np.isfinite(v)
    return [float(v[ok & (run == r)][-1]) for r in np.unique(run[ok])]


# The station's humidifier flags, 1 wet and 0 dry: under RH cycling they
# switch every half cycle (KM8-S6: 30 s wet, 45 s dry). Every fall from wet
# to dry in a hold is an RH cycle, whatever the hold is named and however
# few: the stations never switch the flag at a steady RH, and if one did,
# that would count as a cycle (Corey, 13 Oct).
FCD_WET = ('Wet_Cathode', 'Wet_Anode')
# The OCV plot bin follows the RH cycle only when the cycles come at a steady
# pace: this share of the gaps between them within this fraction of their
# median. A median of irregular gaps is no cycle to bin by.
RH_PACE_SHARE = 0.8
RH_PACE_TOL = 0.25


def _steady_pace(gaps: np.ndarray) -> bool:
    """Whether gaps between RH cycle ends (s) keep a steady pace."""
    gaps = np.asarray(gaps, float)
    gaps = gaps[gaps > 0]
    if not len(gaps):
        return False
    med = float(np.median(gaps))
    return float(np.mean(np.abs(gaps - med) <= RH_PACE_TOL * med)) >= RH_PACE_SHARE


def _wet_falls(t: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """When a Wet flag falls -- the time of the first dry sample after each
    wet stretch, over the rows with both a time and a flag -- and how long
    each of those wet stretches was flagged wet (s)."""
    ok = np.isfinite(t) & np.isfinite(w)
    tt, wet = t[ok], w[ok] > 0.5
    falls = np.flatnonzero(wet[:-1] & ~wet[1:]) + 1
    rises = np.flatnonzero(np.r_[wet[:1], ~wet[:-1] & wet[1:]])
    began = rises[np.searchsorted(rises, falls, side='right') - 1] if len(falls) else falls
    return tt[falls], tt[falls] - tt[began]


def _rh_flag(df: pd.DataFrame, t: np.ndarray) -> Tuple[Optional[str], np.ndarray, np.ndarray]:
    """The Wet flag RH cycles are read from -- the one that falls most, the
    cathode's on a tie -- its falls and its wet stretches' lengths
    (_wet_falls); (None, none, none) without a flag."""
    name: Optional[str] = None
    ends = wet_s = np.empty(0)
    for col in FCD_WET:
        if col in df.columns:
            e, w = _wet_falls(t, df[col].to_numpy(float))
            if name is None or len(e) > len(ends):
                name, ends, wet_s = col, e, w
    return name, ends, wet_s


def fcd_rh_cycle_ends(path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """When each RH cycle of a .fcd hold ended, on the step's own clock --
    each fall of its Wet flag (_rh_flag), so a cycle counts once its wet half
    is over -- and how long each cycle's wet half was; (None, None) when the
    flag never falls."""
    df = read_fcd(path, (FCD_TIME,) + FCD_WET)
    if FCD_TIME not in df.columns:
        return None, None
    _, ends, wet_s = _rh_flag(df, fcd_elapsed(df[FCD_TIME].to_numpy(float)))
    return (ends, wet_s) if len(ends) else (None, None)


def fcd_wet_read(path: str) -> bool:
    """Whether a .fcd file's Wet flag columns hold a reading: a column can be
    there and every row of it blank."""
    df = read_fcd(path, FCD_WET)
    return any(bool(np.isfinite(df[c].to_numpy(float)).any()) for c in df.columns)


# A hold that logs no Wet flag -- every station CSV, and a .fcd hold whose Wet
# columns are blank -- has its RH cycles read off its OCV, which wet gas
# lowers (Corey, 15 Oct: an FCTS 40 h folder's 0% RH steps are "the 10h
# wet/dry cycling segment"). Under RH cycling the OCV steps between a low
# (wet) level and a high (dry) one at the cycle's pace -- KM8-S6's by about
# 20 mV every 75 s, Kolon's by about 10 mV every 76 s -- and each rise from
# the low level to the high one ends a cycle, as the Wet flag's fall does:
# on KM8-S6, whose holds log the flag, the rises count its 1,000 cycles to
# the cycle, each 10-17 s after the flag falls. A steady hold's OCV wanders
# under a millivolt at no steady pace (Gore-A's 15 hold files and Kolon's 30%
# holds: under 0.6 mV). So a hold cycles RH only where its OCV
#  - swings at one period, from OCV_RH_PERIOD_S[0] to [1] (no humidifier
#    cycles faster): the first peak of its autocorrelation (its drift over
#    OCV_RH_DRIFT_S taken out) standing OCV_RH_PEAK_RISE over the lowest it
#    fell to before, as a swing's does and noise's never, that comes within
#    80% of the highest such peak (the period, not a multiple of it);
#  - steps by at least OCV_RH_SWING_MIN_V where it cycles: from its low
#    level to its high one, the OCV's quartiles over the OCV_RH_LEVELS_CYCLES
#    periods about each reading, its drift taken out of the step (so an OCV
#    falling 5 mV an hour does not lift a 1.6 mV wobble over the minimum);
#  - and rises from low to high at least OCV_RH_MIN_CYCLES times, at a steady
#    pace (_steady_pace) and a period apart, give or take RH_PACE_TOL.
# A reading is high over low + OCV_RH_UP of the swing, low under low +
# OCV_RH_DOWN of it, and as it was in between, and where the swing is under
# the minimum. A level held for less than OCV_RH_MIN_RUN of a period is
# noise, and a rise within OCV_RH_GLITCH of a period of the last a glitch:
# neither ends a cycle (Kolon: a reading or two dropping out of a dry half).
# The glitch rule may put aside up to 1 - OCV_RH_KEPT of the rises (dips out
# of a dry half in half the cycles put aside a third); where it puts aside
# more, the period read is a multiple of the cycle, and the cycles are not
# counted.
# A hold whose OCV has no period, never swings OCV_RH_SWING_MIN_V, or swings
# so in under OCV_RH_PART of it and rises from low to high fewer than
# OCV_RH_MIN_CYCLES times, held RH steady. Any other that fails these rules
# is unreadable: why is noted. Cycles counted where the OCV swings in under
# OCV_RH_PART of the hold are noted, and in a hold whose name gives a
# humidified RH are not counted (read_hold_rh_cycles): a short periodic
# disturbance is not RH cycling. Each level must last a quarter of the
# period for its quartile to sit in it.
# The log. Logged too coarsely, a cycle can show as a slower one that no rule
# can tell from it, its rises at that pace (a 76 s cycle logged every 24 s
# rises every 6 cycles, logged every 60 s every 285 s); a short level can
# fall under a quarter of the readings, where its quartile misses it; and a
# cycle logged in step with the log can show no swing at all. So the OCV
# is read only off a log of at least six readings to the shortest cycle
# read, every OCV_RH_LOG_S or finer (OCV_RH_LOG_TOL letting a stamp drift a
# part in a thousand). A coarser log is not read: neither cycles nor a
# steady RH can be told off it.
OCV_RH_PERIOD_S = (30.0, 1800.0)
OCV_RH_DRIFT_S = 1800.0
OCV_RH_PEAK_RISE = 0.25
OCV_RH_SWING_MIN_V = 0.002
OCV_RH_LEVELS_CYCLES = 3
OCV_RH_UP, OCV_RH_DOWN = 0.5, 0.3
OCV_RH_MIN_CYCLES = 10
OCV_RH_GLITCH = 0.5
OCV_RH_MIN_RUN = 0.1
OCV_RH_KEPT = 0.65
OCV_RH_PART = 0.5
OCV_RH_LOG_S = OCV_RH_PERIOD_S[0] / 6
OCV_RH_LOG_TOL = 1.001
# An OCV that held RH steady, as the notes and the Summary say it.
OCV_RH_STEADY = (f'never swings {OCV_RH_SWING_MIN_V * 1000:g} mV or more at a steady pace for '
                 f'{OCV_RH_MIN_CYCLES} cycles')


def _odd(n: float) -> int:
    """A centred rolling window of about n samples: odd, and at least 3."""
    return max(3, int(round(n)) | 1)


def _ocv_period_s(x: np.ndarray, dt: float) -> Optional[float]:
    """The period (s) at which an OCV read every dt s swings: the first peak
    of its autocorrelation within OCV_RH_PERIOD_S, its drift over
    OCV_RH_DRIFT_S taken out, that stands OCV_RH_PEAK_RISE over the lowest
    the autocorrelation fell to before it -- as a swing's peak does, and the
    wiggles noise leaves near lag 0 do not -- and comes within 80% of the
    highest such peak (the period, not a multiple of it). None without one
    above 0."""
    n = len(x)
    lo, hi = max(2, math.ceil(OCV_RH_PERIOD_S[0] / dt)), min(n - 1, int(OCV_RH_PERIOD_S[1] / dt))
    if hi <= lo + 2:
        return None
    x = x - pd.Series(x).rolling(_odd(OCV_RH_DRIFT_S / dt), center=True,
                                 min_periods=1).median().to_numpy()
    x = x - x.mean()
    f = np.fft.rfft(x, 2 * n)                    # zero-padded: no wrap-around
    ac = np.fft.irfft(f * np.conj(f), 2 * n)[:hi + 1]
    if not ac[0] > 0:
        return None
    ac = ac / ac[0]
    seg, floor = ac[lo:], np.minimum.accumulate(ac)[lo:]
    peaks = np.flatnonzero((seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:])) + 1
    peaks = peaks[seg[peaks] - floor[peaks] >= OCV_RH_PEAK_RISE]
    if not len(peaks) or not seg[peaks].max() > 0:
        return None
    first = peaks[np.argmax(seg[peaks] >= 0.8 * seg[peaks].max())]
    return float((lo + first) * dt)


def _debounce(state: np.ndarray, k: int) -> np.ndarray:
    """A level (1 high, 0 low; NaN until first known) with every run of it
    shorter than k readings given the level before it, but the last, which
    the log may have cut short."""
    out = state.copy()
    known = np.flatnonzero(~np.isnan(out))
    if not len(known) or k <= 1:
        return out
    a0 = int(known[0])
    edges = np.flatnonzero(np.diff(out[a0:]) != 0) + 1 + a0
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < k:
            out[a:b] = out[a - 1]
    return out


def _rises(state: np.ndarray) -> np.ndarray:
    """Where a level (1 high, 0 low) goes from low to high: the first high."""
    return np.flatnonzero((state[:-1] == 0.0) & (state[1:] == 1.0)) + 1


class OcvCycles(NamedTuple):
    """RH cycles read off a hold's OCV (ocv_rh_cycle_ends)."""
    ends: Optional[np.ndarray]     # when each ended (s, the step's own clock); None with none counted
    period_s: Optional[float]      # the period the OCV swings at, where it has one
    swing_v: float                 # its median swing where it swings OCV_RH_SWING_MIN_V or more
    share: float                   # the share of the hold it swings so in
    why: str                       # why none were counted though it may cycle RH; '' otherwise


def ocv_rh_cycle_ends(t: np.ndarray, v: np.ndarray) -> OcvCycles:
    """When each RH cycle of a hold that logs no Wet flag ended, read off its
    OCV (t in s on the step's own clock, v in V): each rise from its low (wet)
    level to its high (dry) one, at the first reading over it, where the OCV
    cycles RH as the OCV_RH_* rules have it. With no period, no swing of
    OCV_RH_SWING_MIN_V, or a swing in under OCV_RH_PART of the hold and
    fewer than OCV_RH_MIN_CYCLES rises, it held RH steady; where it cycles
    but the rules fail, or its log cannot be read, `why` says so.

    The period and the levels are read on a regular grid at the median
    sampling interval, and each reading is judged against its levels there.
    A log coarser than OCV_RH_LOG_S, or with too few readings for the grid
    (over four points of it to a reading), is not read."""
    t, v = np.asarray(t, float), np.asarray(v, float)
    ok = np.isfinite(t) & np.isfinite(v)
    t, v = t[ok], v[ok]
    order = np.argsort(t, kind='stable')
    t, v = t[order], v[order]
    d = np.diff(t)
    if not (d > 0).any():
        return OcvCycles(None, None, 0.0, 0.0, 'it logged no OCV to read')
    dt = float(np.median(d[d > 0]))
    if dt > OCV_RH_LOG_S * OCV_RH_LOG_TOL:
        return OcvCycles(None, None, 0.0, 0.0,
                         f'its OCV is logged only every {dt:g} s, too coarsely to read')
    n_grid = int((t[-1] - t[0]) / dt + 1e-9) + 1
    if n_grid > 4 * len(t):
        return OcvCycles(None, None, 0.0, 0.0,
                         f'its OCV is logged too sparsely to read: {len(t):,} readings over '
                         f'{(t[-1] - t[0]) / 3600.0:.3g} h, mostly {dt:g} s apart')
    grid = t[0] + dt * np.arange(n_grid)
    x = np.interp(grid, t, v)
    P = _ocv_period_s(x, dt)
    if P is None:
        return OcvCycles(None, None, 0.0, 0.0, '')
    w = _odd(OCV_RH_LEVELS_CYCLES * P / dt)
    roll = pd.Series(x).rolling(w, center=True, min_periods=w // 2 + 1)
    low_g = roll.quantile(0.25).to_numpy()
    # The swing between the quartiles with the drift taken out (a rolling
    # mean over OCV_RH_DRIFT_S, or the levels' span if longer), so a fall in
    # the OCV across the span adds nothing to it.
    flat = x - pd.Series(x).rolling(max(w, _odd(OCV_RH_DRIFT_S / dt)), center=True,
                                    min_periods=1).mean().to_numpy()
    roll_f = pd.Series(flat).rolling(w, center=True, min_periods=w // 2 + 1)
    swing_g = roll_f.quantile(0.75).to_numpy() - roll_f.quantile(0.25).to_numpy()
    swings = swing_g >= OCV_RH_SWING_MIN_V              # NaN never does
    share = float(np.mean(swings))
    if not swings.any():
        return OcvCycles(None, P, 0.0, 0.0, '')
    sw = float(np.median(swing_g[swings]))
    # Each reading against its levels: high over low + OCV_RH_UP of the
    # swing, low under low + OCV_RH_DOWN, as before in between and where the
    # swing is under the minimum; a level held for under OCV_RH_MIN_RUN of a
    # period is noise (_debounce), and a rise within OCV_RH_GLITCH of a
    # period of the last a glitch.
    low, swing = np.interp(t, grid, low_g), np.interp(t, grid, swing_g)
    live = swing >= OCV_RH_SWING_MIN_V
    up, down = low + OCV_RH_UP * swing, low + OCV_RH_DOWN * swing
    state = np.full(len(v), np.nan)
    state[live & (v > up)] = 1.0
    state[live & (v < down)] = 0.0
    first = np.flatnonzero(live)
    if len(first) and np.isnan(state[first[0]]):     # starting in between: the nearer
        k = first[0]
        state[k] = float(v[k] > (up[k] + down[k]) / 2)
    state = _debounce(pd.Series(state).ffill().to_numpy(),
                      max(1, int(round(OCV_RH_MIN_RUN * P / dt))))
    rises = t[_rises(state)]
    if len(rises) < OCV_RH_MIN_CYCLES and share < OCV_RH_PART:   # no run of cycles: RH held steady
        return OcvCycles(None, P, sw, share, '')
    ends: List[float] = []
    for at in rises:
        if not ends or at - ends[-1] >= OCV_RH_GLITCH * P:
            ends.append(float(at))
    gaps = np.diff(ends)
    rise = 'rises from its low level to its high one'
    if len(rises) < OCV_RH_MIN_CYCLES:
        why = f'{rise} only {len(rises)} times' if len(rises) else f'never {rise}'
    elif len(ends) < max(OCV_RH_MIN_CYCLES, OCV_RH_KEPT * len(rises)):
        why = f'{rise} {len(rises)} times, only {len(ends)} of them a period or so apart'
    elif not _steady_pace(gaps) or abs(float(np.median(gaps)) - P) > RH_PACE_TOL * P:
        why = f'{rise} {len(ends)} times, at no steady pace a period apart'
    else:
        return OcvCycles(np.array(ends), P, sw, share, '')
    return OcvCycles(None, P, sw, share, f'its OCV swings {sw * 1000:.3g} mV at a {P:.4g} s period, '
                                         f'but {why}')


def fcd_loop(path: str, sf: 'StepFile') -> Tuple[bool, float]:
    """Whether a .fcd OCV step is an RH loop -- named for RH cycling, or its
    Wet flag falling at a steady pace -- and the time its wet halves fell
    short of the header's sub-step (s), where that sub-step is the wet one:
    the halves its own limit tripped in, which a loop counts as a hold by.
    The header's time is taken for the wet half's when it fits inside a
    cycle and the dry halves do not last it (KM8-S6: 30 s, against 45 s dry
    and a 75 s cycle); a header giving the whole step's time is not."""
    ends, wet_s = fcd_rh_cycle_ends(path)
    named = '↔' in sf.rh_text
    if ends is None:
        return named, 0.0
    gaps = np.diff(ends)
    loop = named or (len(ends) >= 3 and _steady_pace(gaps))
    lost = 0.0
    if loop and sf.sub_s and len(ends) > 1:
        dry = gaps - wet_s[1:]
        if (sf.sub_s < float(np.median(gaps))
                and abs(float(np.median(dry)) - sf.sub_s) > 0.1 * sf.sub_s):
            lost = float(np.sum(np.clip(sf.sub_s - wet_s, 0.0, None)))
    return loop, lost


def read_time_bounds(path: str) -> Tuple[int, float, float]:
    """Row count and first and last time stamp (s) of a step file.

    Every station layout keeps elapsed seconds in its first column ('Time
    Stamp', or 's' for polarization curves and break-in), so only that column
    is read: a 10 h hold logged at 1 s costs 36,000 floats, not 144,000. A
    .fcd file's Time (Sec) is read alone too, on the step's own clock.
    """
    if is_fcd(path):
        return fcd_time_info(path)[:3]
    col = pd.read_csv(path, usecols=[0]).iloc[:, 0]
    t = pd.to_numeric(col, errors='coerce').dropna().to_numpy(float)
    if len(t) == 0:
        return 0, math.nan, math.nan
    return len(t), float(t[0]), float(t[-1])


# ═══════════════════════════════════════════════════════════════════
#  Timeline
# ═══════════════════════════════════════════════════════════════════

ROLE_LABELS = {
    'hold': 'OCV hold',
    'air-step': 'Air OCV step',
    'h2n2': 'H2/N2 step',
    'n2n2': 'N2/N2 step',
    'ocv-unread': 'OCV, gases unreadable',
    'crossover': 'H2 crossover',
    'ecsa': 'ECSA',
    'eis': 'EIS',
    'polcurve': 'Polarization curve',
    'eclean': 'Electrode cleaning',
    'breakin': 'Break-in',
    'other': 'Other',
}


@dataclass
class StepFile:
    path: str
    folder: str              # relative to the input root, '/'-separated; '.' at the root
    name: str
    step: str
    letter: str
    number: int
    trail: str               # a letter after the number ('b' in f3b), '' for none
    sample: str
    kind: str
    kind_token: str
    T_C: float
    RH_pct: float
    anode_gas: Optional[str]
    anode_slpm: Optional[float]
    anode_humid: str
    cathode_gas: Optional[str]
    cathode_slpm: Optional[float]
    cathode_humid: str
    suffix: str
    n_rows: int = 0
    t_first_s: float = math.nan
    t_last_s: float = math.nan
    role: str = ''
    note: str = ''
    length_s: float = 0.0        # how long the step ran: last stamp + step tail
    seq: int = 0                 # position in the ordered run, from 1
    segment: Optional[int] = None  # hold segment number, from 1 (holds only)
    interval: int = 0            # hold segments completed before this file
    H_start_h: float = 0.0       # hold hours completed before this file
    H_end_h: float = 0.0         # hold hours completed after it
    clock_s: float = 0.0         # seconds this file adds to the hold clock
    source: str = 'csv'          # 'csv' (station CSV) or 'fcd' (Scribner FuelCell)
    rh_text: str = ''            # RH that one number does not say: cycling, a dew point
    P_kPa: Optional[float] = None    # back pressure, where the name gives it (.fcd)
    desc: str = ''               # a .fcd header's step description
    area_cm2: Optional[float] = None  # a .fcd header's SurfaceArea
    term_v: Optional[float] = None    # a .fcd step's own end-on-voltage limit (Term:E<...)
    sub_s: Optional[float] = None     # how long the header's (sub-)step runs: KM8-S6's '30 Seconds'
    # The last voltage of each of its runs (re-runs appended after restarts),
    # read where that limit could end a hold.
    run_end_v: Optional[List[float]] = None
    restarts: int = 0            # times a .fcd file's Time (Sec) starts over (appended re-runs)
    # A .fcd file logging a Wet flag (FCD_WET), which RH cycles are read from:
    # a column for one, and for a hold, a reading in it (fcd_wet_read).
    wet_logged: bool = False
    # A hold's RH cycles: when each ended (s, on the step's own clock), None
    # at a steady RH; and the cycles completed before and after it. A .fcd
    # hold's are its Wet flag's falls; a hold that logs no flag has them read
    # off its OCV (ocv_rh_cycle_ends).
    rh_ends_s: Optional[np.ndarray] = field(default=None, repr=False)
    rh_wet_s: Optional[np.ndarray] = field(default=None, repr=False)   # each cycle's wet half (s)
    # What its RH cycles were counted from, for the manifest: 'Wet flag', or
    # the OCV with its swing and period; '' with none counted.
    rh_from: str = ''
    rh_unread: bool = False      # its RH cycles could not be read or counted (a note says why)
    # An RH loop (read with run_end_v): named for RH cycling, or its Wet flag
    # cycling at a steady pace. Its limit is on one half of the loop, so the
    # station never ends the step on it.
    loop: bool = False
    # The time an RH loop's wet halves fell short of the header's sub-step
    # (s): the halves its limit tripped in. A loop counts as a hold on the
    # length it was programmed to run.
    lost_s: float = 0.0
    C_start: int = 0
    C_end: int = 0

    @property
    def gas_change(self) -> bool:
        """A step named for a change of gas ('N2-to-Air'): its OCV is still
        coming up, so under its limit it has not failed."""
        return bool(_GAS_CHANGE_RE.search(f'{self.kind_token}_{self.suffix}'))

    @property
    def stop_v(self) -> Optional[float]:
        """Where the station ended a run of this step on its own voltage
        limit: the run's last voltage, under the Term:E limit. None when no
        run ended that way."""
        if self.term_v is None:
            return None
        return next((v for v in self.run_end_v or [] if v < self.term_v), None)

    @property
    def cut_short(self) -> bool:
        """A .fcd OCV step the station ended on its own voltage limit (a run
        whose last sample is under the Term:E limit: the station moves on as
        soon as the OCV falls under it), unless it is a gas change, or an RH
        loop (StepFile.loop), which the limit never ends: it trips in one
        half and the loop runs on."""
        return self.stop_v is not None and not self.gas_change and not self.loop

    @property
    def rh_cycles(self) -> int:
        return 0 if self.rh_ends_s is None else len(self.rh_ends_s)

    @property
    def logged_h(self) -> float:
        if self.n_rows < 2:
            return 0.0
        return (self.t_last_s - self.t_first_s) / 3600.0

    @property
    def conditions_key(self) -> Tuple[Any, ...]:
        return (self.T_C, self.RH_pct, self.anode_gas, self.anode_slpm, self.anode_humid,
                self.cathode_gas, self.cathode_slpm, self.cathode_humid, self.rh_text,
                self.P_kPa)

    @property
    def rh_label(self) -> str:
        if self.rh_text:
            return self.rh_text
        return f'{self.RH_pct:g}% RH' if math.isfinite(self.RH_pct) else 'RH not given'

    @property
    def gases(self) -> str:
        """'H2/Air 0.1/0.2 slpm', with 'dry' marked: the stations humidify
        by default, so a dry stream is the exception worth reading."""
        if not self.cathode_gas:
            return ''
        anode = ('dry ' if self.anode_humid == 'dry' else '') + str(self.anode_gas)
        cathode = ('dry ' if self.cathode_humid == 'dry' else '') + str(self.cathode_gas)
        return f'{anode}/{cathode} {self.anode_slpm:g}/{self.cathode_slpm:g} slpm'


@dataclass
class HoldSegment:
    number: int
    folder: str
    files: List[StepFile] = field(default_factory=list)
    H_start_h: float = 0.0
    H_end_h: float = 0.0

    @property
    def conditions(self) -> str:
        """The hold's conditions as its name gives them; the back pressure
        too where the name has one (.fcd), since it tells conditions apart."""
        f = self.files[0]
        text = f'{f.T_C:g} °C, {f.rh_label}, {f.gases}'
        return text if f.P_kPa is None else f'{text}, {f.P_kPa:g} kPa'


@dataclass
class Timeline:
    root: str
    files: List[StepFile]
    folders: List[str]
    segments: List[HoldSegment]
    ignored: Dict[str, int]
    skipped: List[Tuple[str, str]]
    notes: List[str]
    hold_min_h: float
    step_tail_s: float
    # Folders skipped in the upload's folder list: never uploaded, so never
    # read, and named here so the outputs say what the analysis did not see.
    left_out: List[str] = field(default_factory=list)

    @property
    def hours_tested(self) -> float:
        return self.segments[-1].H_end_h if self.segments else 0.0

    @property
    def holds(self) -> List[StepFile]:
        return [f for f in self.files if f.role == 'hold']

    @property
    def cycles_tested(self) -> int:
        """RH cycles completed in the holds, from their Wet flags or, where
        they log none, their OCV: 0 for a test at steady RH."""
        return sum(f.rh_cycles for f in self.holds)

    def _rh_gaps(self) -> np.ndarray:
        """Times (s) between RH cycle ends within each hold. In holds named
        for RH cycling, only cycles whose wet half ran its programmed time
        count: a half the station's limit tripped in comes short
        (station_runs), and so does its cycle. Programmed is within 10% of
        the 90th percentile of those holds' wet halves."""
        cyc = [f for f in self.holds if f.rh_ends_s is not None and len(f.rh_ends_s) > 1]
        loops = [f for f in cyc if '↔' in f.rh_text and f.rh_wet_s is not None]
        gaps = [np.diff(f.rh_ends_s) for f in cyc if f not in loops]
        if loops:
            wet = np.concatenate([f.rh_wet_s[1:] for f in loops])
            whole = wet >= 0.9 * np.percentile(wet, 90)
            gaps.append(np.concatenate([np.diff(f.rh_ends_s) for f in loops])[whole])
        gaps = np.concatenate(gaps) if gaps else np.empty(0)
        return gaps[gaps > 0]

    @property
    def rh_period_s(self) -> Optional[float]:
        """The RH cycle (s): the median time between cycle ends within any
        hold, or None when no hold has two."""
        gaps = self._rh_gaps()
        return float(np.median(gaps)) if len(gaps) else None

    @property
    def rh_steady(self) -> bool:
        """Whether there is an RH cycle for the OCV plot bin to follow: the
        cycles keep a steady pace (_steady_pace), those a hold named for RH
        cycling ran in full (_rh_gaps: cycles its limit tripped in come
        short)."""
        return _steady_pace(self._rh_gaps())

    def cycles_at(self, H_h: float) -> int:
        """RH cycles completed by hold hour H_h: within a hold, those that
        ended by then; between holds, those of every hold before."""
        for f in self.holds:
            if f.H_start_h <= H_h <= f.H_end_h:
                if f.rh_ends_s is None:
                    return f.C_start
                done = int(np.searchsorted(f.rh_ends_s, (H_h - f.H_start_h) * 3600.0 + 1e-6,
                                           side='right'))
                return f.C_start + done
        before = [f for f in self.holds if f.H_end_h <= H_h]
        return before[-1].C_end if before else 0


def step_length_s(sf: StepFile, step_tail_s: float = STEP_TAIL_S) -> float:
    """How long a step ran: its last stamp plus the tail the log misses.

    A log that starts late did not start with its step, so only its own span
    is known; the late start is the caller's to note. A .fcd step is on its
    own clock and ends with its last sample, so its length is that sample's
    time.
    """
    if sf.source == 'fcd':
        return sf.t_last_s
    span = sf.t_last_s if sf.t_first_s <= FIRST_SAMPLE_MAX_S else sf.t_last_s - sf.t_first_s
    return span + step_tail_s


def log_zero_s(sf: StepFile) -> float:
    """The stamp a step file's readings are timed from on the hold clock: a
    station CSV log that started late is placed by its own span, from its
    first stamp; any other log from 0."""
    return sf.t_first_s if sf.source != 'fcd' and sf.t_first_s > FIRST_SAMPLE_MAX_S else 0.0


def classify(sf: StepFile, hold_min_h: float = HOLD_MIN_H) -> str:
    """The file's role in the protocol, from its kind, gases and length.

    Length is the step's run length (`length_s`), the same quantity the hold
    clock adds, so a hold programmed for exactly `hold_min_h` qualifies
    whatever its logging interval. A .fcd OCV step that its own voltage
    limit ended (`cut_short`) is a hold however short: a hold whose OCV
    fails ends that way, and that is an OCV death (station_runs), at hours
    the clock has to count. An RH loop is never ended by its limit, which
    trips in its wet halves instead, each falling short of its time: the
    loop is a hold by the time it was programmed to run (`lost_s` added).
    """
    if sf.kind != 'ocv':
        return sf.kind
    if sf.cathode_gas is None:
        return 'ocv-unread'
    if sf.cathode_gas in OXIDANTS:
        held = (sf.length_s + sf.lost_s) / 3600.0 >= hold_min_h or sf.cut_short
        return 'hold' if held else 'air-step'
    return 'n2n2' if sf.anode_gas == 'N2' else 'h2n2'


def order_folders(first_letter: Dict[str, str],
                  folder_order: Optional[List[str]] = None
                  ) -> Tuple[List[str], List[str]]:
    """Folder order and any notes about it.

    Canonical order is each folder's earliest step letter, then natural sort
    of its path: an FCTS run's BOL folder (steps a, b) leads its 40 h folders
    (steps from c) whatever it is called, and 40H_2 precedes 40H_10 whether
    or not the step letters restart in each folder.

    An explicit `folder_order` (the upload UI sends 'parent/sub' paths)
    overrides it. Entries must match a folder path exactly: substring
    matching lets 'hold1' claim 'A_hold1'. Folders it leaves out follow in
    canonical order.
    """
    canonical = sorted(first_letter, key=lambda d: (first_letter[d], natural_key(d)))
    if not folder_order:
        return canonical, []
    notes: List[str] = []
    chosen: List[str] = []
    for entry in folder_order:
        e = str(entry).strip().strip('/')
        if e in first_letter and e not in chosen:
            chosen.append(e)
        else:
            notes.append(f'folder_order entry "{entry}" matches no folder exactly and was ignored')
    rest = [d for d in canonical if d not in chosen]
    if rest:
        notes.append('folders missing from folder_order were placed after it: ' + ', '.join(rest))
    return chosen + rest, notes


def _missing_steps(files: List[StepFile]) -> List[str]:
    """'40H_1: c7, c8' for gaps inside each folder's run of step numbers."""
    out = []
    by_block: Dict[Tuple[str, str], List[int]] = {}
    for f in files:
        by_block.setdefault((f.folder, f.letter), []).append(f.number)
    for (folder, letter), nums in by_block.items():
        present = set(nums)
        gaps = [n for n in range(min(nums), max(nums) + 1) if n not in present]
        if gaps:
            shown = ', '.join(f'{letter}{n}' for n in gaps[:12])
            more = f' and {len(gaps) - 12} more' if len(gaps) > 12 else ''
            out.append(f'{folder}: no file for {shown}{more}')
    return out


def _named_humid(f: StepFile) -> bool:
    """Whether a hold's name gives a humidified RH -- a steady RH over 0%, or a
    dew point -- rather than RH cycling, 0% RH or no RH."""
    return '↔' not in f.rh_text and math.isfinite(f.RH_pct) and f.RH_pct > 0


def read_hold_rh_cycles(f: StepFile, notes: List[str]) -> None:
    """A hold file's RH cycles (StepFile.rh_ends_s, rh_from): from its Wet
    flags (.fcd) or, where it logs none, off its OCV (ocv_rh_cycle_ends): its
    readings at open circuit, on the clock they sit on in the hold
    (log_zero_s). Cycles read off the OCV in under OCV_RH_PART of a hold
    whose name gives a humidified RH (_named_humid) are a disturbance, not
    RH cycling, and are not counted: the hold is unreadable. Notes say where
    cycles could not be read or counted; where they were counted in under
    OCV_RH_PART of a hold, or off the OCV of a hold whose name gives a
    humidified RH, or of a .fcd hold whose own voltage limit its Wet flag
    would place; and where a hold named for RH cycling shows no cycle."""
    if f.source == 'fcd':
        try:
            f.rh_ends_s, f.rh_wet_s = fcd_rh_cycle_ends(f.path)
        except Exception as e:                       # noqa: BLE001
            f.wet_logged, f.rh_unread = False, True
            notes.append(f'{f.step}: its Wet flags could not be read ({type(e).__name__}: '
                         f'{e}), so its RH cycles are not counted')
            return
        if f.rh_ends_s is not None:
            f.rh_from = 'Wet flag'
            return
        if f.wet_logged:
            f.wet_logged = fcd_wet_read(f.path)
    if not f.wet_logged:
        # Where its cycles cannot be read, neither can its wet halves, which
        # its own limit (.fcd) may trip in (_limit_hits): a hold named for RH
        # cycling has it judged nowhere, and any other only where a run ends.
        unjudged = ('' if f.term_v is None else
                    ', nor is its own voltage limit judged' if '↔' in f.rh_text else
                    ', and its own voltage limit is judged only where a run ends under it')
        try:
            d, _ = read_ocv_columns(f.path)
            oc = ~(np.abs(d['i'].to_numpy(float)) > CURRENT_TOL_A)   # a blank current too
            r = ocv_rh_cycle_ends(d['t'].to_numpy(float)[oc] - log_zero_s(f),
                                  d['v'].to_numpy(float)[oc])
        except Exception as e:                       # noqa: BLE001
            f.rh_unread = True
            notes.append(f'{f.step}: its OCV could not be read for RH cycles '
                         f'({type(e).__name__}: {e}), so they are not counted{unjudged}')
            return
        if r.why:
            f.rh_unread = True
            notes.append(f'{f.step}: {r.why}, so its RH cycles are not counted{unjudged}')
            return
        if r.ends is not None:
            every = float(np.median(np.diff(r.ends)))
            swing = f'{r.swing_v * 1000:.3g} mV every {every:.4g} s'
            if r.share < OCV_RH_PART and _named_humid(f):
                f.rh_unread = True
                notes.append(f'{f.step}: its name gives {f.rh_label}, and its OCV swings {swing} '
                             f'in only {r.share:.0%} of it, so the {len(r.ends)} cycles it shows there '
                             f'are not counted as RH cycles{unjudged}')
                return
            f.rh_ends_s = r.ends
            f.rh_from = f'OCV, swinging {swing}'
            if r.share < OCV_RH_PART:
                notes.append(f'{f.step}: its OCV swings {OCV_RH_SWING_MIN_V * 1000:g} mV or more in '
                             f'only {r.share:.0%} of it; all {f.rh_cycles} RH cycles counted in it '
                             f'fall there')
            if _named_humid(f):
                notes.append(f'{f.step}: its name gives {f.rh_label}, but its OCV swings {swing}, '
                             f'as under RH cycling: {f.rh_cycles} RH cycles are counted from it')
            if '↔' not in f.rh_text and f.term_v is not None:
                notes.append(f'{f.step}: it logs no Wet flag ({", ".join(FCD_WET)}), so its RH '
                             f'cycles are read off its OCV, and its own voltage limit is judged only '
                             f'where a run ends under it, not in the wet halves it may trip in')
    if '↔' not in f.rh_text:
        return
    if f.rh_ends_s is None:
        notes.append(f'{f.step}: named for RH cycling, but '
                     + ('its Wet flags never switch from wet to dry' if f.wet_logged
                        else f'it logs no Wet flag ({", ".join(FCD_WET)}), and its OCV '
                             f'{OCV_RH_STEADY}')
                     + ', so no RH cycles are counted for it'
                     + (', nor is its own voltage limit judged' if f.term_v is not None else ''))
    elif f.term_v is not None and not f.wet_logged:
        notes.append(f'{f.step}: named for RH cycling, but it logs no Wet flag '
                     f'({", ".join(FCD_WET)}): its RH cycles are read off its OCV, and its own '
                     f'voltage limit is not judged')


def build_timeline(input_dir: str, hold_min_h: float = HOLD_MIN_H,
                   step_tail_s: float = STEP_TAIL_S,
                   folder_order: Optional[List[str]] = None) -> Timeline:
    """Discover, parse, classify, order, segment and clock a durability run."""
    root = Path(input_dir)
    files: List[StepFile] = []
    ignored: Dict[str, int] = {}
    skipped: List[Tuple[str, str]] = []

    for p in sorted(root.rglob('*')):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part == '__MACOSX' for part in rel.parts) or p.name.startswith('._'):
            ignored['macOS metadata'] = ignored.get('macOS metadata', 0) + 1
            continue
        ext = p.suffix.lower()
        if ext not in ('.csv', '.fcd'):
            ext = ext or '(no extension)'
            ignored[ext] = ignored.get(ext, 0) + 1
            continue
        if ext == '.csv' and re.search(r'_FILTERDATA$', p.stem, re.IGNORECASE):
            skipped.append((rel.as_posix(), 'station pol-curve summary (FILTERDATA)'))
            continue
        rec = parse_step_name(p.name) if ext == '.csv' else parse_fcd_step_name(p.name)
        if rec is None:
            skipped.append((rel.as_posix(), 'name does not follow the step-file pattern'
                            + (' for .fcd files' if ext == '.fcd' else '')))
            continue
        sf = StepFile(path=str(p), folder=rel.parent.as_posix(), name=p.name, **rec)
        try:
            if ext == '.fcd':
                head, cols, _ = fcd_layout(str(p))
                sf.wet_logged = any(c in cols for c in FCD_WET)
                sf.desc = head.get('Description', '')
                try:
                    sf.area_cm2 = float(head.get('SurfaceArea', ''))
                except ValueError:
                    sf.area_cm2 = None
                term = _FCD_TERM_RE.search(sf.desc)
                sf.term_v = float(term['v']) if term else None
                runs_for = _FCD_DURATION_RE.search(sf.desc)
                sf.sub_s = (float(runs_for['n']) * _FCD_UNIT_S[runs_for['u'].lower()]
                            if runs_for else None)
                sf.n_rows, sf.t_first_s, sf.t_last_s, sf.restarts = fcd_time_info(str(p))
                if sf.term_v is not None and sf.kind == 'ocv' and sf.cathode_gas in OXIDANTS:
                    sf.run_end_v = fcd_run_end_v(str(p))
                    try:
                        sf.loop, sf.lost_s = fcd_loop(str(p), sf)
                    except Exception:                 # noqa: BLE001  (said when the hold is read)
                        sf.loop, sf.lost_s = '↔' in sf.rh_text, 0.0
            else:
                sf.n_rows, sf.t_first_s, sf.t_last_s = read_time_bounds(str(p))
        except Exception as e:                       # noqa: BLE001
            skipped.append((rel.as_posix(), f'unreadable: {type(e).__name__}: {e}'))
            continue
        if sf.n_rows == 0:
            skipped.append((rel.as_posix(), 'no time stamps'))
            continue
        files.append(sf)

    first_letter: Dict[str, str] = {}
    for f in files:
        first_letter[f.folder] = min(first_letter.get(f.folder, '~'), f.letter)
    folders, notes = order_folders(first_letter, folder_order)
    rank = {d: i for i, d in enumerate(folders)}
    files.sort(key=lambda f: (rank[f.folder], f.letter, f.number, f.trail, f.name))

    seen: Dict[Tuple[str, str], str] = {}
    for f in files:
        key = (f.folder, f.step)
        if key in seen:
            notes.append(f'{f.folder}: step {f.step} appears twice ({seen[key]}, {f.name})')
        else:
            seen[key] = f.name
    notes.extend(_missing_steps(files))

    H = 0.0
    C = 0                          # RH cycles completed, as H counts hold hours
    segments: List[HoldSegment] = []
    current: Optional[HoldSegment] = None
    for i, f in enumerate(files, 1):
        f.seq = i
        f.length_s = step_length_s(f, step_tail_s)
        if f.source != 'fcd' and f.t_first_s > FIRST_SAMPLE_MAX_S:
            f.note = (f'log starts at {f.t_first_s:.0f} s, so only its logged '
                      f'span is counted')
        if f.restarts:
            f.note = (f'Time (Sec) starts over {f.restarts} time(s): runs of the step appended '
                      f'to the file, joined end to end')
            notes.append(f'{f.step}: {f.note}')
        f.role = classify(f, hold_min_h)
        if (f.role == 'hold' and f.lost_s and not f.cut_short
                and f.length_s / 3600.0 < hold_min_h):
            notes.append(f'{f.step}: counted as a hold though it ran {f.length_s / 3600.0:.2f} h, '
                         f'short of the {hold_min_h:g} h hold rule: its wet halves fell '
                         f'{f.lost_s / 60.0:.1f} min short of the header\'s {f.sub_s:g} s, as they do '
                         f'when its own limit (Term:E<{f.term_v:g}) trips in them')
        elif (f.role == 'hold' and f.cut_short and f.cathode_gas in OXIDANTS
                and f.length_s / 3600.0 < hold_min_h):
            notes.append(f'{f.step}: counted as a hold though it ran {f.length_s / 3600.0:.2f} h, '
                         f'short of the {hold_min_h:g} h hold rule: the station ended it on its '
                         f'own limit (Term:E<{f.term_v:g}) at {f.stop_v:.3f} V')
        elif f.role != 'hold' and f.gas_change and f.stop_v is not None:
            notes.append(f'{f.step}: the station ended it on its own limit (Term:E<{f.term_v:g}) '
                         f'at {f.stop_v:.3f} V, but it is named for a gas change, so it is '
                         f'neither a hold nor an OCV death')
        if f.role != 'hold':
            current = None
            f.interval = len(segments)
            f.H_start_h = f.H_end_h = H
            f.C_start = f.C_end = C
            continue
        read_hold_rh_cycles(f, notes)
        f.C_start = C
        C += f.rh_cycles
        f.C_end = C
        f.clock_s = f.length_s
        # A hold continues across file rollovers (Gore-A's 15 x 10 h files),
        # but not across folders -- each FCTS folder is a separate program run
        # with an unknown gap before it -- nor across a change of conditions,
        # which restarts the settling the OCV analysis skips.
        if (current is None or current.folder != f.folder
                or current.files[-1].conditions_key != f.conditions_key):
            current = HoldSegment(number=len(segments) + 1, folder=f.folder, H_start_h=H)
            segments.append(current)
        f.segment = current.number
        f.interval = current.number - 1
        f.H_start_h = H
        H += f.clock_s / 3600.0
        f.H_end_h = H
        current.files.append(f)
        current.H_end_h = H

    if not segments:
        notes.append('no OCV holds found: no OCV step with air or O2 on the cathode '
                     f'runs {hold_min_h:g} h or more')
    return Timeline(root=str(root), files=files, folders=folders, segments=segments,
                    ignored=ignored, skipped=skipped, notes=notes,
                    hold_min_h=hold_min_h, step_tail_s=step_tail_s)


def manifest_rows(tl: Timeline) -> List[Dict[str, Any]]:
    """One row per step file in run order, then the files that were skipped."""
    rows: List[Dict[str, Any]] = []
    for f in tl.files:
        rows.append({
            'Order': f.seq,
            'Folder': f.folder,
            'Step': f.step,
            'Kind': f.kind_token,
            'Role': ROLE_LABELS.get(f.role, f.role),
            'T (°C)': f.T_C,
            'RH (%)': f.rh_text or (f.RH_pct if math.isfinite(f.RH_pct) else ''),
            'Gases': f.gases,
            'Rows': f.n_rows,
            'Logged (h)': round(f.logged_h, 4),
            'Hold segment': f.segment if f.segment else '',
            'Interval': f.interval,
            'Hold hours before (h)': round(f.H_start_h, 4),
            'Hold hours after (h)': round(f.H_end_h, 4),
            'Note': f.note,
            'File': f.name,
        })
        if f.rh_ends_s is not None:
            rows[-1]['RH cycles'] = f.rh_cycles
            rows[-1]['RH cycles from'] = f.rh_from
    for path, reason in tl.skipped:
        folder, _, name = path.rpartition('/')
        rows.append({'Order': '', 'Folder': folder or '.', 'Step': '', 'Kind': '',
                     'Role': 'Skipped', 'T (°C)': '', 'RH (%)': '',
                     'Gases': '', 'Rows': '', 'Logged (h)': '',
                     'Hold segment': '', 'Interval': '', 'Hold hours before (h)': '',
                     'Hold hours after (h)': '', 'Note': reason, 'File': name})
    return rows


def segment_rows(tl: Timeline) -> List[Dict[str, Any]]:
    rows = [{'Segment': s.number,
             'Folder': s.folder,
             'Files': f'{s.files[0].step}–{s.files[-1].step}' if len(s.files) > 1 else s.files[0].step,
             'File count': len(s.files),
             'Conditions': s.conditions,
             'Start (h)': round(s.H_start_h, 4),
             'End (h)': round(s.H_end_h, 4),
             'Length (h)': round(s.H_end_h - s.H_start_h, 4)}
            for s in tl.segments]
    if tl.cycles_tested:
        for row, s in zip(rows, tl.segments):
            row['RH cycles'] = sum(f.rh_cycles for f in s.files)
            row['RH cycles before'] = s.files[0].C_start
    return rows


# ═══════════════════════════════════════════════════════════════════
#  OCV analysis
# ═══════════════════════════════════════════════════════════════════

# OCV death (plan decision 4): the hold OCV's 10-min rolling median stays
# below 0.7 V for 30 min, and death is the start of that stretch. The median
# is centred, so a trip is dated when the OCV fell rather than half a window
# later.
OCV_THRESHOLD_V = 0.7
OCV_WINDOW_MIN = 10.0
OCV_PERSIST_MIN = 30.0
# Not judged at the start of each hold, where the OCV overshoots after the
# H2/N2 step and settles (40H_1 c3: 0.947 V at 1 s, 0.930 V at 10 min).
SETTLE_MIN = 10.0
# Holds log exactly 0 A. A sample drawing more than this is not at open
# circuit -- a load pulse or a station fault -- and is not judged.
CURRENT_TOL_A = 1e-3
# Settled start and end OCV: medians of the first and last 30 min judged.
STAT_SPAN_MIN = 30.0
# Rolling decay rate (plan decision 14): a straight line through the 1-min
# medians of a 10 h window. Windows end every hour of a hold and at its end,
# never reach back past its start, and need judged data in 80% of their
# minutes.
RATE_WINDOW_H = 10.0
RATE_STEP_H = 1.0
RATE_MIN_COVERAGE = 0.8
TRACE_BIN_S = 60.0       # resolution of the OCV trace sheet and the rate fits
# The OCV plot's bins: never finer than 1 min, nor so fine that the test
# takes more than about this many (the plot is also the copy the portal
# stores and redraws). The automatic bin is the finest that allows, in whole
# RH cycles when the holds cycle RH at a steady pace, so every bin holds as
# much wet as dry time and the median does not beat against the cycle
# (Corey, 13 Oct: match RH cycles if possible, else the bin limit). A
# width set on the form (ocv_plot_bin_min) is used as set, within the same
# limits. Each hold starts a bin of its own, so a test of n holds can have up
# to n more. At the cap Gore-A's 150 h hold is 3-min bins, 6,000 points (median
# and band edges), and its stored record 52 KB of the 900 KB a push may send.
PLOT_BINS_CAP = 3000
# Whole RH cycles only while a cycle is no longer than this many of the
# finest bins; a slower cycle is cut into equal bins at least that fine, so
# the plot follows the cycle, rather than averaging it into bins hours wide or
# beating against it.
RH_BIN_MAX_FINEST = 2.0


@dataclass
class OcvSettings:
    """The OCV rule's parameters; run() takes each from its params."""
    threshold_v: float = OCV_THRESHOLD_V
    window_min: float = OCV_WINDOW_MIN
    persist_min: float = OCV_PERSIST_MIN
    settle_min: float = SETTLE_MIN
    rate_window_h: float = RATE_WINDOW_H
    current_tol_a: float = CURRENT_TOL_A
    plot_bin_min: float = 0.0          # OCV plot bin width; 0 is automatic

    def describe(self) -> str:
        return (f'{self.window_min:g}-min centred rolling median below {self.threshold_v:g} V '
                f'for {self.persist_min:g} min; the first {self.settle_min:g} min of each hold '
                f'and samples above {self.current_tol_a * 1000:g} mA are not judged')


_OCV_COLUMNS = ('Time Stamp', 'Working Electrode Voltage', 'Current')


def read_ocv_columns(path: str) -> Tuple[pd.DataFrame, bool]:
    """Time (s), cell voltage (V) and current (A) of an OCV step file, and
    whether it has a current column.

    Columns are found by header name, since Excel-edited copies carry extra
    ones. The cell voltage is the station's 'Working Electrode Voltage', the
    column Corey's Gore-A workbook stitches. A file without a current column
    is read as at open circuit; within a current column, a blank reading
    stays blank, and the sample is not judged. A .fcd file's E_Stack and
    I (A) stand in for the two, and it gives each sample's run (re-runs
    appended after restarts) and Wet flag (the station's limit cuts wet
    halves short, station_runs); a station CSV has neither.
    """
    if is_fcd(path):
        try:
            df = fcd_columns(path, _OCV_COLUMNS + ('Run', 'Wet'))
        except Exception:                             # noqa: BLE001
            # The OCV is not lost to a Wet flag that cannot be read: the
            # file's RH cycles say so (read_hold_rh_cycles), and its halves go
            # unjudged by the station's limit.
            df = fcd_columns(path, _OCV_COLUMNS + ('Run',))
    else:
        df = pd.read_csv(path, usecols=lambda c: str(c).strip() in _OCV_COLUMNS)
        df.columns = [str(c).strip() for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
    for col in _OCV_COLUMNS[:2]:
        if col not in df.columns:
            raise ValueError(f"no '{col}' column")

    def num(col: str) -> np.ndarray:
        return pd.to_numeric(df[col], errors='coerce').to_numpy(float)

    has_current = 'Current' in df.columns
    return (pd.DataFrame({'t': num('Time Stamp'), 'v': num('Working Electrode Voltage'),
                          'i': num('Current') if has_current else np.zeros(len(df)),
                          'run': (df['Run'].to_numpy(int) if 'Run' in df.columns
                                  else np.zeros(len(df), dtype=int)),
                          'wet': num('Wet') if 'Wet' in df.columns else np.full(len(df), np.nan)}),
            has_current)


@dataclass
class HoldTrace:
    """One hold's OCV at full resolution, on the hold clock in seconds."""
    segment: HoldSegment
    t_s: np.ndarray
    v: np.ndarray
    i: np.ndarray
    file_idx: np.ndarray       # index into segment.files
    finite: np.ndarray         # has a time, a voltage and a current reading
    on_load: np.ndarray        # finite, but drawing current
    settling: np.ndarray       # finite, at open circuit, inside the settle window
    valid: np.ndarray          # judged: finite, at open circuit and settled
    median: np.ndarray         # centred rolling median of the judged samples; NaN elsewhere
    unread: List[Tuple[str, str]] = field(default_factory=list)
    no_current: List[str] = field(default_factory=list)   # files read as at open circuit
    run_idx: Optional[np.ndarray] = None   # each sample's run of its file (re-runs of a .fcd)
    wet: Optional[np.ndarray] = None       # each sample's Wet flag (1 wet, 0 dry; NaN unread)


def load_hold_trace(seg: HoldSegment, settings: Optional[OcvSettings] = None) -> HoldTrace:
    """Stitch a hold's files onto the hold clock, mark what is judged, smooth.

    A sample sits at its file's hold hours before plus its own time stamp, so
    each file's log ends one step tail (30 s) before the next file starts, as
    on the clock (a .fcd log has no tail, and its stamps are on the step's own
    clock already); a log that started late is placed by its own span, as the
    clock counts it. Settling is timed from the start of the hold, not of each
    file: Gore-A's rollovers are seamless, each file starting at the previous
    file's last voltage. The rolling median is taken over judged samples only.
    """
    s = settings or OcvSettings()
    ts, vs, cs, ks, rs, ws, unread, no_current = [], [], [], [], [], [], [], []
    for k, f in enumerate(seg.files):
        try:
            d, has_current = read_ocv_columns(f.path)
        except Exception as e:                       # noqa: BLE001
            unread.append((f.name, f'{type(e).__name__}: {e}'))
            continue
        if not has_current:
            no_current.append(f.name)
        ts.append(f.H_start_h * 3600.0 + d['t'].to_numpy() - log_zero_s(f))
        vs.append(d['v'].to_numpy())
        cs.append(d['i'].to_numpy())
        ks.append(np.full(len(d), k))
        rs.append(d['run'].to_numpy())
        ws.append(d['wet'].to_numpy())
    if ts:
        t, v, i, k, r, w = (np.concatenate(a) for a in (ts, vs, cs, ks, rs, ws))
        order = np.argsort(t, kind='stable')
        t, v, i, k, r, w = t[order], v[order], i[order], k[order], r[order], w[order]
    else:
        t = v = i = w = np.empty(0)
        k = r = np.empty(0, dtype=int)
    finite = np.isfinite(t) & np.isfinite(v) & np.isfinite(i)
    on_load = finite & (np.abs(np.nan_to_num(i)) > s.current_tol_a)
    settling = finite & ~on_load & (t < seg.H_start_h * 3600.0 + s.settle_min * 60.0)
    valid = finite & ~on_load & ~settling
    median = np.full(len(t), np.nan)
    if valid.any():
        series = pd.Series(v[valid], index=pd.to_timedelta(t[valid], unit='s'))
        median[valid] = series.rolling(pd.Timedelta(seconds=s.window_min * 60.0),
                                       center=True).median().to_numpy()
    return HoldTrace(seg, t, v, i, k, finite, on_load, settling, valid, median, unread,
                     no_current, run_idx=r, wet=w)


def below_runs(t_s: np.ndarray, median: np.ndarray, threshold_v: float,
               max_gap_s: float) -> List[Tuple[int, int, float]]:
    """(first, last, seconds) of each stretch where the median is below threshold.

    Takes judged samples in time order. A stretch carries across a gap in the
    data no longer than `max_gap_s` (the median window; a file rollover is
    31 s) and ends at a longer one, since nothing is known across it. Each
    sample counts one sampling interval, so 1,800 samples at 1 s last 30 min;
    the interval is taken over distinct time stamps, so duplicated rows do
    not shrink it.
    """
    n = len(t_s)
    if n == 0:
        return []
    below = median < threshold_v
    joined = below[:-1] & below[1:] & (np.diff(t_s) <= max_gap_s)
    starts = np.flatnonzero(below & ~np.r_[False, joined])
    ends = np.flatnonzero(below & ~np.r_[joined, False])
    distinct = np.unique(t_s)
    dt = float(np.median(np.diff(distinct))) if len(distinct) > 1 else 0.0
    return [(int(a), int(b), float(t_s[b] - t_s[a] + dt)) for a, b in zip(starts, ends)]


def rolling_decay(t_h: np.ndarray, v: np.ndarray, H0: float, H1: float,
                  window_h: float = RATE_WINDOW_H, step_h: float = RATE_STEP_H,
                  min_coverage: float = RATE_MIN_COVERAGE,
                  bin_h: float = TRACE_BIN_S / 3600.0) -> List[Dict[str, float]]:
    """Decay-rate windows of one hold, fitted to its 1-min medians.

    Window ends run every `step_h` from H0 + window_h, plus one at H1, and no
    window reaches back past H0, so a hold shorter than the window gets one
    window covering all of it. Decay is the fitted slope's negative in μV/h:
    positive while the OCV falls.
    """
    rows: List[Dict[str, float]] = []
    if not H1 > H0:
        return rows
    ends = [float(e) for e in np.arange(H0 + window_h, H1 + 1e-9, step_h)]
    if not ends or H1 - ends[-1] > 1e-6:
        ends.append(H1)
    for e in ends:
        a = max(H0, e - window_h)
        sel = (t_h >= a) & (t_h < e)
        n = int(sel.sum())
        coverage = n * bin_h / (e - a)
        decay = math.nan
        if n >= 3 and coverage >= min_coverage:
            decay = -float(np.polyfit(t_h[sel], v[sel], 1)[0]) * 1e6
        rows.append({'start': a, 'end': e, 'coverage': coverage, 'decay': decay,
                     'bins': n, 'mean': float(np.mean(v[sel])) if n else math.nan})
    return rows


@dataclass
class OcvRun:
    """A stretch of one hold where the 10-min median is below the threshold."""
    segment: int
    step: str
    start_h: float
    end_h: float
    minutes: float
    lowest_v: float
    recovered_h: Optional[float] = None   # median back at or above the threshold, same hold
    verdict: str = 'dip'                  # 'death', 'dip' or 'after death'
    # 'median': the 10-min median under the threshold; 'station': a .fcd hold
    # whose own voltage limit acted (station_runs), a death however short.
    trigger: str = 'median'
    trips: int = 0           # station: halves of an RH loop its limit tripped in
    stopped: bool = False    # station: a run of a step without RH cycling it ended
    run_ended: bool = False  # station: its last trip was in the last half of a run
    half: str = ''           # station: the halves the limit is on, 'wet' (KM8-S6) or 'dry'
    median_fell: bool = True  # station: the 10-min median fell under the threshold in it

    def station_acted(self, limit: str = '', until: bool = False) -> str:
        """What the station's limit did, for the notes and the events sheet:
        'the station ended the step on its own limit', or 'the station's own
        limit tripped in 40 wet halves, the RH loop running on'. `limit` is
        said after 'limit'; `until` adds the last trip's hour."""
        if not self.trips:
            return f'the station ended the step on its own limit{limit}'
        one = self.trips == 1
        halves = f'a {self.half} half' if one else f'{self.trips} {self.half} halves'
        upto = f' up to {self.end_h:.2f} h' if until and not one else ''
        tail = (', then the station ended the step' if self.stopped
                else (' as the run ended' if one else ', the last as the run ended') if self.run_ended
                else (' and the RH loop ran on' if one else ', the RH loop running on'))
        return f"the station's own limit{limit} tripped in {halves}{upto}{tail}"


@dataclass
class HoldOcv:
    """What one hold contributes: settled values, events, rates, binned traces."""
    segment: HoldSegment
    samples: int = 0
    judged: int = 0
    settling: int = 0
    on_load: int = 0
    not_finite: int = 0
    raw_below: int = 0             # judged samples under threshold while the median was not
    start_v: float = math.nan      # settled start: median of the first 30 min judged
    start_at_h: float = math.nan   # mean time of those samples
    end_v: float = math.nan        # median of the last 30 min judged
    end_at_h: float = math.nan
    mean_v: float = math.nan
    lowest_v: float = math.nan     # lowest 10-min median
    lowest_h: float = math.nan
    first_ok_h: Optional[float] = None   # first judged sample whose median is at or above threshold
    loss_breach_h: Optional[float] = None  # DOE: median below (1 - 20%) of the initial OCV, persisting
    runs: List[OcvRun] = field(default_factory=list)
    rates: List[Dict[str, float]] = field(default_factory=list)
    trace: Optional[pd.DataFrame] = None   # 1-min bins, for the workbook
    plot: Optional[pd.DataFrame] = None    # plot bins of judged samples: t_h, median, min
    files: List[Dict[str, Any]] = field(default_factory=list)
    unread: List[Tuple[str, str]] = field(default_factory=list)
    no_current: List[str] = field(default_factory=list)

    @property
    def conditions_key(self) -> Tuple[Any, ...]:
        return self.segment.files[0].conditions_key

    @property
    def latest_rate(self) -> Optional[Dict[str, float]]:
        """The last window with a rate: the hold's final 10 h when the data allow."""
        return next((r for r in reversed(self.rates) if math.isfinite(r['decay'])), None)


def _r(x: Optional[float], nd: int) -> Optional[float]:
    """Rounded for a sheet; a missing value becomes an empty cell."""
    return round(float(x), nd) if x is not None and math.isfinite(x) else None


def _hold_label(seg: HoldSegment) -> str:
    steps = (seg.files[0].step if len(seg.files) == 1
             else f'{seg.files[0].step}–{seg.files[-1].step}')
    return f'hold {seg.number} ({steps})'


def _trace_bins(tr: HoldTrace, origin_s: float) -> pd.DataFrame:
    """1-min bins of every finite open-circuit sample, settling included: what
    was logged, and how much of it was judged."""
    m = tr.finite & ~tr.on_load
    if not m.any():
        return pd.DataFrame()
    d = pd.DataFrame({'k': np.floor((tr.t_s[m] - origin_s) / TRACE_BIN_S).astype(np.int64),
                      'v': tr.v[m], 'm': tr.median[m], 'j': tr.valid[m], 'f': tr.file_idx[m]})
    g = d.groupby('k', sort=True).agg(n=('v', 'size'), j=('j', 'sum'), f=('f', 'first'),
                                      med=('v', 'median'), lo=('v', 'min'), hi=('v', 'max'),
                                      m10=('m', 'median'))
    seg = tr.segment
    return pd.DataFrame({
        'Hold hours (h)': np.round((origin_s + (g.index.to_numpy() + 0.5) * TRACE_BIN_S) / 3600.0, 4),
        'Segment': seg.number,
        'Step': [seg.files[i].step for i in g['f'].to_numpy()],
        'Samples': g['n'].to_numpy(),
        'Judged': g['j'].to_numpy().astype(int),
        'Median (V)': g['med'].round(5).to_numpy(),
        'Min (V)': g['lo'].round(5).to_numpy(),
        'Max (V)': g['hi'].round(5).to_numpy(),
        '10-min median (V)': g['m10'].round(5).to_numpy(),
    })


def _file_rows(tr: HoldTrace, span_s: float) -> List[Dict[str, Any]]:
    """Per-file statistics of a hold's judged samples."""
    seg = tr.segment
    unread = dict(tr.unread)
    rows: List[Dict[str, Any]] = []
    for k, f in enumerate(seg.files):
        mine = tr.file_idx == k
        sel = mine & tr.valid
        row: Dict[str, Any] = {
            'Segment': seg.number, 'Step': f.step,
            'Start (h)': round(f.H_start_h, 4), 'End (h)': round(f.H_end_h, 4),
            'Samples': int(mine.sum()), 'Judged': int(sel.sum()),
            'Settling': int((mine & tr.settling).sum()),
            'On load': int((mine & tr.on_load).sum()),
            'Not finite': int((mine & ~tr.finite).sum()),
            'First 30 min (V)': None, 'Last 30 min (V)': None, 'Mean (V)': None,
            'Min (V)': None, 'Max (V)': None, 'Lowest 10-min median (V)': None,
            'Note': f'could not be read: {unread[f.name]}' if f.name in unread else '',
            'File': f.name,
        }
        if f.rh_ends_s is not None:
            row['RH cycles'] = f.rh_cycles
        if sel.any():
            t, v, m = tr.t_s[sel], tr.v[sel], tr.median[sel]
            row.update({
                'First 30 min (V)': _r(np.median(v[t < t[0] + span_s]), 5),
                'Last 30 min (V)': _r(np.median(v[t >= t[-1] - span_s]), 5),
                'Mean (V)': _r(v.mean(), 5),
                'Min (V)': _r(v.min(), 5),
                'Max (V)': _r(v.max(), 5),
                'Lowest 10-min median (V)': _r(np.nanmin(m), 5),
            })
        rows.append(row)
    return rows


def analyze_hold(tr: HoldTrace, settings: Optional[OcvSettings] = None,
                 plot_bin_s: float = TRACE_BIN_S, loss_ref_v: Optional[float] = None,
                 loss_pct: Optional[float] = None) -> HoldOcv:
    """Settled values, below-threshold stretches, rates and binned traces of one hold.

    With `loss_pct`, also the first time the median stays below that loss
    from `loss_ref_v` (this hold's own settled start when None) for the
    persistence time: the DOE OCV-loss breach.
    """
    s = settings or OcvSettings()
    seg = tr.segment
    thr = s.threshold_v
    span = STAT_SPAN_MIN * 60.0
    origin = seg.H_start_h * 3600.0
    h = HoldOcv(segment=seg, unread=list(tr.unread), no_current=list(tr.no_current))
    h.samples = len(tr.t_s)
    h.not_finite = int((~tr.finite).sum())
    h.on_load = int(tr.on_load.sum())
    h.settling = int(tr.settling.sum())
    tv, vv, mv, kv = (a[tr.valid] for a in (tr.t_s, tr.v, tr.median, tr.file_idx))
    h.judged = len(tv)
    h.trace = _trace_bins(tr, origin)
    h.files = _file_rows(tr, span)
    if not h.judged:
        return h

    first, last = tv < tv[0] + span, tv >= tv[-1] - span
    h.start_v, h.start_at_h = float(np.median(vv[first])), float(tv[first].mean()) / 3600.0
    h.end_v, h.end_at_h = float(np.median(vv[last])), float(tv[last].mean()) / 3600.0
    h.mean_v = float(vv.mean())
    j = int(np.argmin(mv))
    h.lowest_v, h.lowest_h = float(mv[j]), float(tv[j]) / 3600.0
    h.raw_below = int(((vv < thr) & (mv >= thr)).sum())
    ok = np.flatnonzero(mv >= thr)
    h.first_ok_h = float(tv[ok[0]]) / 3600.0 if len(ok) else None
    for a, b, secs in below_runs(tv, mv, thr, s.window_min * 60.0):
        back = ok[ok > b]
        h.runs.append(OcvRun(segment=seg.number, step=seg.files[kv[a]].step,
                             start_h=float(tv[a]) / 3600.0, end_h=float(tv[b]) / 3600.0,
                             minutes=secs / 60.0, lowest_v=float(mv[a:b + 1].min()),
                             recovered_h=float(tv[back[0]]) / 3600.0 if len(back) else None))
    if loss_pct is not None:
        ref_v = loss_ref_v if loss_ref_v is not None else h.start_v
        floor_v = ref_v * (1.0 - loss_pct / 100.0)
        h.loss_breach_h = next((float(tv[a]) / 3600.0 for a, _, secs
                                in below_runs(tv, mv, floor_v, s.window_min * 60.0)
                                if secs / 60.0 >= s.persist_min - 1e-9), None)

    def binned(width_s: float) -> pd.DataFrame:
        k = np.floor((tv - origin) / width_s).astype(np.int64)
        return (pd.DataFrame({'k': k, 't': tv, 'v': vv}).groupby('k', sort=True)
                .agg(t=('t', 'mean'), median=('v', 'median'), min=('v', 'min'),
                     max=('v', 'max')))

    per_min = binned(TRACE_BIN_S)
    h.rates = rolling_decay(per_min['t'].to_numpy() / 3600.0, per_min['median'].to_numpy(),
                            seg.H_start_h, seg.H_end_h, s.rate_window_h)
    plot = binned(plot_bin_s)
    h.plot = pd.DataFrame({'t_h': plot['t'].to_numpy() / 3600.0,
                           'median': plot['median'].to_numpy(),
                           'min': plot['min'].to_numpy(),
                           'max': plot['max'].to_numpy()})
    return h


@dataclass
class OcvResult:
    """The OCV analysis of a whole test."""
    settings: OcvSettings
    holds: List[HoldOcv] = field(default_factory=list)
    reference: str = ''             # conditions of the first hold; loss and rate follow them
    initial_v: float = math.nan     # settled start of the first reference hold
    final_v: float = math.nan       # end of the last reference hold
    loss_pct: float = math.nan
    decay_uV_h: float = math.nan    # latest rolling rate at the reference conditions
    decay_window_h: Tuple[float, float] = (math.nan, math.nan)
    lowest_v: float = math.nan      # lowest 10-min median of any hold
    lowest_h: float = math.nan
    death: Optional[OcvRun] = None
    recovered_h: Optional[float] = None
    loss_breach_h: Optional[float] = None    # DOE: first persistent 20% loss from initial
    rate_cutoff_h: Optional[float] = None    # the death the headline rate stops at
    rate_note: str = ''
    initial_from: Optional[HoldOcv] = None   # the reference holds the initial and
    final_from: Optional[HoldOcv] = None     # final values came from
    by_condition: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    plot_bin_s: float = math.nan     # bin width of the OCV plot (and its stored copy)
    plot_bin_rule: str = ''          # how that width was chosen

    @property
    def death_h(self) -> Optional[float]:
        return self.death.start_h if self.death else None

    @property
    def judged(self) -> bool:
        """Whether any hold had OCV to judge: without it, no death means
        nothing, rather than a sample that lived."""
        return any(h.judged for h in self.holds)

    @property
    def runs(self) -> List[OcvRun]:
        return [r for h in self.holds for r in h.runs]


def _by_condition(holds: List[HoldOcv], ref_key: Tuple[Any, ...],
                  cutoff_h: float = math.inf) -> List[Dict[str, Any]]:
    """One row per hold condition; rates are the latest window ending by
    `cutoff_h` (the death, if any), as for the headline rate."""
    groups: Dict[Tuple[Any, ...], List[HoldOcv]] = {}
    for h in holds:
        groups.setdefault(h.conditions_key, []).append(h)
    rows = []
    for key, hs in groups.items():
        start = next((h.start_v for h in hs if math.isfinite(h.start_v)), math.nan)
        end = next((h.end_v for h in reversed(hs) if math.isfinite(h.end_v)), math.nan)
        rated = [r for h in hs for r in h.rates
                 if math.isfinite(r['decay']) and r['end'] <= cutoff_h]
        rate = rated[-1] if rated else None
        lows = [h.lowest_v for h in hs if math.isfinite(h.lowest_v)]
        rows.append({
            'Conditions': hs[0].segment.conditions,
            'Reference': 'yes' if key == ref_key else '',
            'Holds': len(hs),
            'Hold hours (h)': round(sum(h.segment.H_end_h - h.segment.H_start_h for h in hs), 4),
            'First settled start (V)': _r(start, 5),
            'Last end (V)': _r(end, 5),
            'Loss (%)': _r((start - end) / start * 100.0, 3) if start else None,
            'Latest decay rate (μV/h)': _r(rate['decay'], 1) if rate else None,
            'Lowest 10-min median (V)': _r(min(lows), 5) if lows else None,
        })
    return rows


def _ocv_notes(res: OcvResult) -> List[str]:
    s = res.settings
    out: List[str] = []
    for h in res.holds:
        lab = _hold_label(h.segment)
        for name, why in h.unread:
            out.append(f'{lab}: {name} could not be read ({why}); its hours count, '
                       f'its OCV is not judged')
        for name in h.no_current:
            out.append(f'{lab}: {name} has no Current column; its samples are taken as '
                       f'at open circuit')
        if h.on_load:
            out.append(f'{lab}: {h.on_load} samples drew more than '
                       f'{s.current_tol_a * 1000:g} mA and were not judged')
        if h.not_finite:
            out.append(f'{lab}: {h.not_finite} samples lack a time, voltage or current '
                       f'reading and were not judged')
        if not h.judged:
            out.append(f'{lab}: no OCV left to judge')
        dips = [r for r in h.runs if r.verdict == 'dip']
        if dips:
            worst = max(dips, key=lambda r: r.minutes)
            out.append(f'{lab}: 10-min median under {s.threshold_v:g} V {len(dips)} time(s), '
                       f'longest {worst.minutes:.1f} min from {worst.start_h:.2f} h; shorter '
                       f'than {s.persist_min:g} min, so not a death')
        if h.raw_below:
            out.append(f'{lab}: {h.raw_below} judged samples under {s.threshold_v:g} V while '
                       f'the 10-min median stayed above it')
    if res.death:
        d = res.death
        seg = res.holds[d.segment - 1].segment
        where = f'{_hold_label(seg)}, step {d.step}'
        if d.trigger == 'station':
            f = next(f for f in seg.files if f.step == d.step)
            acted = d.station_acted(f' (Term:E<{f.term_v:g})', until=True)
            under = (f'{d.minutes * 60.0:.0f} s' if d.minutes < 1.0 else f'{d.minutes:.1f} min')
            out.append(f'OCV death at {d.start_h:.2f} h in {where}: {acted}, the OCV under it from '
                       f'then until it did ({d.minutes:.1f} min, lowest {d.lowest_v:.3f} V)'
                       if not d.trips else
                       f'OCV death at {d.start_h:.2f} h in {where}: {acted}; the OCV was under it '
                       f'{under} in all (lowest reading {d.lowest_v:.4f} V)'
                       + ('' if d.median_fell else
                          f'; the 10-min median never fell under {s.threshold_v:g} V in it'))
        else:
            out.append(f'OCV death at {d.start_h:.2f} h in {where}: 10-min median under '
                       f'{s.threshold_v:g} V for {d.minutes:.0f} min')
        if res.recovered_h is not None:
            thr, at = f'{s.threshold_v:g} V', f'{res.recovered_h:.2f} h'
            out.append('OCV recovered: ' + (
                f'the 10-min median was back at {thr} or more at {at}' if d.trigger != 'station' else
                f"once the station's limit stopped acting, the 10-min median was back at {thr} or "
                f'more at {at}' if d.median_fell else
                f"the 10-min median never fell under {thr} while the station's limit acted, and "
                f'was at {thr} or more after it at {at}')
                + f'. Flagged; the death at {d.start_h:.2f} h stands')
    if res.rate_note:
        out.append(res.rate_note)
    if not res.judged:
        out.append('no hold has OCV to judge, so the OCV death rule could not be applied')
    ref = [h for h in res.holds if h.conditions_key == res.holds[0].conditions_key]
    if res.initial_from is not None and res.initial_from is not ref[0]:
        out.append(f'OCV initial is the settled start of {_hold_label(res.initial_from.segment)}: '
                   f'the first hold at the reference conditions has no judged OCV')
    if res.final_from is not None and res.final_from is not ref[-1]:
        out.append(f'OCV final is the end of {_hold_label(res.final_from.segment)}: the last '
                   f'hold at the reference conditions has no judged OCV')
    if len(res.by_condition) > 1:
        out.append(f"OCV initial, final, loss and decay rate are for the first hold's conditions "
                   f'({res.reference}); every condition is in the "OCV by condition" sheet')
    if res.plot_bin_rule.startswith('raised'):
        out.append(f'OCV plot bin {res.plot_bin_s / 60:.4g} min, {res.plot_bin_rule}')
    return out


def ocv_plot_bin(hours_tested: float, plot_bin_min: float = 0.0,
                 rh_period_s: Optional[float] = None,
                 rh_unsteady: bool = False) -> Tuple[float, str]:
    """The OCV plot's bin width (s), and how it was chosen.

    Automatic (0): the finest bin about PLOT_BINS_CAP bins over the test
    allow (_plot_bin_cap_s), at least 1 min; with a steady RH cycle
    (`rh_period_s`), the fewest whole cycles at least that wide, or equal
    bins to a cycle when it is long. `rh_unsteady`: there are RH cycles, but
    at no steady pace, so the rule says why it does not follow them. Set:
    that width, within the same limits. Bins start at each hold's start, so
    a test of n holds can have up to n more. Only the plot and its stored
    copy follow it; every rule works on the samples.
    """
    total_s = hours_tested * 3600.0
    finest = max(TRACE_BIN_S, _plot_bin_cap_s(total_s))
    if not plot_bin_min > 0:
        limits = f'at least 1 min and no more than about {PLOT_BINS_CAP:,} bins over the test'
        P = rh_period_s
        if P and P > 0:
            # Whole cycles while that stays near the finest width; a longer
            # cycle is cut into equal bins, so the plot follows it without
            # beating against it.
            if P <= RH_BIN_MAX_FINEST * finest + 1e-9:
                k = max(1, math.ceil(finest / P - 1e-9))
                return k * P, f'automatic: {k} whole RH cycle{"s" if k > 1 else ""} of {P:.4g} s, {limits}'
            n = max(1, math.floor(P / finest + 1e-9))
            return P / n, f'automatic: {n} bins to each RH cycle of {P:.4g} s, {limits}'
        return finest, (f'automatic: the finest that keeps about {PLOT_BINS_CAP:,} bins over the '
                        f'test, at least 1 min'
                        + ('; no steady RH cycle to bin by' if rh_unsteady else ''))
    asked_s = plot_bin_min * 60.0
    bin_s = max(TRACE_BIN_S, asked_s, _plot_bin_cap_s(total_s))
    # Widths typed in minutes land a hair off in seconds (2.05 × 60 is
    # 122.99999999999999): only a real raise is called one.
    if bin_s > asked_s * (1 + 1e-9):
        # The width set as typed, every figure of it, so it never prints as
        # the width it was raised to.
        return bin_s, (f'raised from the {plot_bin_min:.10g} min set: bins are at least 1 min, '
                       f'and no more than about {PLOT_BINS_CAP:,} are plotted and stored')
    return bin_s, 'as set'


def _plot_bin_cap_s(total_s: float) -> float:
    """The finest bin (s) about PLOT_BINS_CAP bins over the test allow, rounded
    up at the four significant figures of minutes the readout, workbook and
    notes print: the width they show, typed back on the form, is then taken as
    set, and a raised width never prints as the one asked for."""
    cap_min = total_s / PLOT_BINS_CAP / 60.0
    if not cap_min > 0:
        return 0.0
    digits = 3 - math.floor(math.log10(cap_min))
    return math.ceil(cap_min * 10.0 ** digits - 1e-6) / 10.0 ** digits * 60.0


def analyze_ocv(tl: Timeline, settings: Optional[OcvSettings] = None) -> OcvResult:
    """OCV of every hold, the OCV death verdict, and the decay of the test.

    Holds are read one at a time and only their summaries kept, so a long
    test never has more than one hold's samples in memory.
    """
    s = settings or OcvSettings()
    res = OcvResult(settings=s)
    steady = tl.rh_steady
    plot_bin_s, res.plot_bin_rule = ocv_plot_bin(
        tl.hours_tested, s.plot_bin_min, tl.rh_period_s if steady else None,
        rh_unsteady=bool(tl.cycles_tested) and not steady)
    res.plot_bin_s = plot_bin_s
    ref_key = tl.segments[0].files[0].conditions_key if tl.segments else None
    initial: Optional[float] = None
    for seg in tl.segments:
        is_ref = seg.files[0].conditions_key == ref_key
        tr = load_hold_trace(seg, s)
        h = analyze_hold(tr, s, plot_bin_s, loss_ref_v=initial,
                         loss_pct=DOE_OCV_LOSS_PCT if is_ref else None)
        # In clock order; the station's act first where a median stretch
        # starts with it, so that stretch is after the death, not a dip.
        h.runs = sorted(h.runs + station_runs(tr, s),
                        key=lambda r: (r.start_h, r.trigger != 'station'))
        if is_ref and initial is None and math.isfinite(h.start_v):
            initial = h.start_v
        res.holds.append(h)
    if not res.holds:
        return res
    res.loss_breach_h = next((h.loss_breach_h for h in res.holds
                              if h.loss_breach_h is not None), None)

    # Death is the first stretch, in clock order, that lasts persist_min, or
    # the first event of a hold's own limit (a step the station ended, a wet
    # half it cut short), whichever is earlier.
    # Later stretches are kept and labelled but never move it (plan: a
    # recovery is flagged, not forgiven).
    for r in res.runs:
        if res.death is None and (r.trigger == 'station' or r.minutes >= s.persist_min - 1e-9):
            r.verdict, res.death = 'death', r
        elif res.death is not None:
            r.verdict = 'after death'
    if res.death is not None:
        res.recovered_h = res.death.recovered_h
        if res.recovered_h is None:
            res.recovered_h = next((h.first_ok_h for h in res.holds
                                    if h.segment.number > res.death.segment
                                    and h.first_ok_h is not None), None)

    ref_key = res.holds[0].conditions_key
    ref = [h for h in res.holds if h.conditions_key == ref_key]
    res.reference = res.holds[0].segment.conditions
    res.initial_from = next((h for h in ref if math.isfinite(h.start_v)), None)
    res.final_from = next((h for h in reversed(ref) if math.isfinite(h.end_v)), None)
    if res.initial_from is not None:
        res.initial_v = res.initial_from.start_v
        res.final_v = res.final_from.end_v
    if math.isfinite(res.initial_v) and math.isfinite(res.final_v) and res.initial_v:
        res.loss_pct = (res.initial_v - res.final_v) / res.initial_v * 100.0
    lows = [h for h in res.holds if math.isfinite(h.lowest_v)]
    if lows:
        low = min(lows, key=lambda h: h.lowest_v)
        res.lowest_v, res.lowest_h = low.lowest_v, low.lowest_h
    set_rate_cutoff(res, res.death.start_h if res.death else None, 'OCV')
    res.notes = _ocv_notes(res)
    return res


# A half of an RH loop that ends early without a reading under the limit
# logged counts as the limit tripping in it only with a reading this close
# above the limit at its end: a half cut short while the OCV sits far above
# it is a flag or logging glitch, not the station acting.
TRIP_NEAR_V = 0.02

# (first index of the stretch under the limit, index where the limit acted,
# 'stop', 'trip' or 'trip-end' -- a trip in the last half of its run -- the
# file, and the state of the half: 'wet' or 'dry')
_Hit = Tuple[int, int, str, StepFile, str]


def station_runs(tr: HoldTrace, s: Optional[OcvSettings] = None) -> List[OcvRun]:
    """Where a .fcd hold's own voltage limit (Term:E<0.7 NextStep in the
    step's description) acted, each an OCV death however short (Corey,
    13 Oct). Per run of each file (FuelCell appends a re-run after a
    restart), _limit_hits:

    - a run whose Wet flag does not cycle that ends under the limit: the
      station ended the step;
    - in a run that cycles RH, a half of the sub-step the limit belongs to
      that the limit tripped in, the loop running on (KM8-S6: NextStep from
      the 30 s wet sub-step is the dry one).

    Each is dated from the first reading of the stretch under the limit it
    ends. A trip does not count while the OCV is still coming up: at the
    start of the hold, or of a re-run after a restart, before a reading of
    that run has reached the limit, or within the settle -- unless none ever
    does, a cell dead from the start. A file the hold rolls over to goes on
    as the one before it left off, with no settle.
    Acts less than the median window apart are one event, so a loop
    tripping every cycle is one stretch, not hundreds; an event's minutes
    are the time the OCV was under the limit in it. A step named for a gas
    change is left out (StepFile.gas_change), and so are readings drawing
    current, as for the OCV rule. The first 10-min median at or over the
    threshold after an event is its recovery, flagged as the OCV rule's
    are.
    """
    s = s or OcvSettings()
    n = len(tr.t_s)
    runs = tr.run_idx if tr.run_idx is not None else np.zeros(n, dtype=int)
    wet = tr.wet if tr.wet is not None else np.full(n, np.nan)
    ocv = tr.finite & ~tr.on_load
    settle_s = s.settle_min * 60.0
    hits: List[_Hit] = []
    came_up = False                  # the hold's run before reached the limit
    for k, f in enumerate(tr.segment.files):
        timed = (tr.file_idx == k) & np.isfinite(tr.t_s)
        for j, r_ in enumerate(np.unique(runs[timed])):
            rows = np.flatnonzero(timed & (runs == r_))
            if f.term_v is None or f.gas_change:
                came_up = False
                continue
            at = rows[ocv[rows] & (tr.v[rows] >= f.term_v)]
            if k > 0 and j == 0 and came_up:
                # Rolled over from the file before, the OCV already up.
                settled_s, up_s = -math.inf, -math.inf
            else:
                settled_s = (tr.segment.H_start_h * 3600.0 if k == 0 and j == 0
                             else float(tr.t_s[rows[0]])) + settle_s
                up_s = float(tr.t_s[at[0]]) if len(at) else None
            hits += _limit_hits(tr, f, rows, wet[rows], settled_s, up_s)
            came_up = up_s == -math.inf or len(at) > 0
    hits.sort(key=lambda h: h[1])
    out: List[OcvRun] = []
    group: List[_Hit] = []
    for h in hits:
        if group and tr.t_s[h[0]] - tr.t_s[group[-1][1]] > s.window_min * 60.0:
            out.append(_station_run(tr, group, s))
            group = []
        group.append(h)
    if group:
        out.append(_station_run(tr, group, s))
    return out


def _limit_hits(tr: HoldTrace, f: StepFile, rows: np.ndarray, w: np.ndarray,
                settled_s: float, up_s: Optional[float]) -> List[_Hit]:
    """Where a hold file's own limit acted in one run (`rows`: the run's
    samples that have a time, in time order; `w` their Wet flags).

    Only readings at open circuit count, and the halves are read from the
    rows with a Wet flag: a row whose flag is blank belongs to neither.

    - A step that is not an RH loop (StepFile.loop: a steady RH, a flag
      switched now and then, or none), the station ends on the limit: a run
      whose last reading is under it was stopped there, dated from the start
      of the stretch under it, or from the run's first reading if the OCV
      never came up to the limit (a cell already dead when the hold began).
    - In an RH loop whose flags do not switch in the run, the halves cannot
      be told apart, so its limit is not judged (for a hold named for RH
      cycling, read_hold_rh_cycles says its flags never switch).
    - In an RH loop, the limit belongs to one sub-step of the loop: the one
      whose whole halves last the time the header gives (KM8-S6: 30 Seconds,
      the wet one; if neither or both match, the wet one). The limit tripped
      in such a half when a reading under it ends the half -- its last
      flagged row, or a row with a blank flag before the next half -- the run
      then moving on to the other half, or ending. FuelCell may act on a
      reading it does not log, so a half that ended at least a sample and a
      half early is a trip too when a reading at its end is within
      TRIP_NEAR_V of the limit -- only where the limit's whole halves are
      seen to last the header's time, never in a run's first or last half
      (which may lie partly outside its log), and only on a clean change of
      flag at both its ends: no blank flag in it or either side of it, no
      row missing from the log, and halves of more than a sample either
      side, so a flag or logging glitch is not read as the station acting. A run
      ending in the other half under the limit is no act of the station's:
      that sub-step has no limit known, and the OCV rule judges it.

    A trip counts once `up_s`, the run's first reading at the limit, has
    passed and the run has settled (`settled_s`); with no reading at the
    limit in the run (up_s None), from the first.
    """
    lim = f.term_v
    t, v = tr.t_s[rows], tr.v[rows]
    ocv = tr.finite[rows] & ~tr.on_load[rows]
    oc = np.flatnonzero(ocv)
    if not len(oc):
        return []
    under = ocv & (v < lim)
    # Where the stretch under the limit around each reading started; rows
    # without a reading neither end a stretch nor add to it. (Readings over
    # the limit get the next one, clipped: never looked up.)
    start_of = oc[np.minimum(np.maximum.accumulate(np.where(under[oc], -1, np.arange(len(oc)))) + 1,
                             len(oc) - 1)]
    stretch = lambda p: int(rows[start_of[np.searchsorted(oc, p)]])
    if not f.loop:
        p = oc[-1]
        return [(stretch(p), int(rows[p]), 'stop', f, '')] if under[p] else []
    have = np.flatnonzero(np.isfinite(w))
    state = (w[have] > 0.5).astype(float)
    if not len(have) or np.all(state == state[0]):
        return []
    # The halves: stretches of one state, (state, first row, last row, followed).
    edge = np.flatnonzero(np.diff(state) != 0) + 1
    firsts, lasts = np.r_[0, edge], np.r_[edge, len(have)] - 1
    halves = [(state[a], have[a], have[b], i < len(firsts) - 1)
              for i, (a, b) in enumerate(zip(firsts, lasts))]
    # A half with a flag on every row (no blank one inside it).
    flagged = [lasts[i] - firsts[i] == h[2] - h[1] for i, h in enumerate(halves)]
    d = np.diff(t)
    dt = float(np.median(d[d > 0])) if (d > 0).any() else 0.0
    lasts_s = lambda a, b: float(t[b] - t[a]) + dt
    # A whole half (another before it and after it) that lasts the header's
    # time, within a sample and a half.
    whole = [h for i, h in enumerate(halves) if f.sub_s and h[3] and i > 0
             and abs(lasts_s(h[1], h[2]) - f.sub_s) <= 1.5 * dt]
    states = {h[0] for h in whole}
    lim_state = states.pop() if len(states) == 1 else 1.0
    early_ok = dt > 0 and any(h[0] == lim_state for h in whole)
    name = 'wet' if lim_state == 1.0 else 'dry'
    out: List[_Hit] = []
    for i, (sv, a, b, followed) in enumerate(halves):
        if sv != lim_state:
            continue
        nxt = halves[i + 1][1] if followed else len(rows)       # the next half's first row
        ends = np.arange(b, nxt)                                # its last flagged row and blank ones after
        cand = ends[under[ends]]
        if len(cand):
            p = int(cand[0])
        elif early_ok and followed and i > 0 and lasts_s(a, b) < f.sub_s - 1.5 * dt:
            # Ended early: only a clean change of flag at both its ends, the
            # halves either side of it more than a sample long, nothing missing
            # from the log, and a reading at its end near the limit.
            prev, after = halves[i - 1], halves[i + 1]
            clean = (prev[2] == a - 1 and after[1] == b + 1 and b > a
                     and prev[2] > prev[1] and after[2] > after[1]
                     and all(flagged[i - 1:i + 2])
                     and np.all(np.diff(t[a - 1:b + 2]) <= 1.5 * dt))
            mine = oc[(oc >= a) & (oc <= b)]
            near = np.r_[mine[-2:], oc[(oc > b) & (oc <= after[2])][:2]]
            if not clean or not len(mine) or v[near].min() >= lim + TRIP_NEAR_V:
                continue
            p = int(mine[-1])
        else:
            continue
        if tr.on_load[rows[b]] or (up_s is not None and (t[p] < up_s or t[p] < settled_s)):
            continue
        under_p = bool(under[p])
        out.append((stretch(p) if under_p else int(rows[p]), int(rows[p]),
                    'trip' if followed else 'trip-end', f, name))
    return out


def _station_run(tr: HoldTrace, group: List[_Hit], s: OcvSettings) -> OcvRun:
    """One event of a hold's own limit: from the start of its first stretch
    under the limit to the last time the limit acted, the time under the
    limit in it, its lowest reading, whether the 10-min median fell under the
    threshold in it, and the first judged median at or over the threshold
    after it."""
    lead = min(group, key=lambda h: h[0])
    last = max(group, key=lambda h: h[1])
    a, b = lead[0], last[1]
    span = tr.finite & ~tr.on_load & (tr.t_s >= tr.t_s[a]) & (tr.t_s <= tr.t_s[b])
    trips = [h for h in group if h[2] != 'stop']
    if trips:
        # The time under the limit, reading by reading: in a loop the OCV
        # can be over it between trips.
        d = np.diff(np.unique(tr.t_s[np.isfinite(tr.t_s)]))
        dt = float(np.median(d)) if len(d) else 0.0
        minutes = float(np.sum(span & (tr.v < max(h[3].term_v for h in group)))) * dt / 60.0
    else:
        minutes = float(tr.t_s[b] - tr.t_s[a]) / 60.0
    back = np.flatnonzero(tr.valid & (tr.median >= s.threshold_v))
    later = back[back > b]
    judged = span & tr.valid
    return OcvRun(segment=tr.segment.number, step=lead[3].step,
                  start_h=float(tr.t_s[a]) / 3600.0, end_h=float(tr.t_s[b]) / 3600.0,
                  minutes=minutes,
                  lowest_v=float(tr.v[span].min()) if span.any() else float(tr.v[b]),
                  recovered_h=float(tr.t_s[later[0]]) / 3600.0 if len(later) else None,
                  verdict='death', trigger='station', trips=len(trips),
                  stopped=any(h[2] == 'stop' for h in group),
                  run_ended=last[2] in ('stop', 'trip-end'),
                  half=trips[0][4] if trips else '',
                  median_fell=bool(np.any(tr.median[judged] < s.threshold_v)))


def set_rate_cutoff(res: OcvResult, death_h: Optional[float], cause: str = '') -> None:
    """Choose the headline decay rate: the latest window at the reference
    conditions, and for a sample that died -- by either rule -- the latest
    one ending by the death. A rate over a dead cell's trace says nothing
    about how it got there. Sets rate_note for the notes."""
    if not res.holds:
        return
    cutoff = death_h + 1e-9 if death_h is not None else math.inf
    ref_key = res.holds[0].conditions_key
    rated = [r for h in res.holds if h.conditions_key == ref_key for r in h.rates
             if math.isfinite(r['decay']) and r['end'] <= cutoff]
    res.decay_uV_h, res.decay_window_h = math.nan, (math.nan, math.nan)
    if rated:
        res.decay_uV_h = rated[-1]['decay']
        res.decay_window_h = (rated[-1]['start'], rated[-1]['end'])
    res.rate_cutoff_h = death_h
    res.by_condition = _by_condition(res.holds, ref_key, cutoff)
    res.rate_note = ''
    if death_h is not None:
        what = f'the {cause} death at {death_h:.2f} h' if cause else f'the death at {death_h:.2f} h'
        a, b = res.decay_window_h
        res.rate_note = (f'OCV decay rate is from the latest window ending by {what} '
                         f'({a:.2f}–{b:.2f} h)' if math.isfinite(a) else
                         f'OCV decay rate: no rolling window at the reference conditions ends '
                         f'by {what}')


def hold_rows(tl: Timeline, ocv: Optional[OcvResult] = None) -> List[Dict[str, Any]]:
    """The Holds sheet: each hold segment, with its OCV when analyzed."""
    rows = segment_rows(tl)
    if ocv is None:
        return rows
    for row, h in zip(rows, ocv.holds):
        rate = h.latest_rate
        died = ocv.death is not None and ocv.death.segment == h.segment.number
        row.update({
            'Settled start (V)': _r(h.start_v, 5),
            'End (V)': _r(h.end_v, 5),
            'Loss in hold (%)': _r((h.start_v - h.end_v) / h.start_v * 100.0, 3) if h.start_v else None,
            'Mean (V)': _r(h.mean_v, 5),
            'Lowest 10-min median (V)': _r(h.lowest_v, 5),
            'Lowest at (h)': _r(h.lowest_h, 4),
            'Latest decay rate (μV/h)': _r(rate['decay'], 1) if rate else None,
            'Dips': sum(r.verdict == 'dip' for r in h.runs),
            'OCV death (h)': _r(ocv.death.start_h, 4) if died else None,
            'Samples': h.samples,
            'Judged': h.judged,
            'Settling': h.settling,
            'On load': h.on_load,
            'Not finite': h.not_finite,
            'Judged samples under threshold': h.raw_below,
        })
    return rows


EVENT_COLUMNS = ['Segment', 'Step', 'Start (h)', 'End (h)', 'Minutes',
                 'Lowest 10-min median (V)', 'Verdict', 'Median at or over threshold after (h)']


def ocv_event_rows(res: OcvResult, tl: Optional[Timeline] = None) -> List[Dict[str, Any]]:
    """Every stretch of 10-min median under the threshold, judged or not, and
    every event of a hold's own limit (a step the station ended, wet halves
    it cut short); with RH cycling, where each starts in RH cycles too."""
    words = {'death': 'death',
             'dip': f'dip, shorter than {res.settings.persist_min:g} min',
             'after death': 'after death'}
    return [{'Segment': r.segment, 'Step': r.step,
             'Start (h)': round(r.start_h, 4), 'End (h)': round(r.end_h, 4),
             'Minutes': round(r.minutes, 2), 'Lowest 10-min median (V)': round(r.lowest_v, 5),
             'Verdict': words[r.verdict] + (f': {r.station_acted()} (lowest sample shown)'
                                            if r.trigger == 'station' else ''),
             'Median at or over threshold after (h)': _r(r.recovered_h, 4),
             **({'Start (RH cycles)': tl.cycles_at(r.start_h)} if tl is not None and tl.cycles_tested
                else {})}
            for r in res.runs]


DECAY_COLUMNS = ['Segment', 'Conditions', 'Window start (h)', 'Window end (h)', 'Window (h)',
                 'Coverage', 'Mean (V)', 'Decay rate (μV/h)']


def decay_rows(res: OcvResult) -> List[Dict[str, Any]]:
    return [{'Segment': h.segment.number, 'Conditions': h.segment.conditions,
             'Window start (h)': round(r['start'], 4), 'Window end (h)': round(r['end'], 4),
             'Window (h)': round(r['end'] - r['start'], 4), 'Coverage': round(r['coverage'], 3),
             'Mean (V)': _r(r['mean'], 5), 'Decay rate (μV/h)': _r(r['decay'], 1)}
            for h in res.holds for r in h.rates]


# ═══════════════════════════════════════════════════════════════════
#  Characterizations
# ═══════════════════════════════════════════════════════════════════

# Crossover death (plan decisions 3, 5, 12, 13): at a characterization
# interval, the 100% RH crossover -- the window mean of the averaged CV
# current over 0.35-0.50 V, last of any repeats -- above 15 mA/cm².
XOVER_THRESHOLD = 15.0           # mA/cm²
XOVER_WINDOW_V = (0.35, 0.50)
XOVER_DEATH_RH = 100.0
# Repeats of one measurement in one block more than this far apart are
# flagged; the last still represents the block (decision 12).
REPEAT_SPREAD = 0.20
# ECSA as Full Analysis's station preset: H_UPD charge over 0.08-0.40 V on
# the second cycle, at the scan rate the log shows.
ECSA_WINDOW_V = (0.08, 0.40)
ECSA_CYCLE = 2
ECSA_NOMINAL_RATE = 0.050        # V/s, if the log gives no usable rate
# DOE membrane chemical-stability AST targets (decision 8): reported beside
# the verdict as hours to first breach, never judged.
DOE_XOVER = 2.0                  # mA/cm²
DOE_OCV_LOSS_PCT = 20.0
DOE_R_SHORT = 1000.0             # Ω·cm²
# Evolution plots: at most two series per plot and four curves per series
# (BOL, last, evenly between), each thinned to these many points, so that a
# long two-RH test keeps its sidecars near the plan's ~150 KB.
EVOLUTION_MAX_SERIES = 2
EVOLUTION_MAX_CURVES = 4
EVOLUTION_POINTS = {'crossover': 50, 'ecsa': 60, 'eis': 61, 'polcurve': 40}
EVOLUTION_MAX_POINTS = max(EVOLUTION_POINTS.values())

CHAR_KINDS = ('crossover', 'ecsa', 'eis', 'polcurve')
KIND_LABELS = {'crossover': 'Crossover', 'ecsa': 'ECSA', 'eis': 'EIS', 'polcurve': 'Pol curve'}
# kind -> [(value key, column heading, decimals)]; the first is the block's headline.
METRICS: Dict[str, List[Tuple[str, str, int]]] = {
    'crossover': [('j_xover', 'j_xover (mA/cm²)', 3), ('j_intercept', 'Intercept (mA/cm²)', 3),
                  ('R_short', 'R_short (Ω·cm²)', 0), ('fit_R2', 'Fit R²', 4)],
    'ecsa': [('RF', 'Roughness factor', 2), ('ECSA', 'ECSA (m²/g)', 2),
             ('Q_H', 'Q_H (mC/cm²)', 4), ('scan_rate', 'Scan rate (mV/s)', 2),
             ('cycle', 'Cycle used', 0)],
    'eis': [('HFR', 'HFR (mΩ·cm²)', 2)],
    'polcurve': [('V_1A', 'V @ 1 A/cm² (V)', 4), ('j_0.65V', 'j @ 0.65 V (A/cm²)', 4),
                 ('peak_power', 'Peak power (W/cm²)', 4)],
}


@dataclass
class CharSettings:
    """Parameters of the characterization analysis; run() takes each from its params."""
    geo_area: float = DEFAULT_GEO_AREA
    loading: float = DEFAULT_LOADING
    xover_threshold: float = XOVER_THRESHOLD
    xover_lo: float = XOVER_WINDOW_V[0]
    xover_hi: float = XOVER_WINDOW_V[1]
    death_rh: float = XOVER_DEATH_RH


def _quiet(fn, *args, **kwargs):
    """Call a portal analysis function without its console chatter."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _columns(path: str, names: Tuple[str, ...],
             skip_modes: Tuple[int, ...] = ()) -> Dict[str, np.ndarray]:
    """Named numeric columns of a step file, by header name, with rows that
    lack any of them dropped. A .fcd file's columns come under the station
    names (fcd_columns), less rows in a control mode of `skip_modes`."""
    if is_fcd(path):
        df = fcd_columns(path, names, skip_modes)
    else:
        df = pd.read_csv(path, usecols=lambda c: str(c).strip() in names)
        df.columns = [str(c).strip() for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
    missing = [n for n in names if n not in df.columns]
    if missing:
        raise ValueError('no ' + ', '.join(f"'{n}'" for n in missing) + ' column')
    cols = {n: pd.to_numeric(df[n], errors='coerce').to_numpy(float) for n in names}
    keep = np.all([np.isfinite(a) for a in cols.values()], axis=0)
    if keep.sum() < 5:
        raise ValueError(f'only {int(keep.sum())} readable rows')
    return {n: a[keep] for n, a in cols.items()}


def analyze_crossover_file(path: str, cs: CharSettings):
    """Crossover from an H2/N2 CV, by the portal's analyze_crossover on
    working-electrode voltage and current density: the window mean (decision
    3), and the short-corrected intercept and R_short from a line through the
    same window. R_short is left out when that line is flat or falling."""
    from scripts.h2_crossover_analysis import analyze_crossover
    c = _columns(path, ('Working Electrode Voltage', 'Current'), FCD_SKIP_MODES)
    V, j = c['Working Electrode Voltage'], c['Current'] / cs.geo_area
    r = _quiet(analyze_crossover, V, j, avg_V_min=cs.xover_lo, avg_V_max=cs.xover_hi,
               geo_area=cs.geo_area)
    note = ''
    if r['j_avg_mean_mA_cm2'] < 0:
        # Oxidation logged as negative current: flip it, or the intercept
        # and R_short come out with the wrong sign. Both stations log it positive.
        r = _quiet(analyze_crossover, V, -j, avg_V_min=cs.xover_lo, avg_V_max=cs.xover_hi,
                   geo_area=cs.geo_area)
        note = 'oxidation current logged as negative; sign flipped'
    R = float(r['R_short_ohm_cm2'])
    values = {'j_xover': float(r['j_xover_mA_cm2']),
              'j_intercept': float(r['j_intercept_mA_cm2']),
              'R_short': R if math.isfinite(R) and R > 0 else math.nan,
              'fit_R2': float(r['R_squared'])}
    return values, (np.asarray(r['V_grid'], float), np.asarray(r['j_avg'], float) * 1000.0), note


def analyze_ecsa_file(path: str, cs: CharSettings):
    """Roughness factor and ECSA from an H2/N2 CV by the portal's
    compute_ecsa_hupd, with Full Analysis's station preset (0.08-0.40 V,
    second cycle) and the scan rate the log shows."""
    from scripts.ecsa_analysis import compute_ecsa_hupd, extract_cycles, select_cycle
    c = _columns(path, ('Time Stamp', 'Working Electrode Voltage', 'Current'))
    t, V, I = c['Time Stamp'], c['Working Electrode Voltage'], c['Current']
    dt = np.diff(t)
    step = dt > 0
    measured = float(np.median(np.abs(np.diff(V))[step] / dt[step])) if step.any() else math.nan
    assumed = not (0.005 <= measured <= 0.5)
    rate = ECSA_NOMINAL_RATE if assumed else measured
    cycles, _ = _quiet(extract_cycles, V, I)
    cycle = ECSA_CYCLE if len(cycles) >= ECSA_CYCLE else 'last'
    r = _quiet(compute_ecsa_hupd, V, I, rate, cs.geo_area, v_low=ECSA_WINDOW_V[0],
               v_high=ECSA_WINDOW_V[1], loading_mg_cm2=cs.loading, cycle=cycle)
    Vc, Ic = select_cycle(r['_cycles'], cycle)
    values = {'RF': float(r['roughness_factor']),
              'ECSA': float(r.get('ECSA_m2_per_g', math.nan)),
              'Q_H': float(r['Q_hupd_mC_cm2']),
              'scan_rate': math.nan if assumed else rate * 1000.0,
              'cycle': float(cycle if isinstance(cycle, int) else len(cycles))}
    notes = []
    if assumed:
        notes.append(f'scan rate from the log ({measured * 1000:.3g} mV/s) is implausible; '
                     f'assumed {ECSA_NOMINAL_RATE * 1000:g} mV/s')
    if cycle == 'last':
        notes.append(f'only {len(cycles)} cycle(s); the last was analyzed, not cycle {ECSA_CYCLE}')
    return (values, (np.asarray(Vc, float), np.asarray(Ic, float) / cs.geo_area * 1000.0),
            '; '.join(notes))


def analyze_eis_file(path: str, cs: CharSettings):
    """HFR from a GEIS sweep by the portal's extract_hfr (the high-frequency
    zero crossing of Z''), in mΩ·cm². The stations store Z'' negative for a
    capacitive arc, the convention extract_hfr expects."""
    from scripts.eis_analysis import extract_hfr
    c = _columns(path, ('Frequency', 'Real Impedance', 'Imaginary Impedance'))
    order = np.argsort(c['Frequency'])[::-1]
    f, zr, zi = (c[k][order] for k in ('Frequency', 'Real Impedance', 'Imaginary Impedance'))
    hfr = float(extract_hfr(f, zr, zi))
    scale = cs.geo_area * 1000.0
    return {'HFR': hfr * scale}, (zr * scale, -zi * scale), ''


def analyze_polcurve_file(path: str, cs: CharSettings):
    """V @ 1 A/cm², j @ 0.65 V and peak power from a stepped pol curve (mV and
    mA columns), through the same steps as the portal's pol-curve batch:
    dwell endpoints, sweeps, the last up-sweep, then analyze_polcurve."""
    import scripts.polcurve_analysis as PC
    c = _columns(path, ('mV', 'mA'), FCD_SKIP_MODES)
    j_raw = c['mA'] * 1e-3 / cs.geo_area
    V_raw = c['mV'] * 1e-3

    def representative():
        v_step = PC.detect_voltage_step(V_raw)
        j_d, V_d, H_d, c_d = PC.extract_dwell_endpoints(j_raw, V_raw, None, None, v_step=v_step)
        cycles = PC.extract_polcurve_cycles(j_d, V_d, H_d, c_d, v_step=v_step)
        cycles = PC.enforce_consensus_setpoints(cycles, v_step)
        rep = PC.select_representative_cycle(cycles, choice='last_up')
        j, V, H, cond, _ = PC.deduplicate_polcurve(rep['j'], rep['V'], rep.get('HFR'),
                                                   rep.get('conditions'), v_step=v_step)
        j, V, H, cond = PC.enforce_v_monotonicity(j, V, H, cond)
        j, V, H, cond = PC.smooth_curve_selection(j, V, H, cond)
        return np.asarray(j, float), np.asarray(V, float)

    j, V = _quiet(representative)
    r = _quiet(PC.analyze_polcurve, j, V, HFR=None, geo_area=cs.geo_area)
    v1 = r.get('V_at_1Acm2')
    values = {'V_1A': float(v1) if v1 is not None else math.nan,
              'j_0.65V': float(r.get('j_at_0.65V', math.nan)),
              'peak_power': float(r['peak_power_W_cm2'])}
    note = '' if v1 is not None else f'reaches {float(j.max()):.2f} A/cm², short of 1 A/cm²'
    return values, (j, V), note


ANALYZERS = {'crossover': analyze_crossover_file, 'ecsa': analyze_ecsa_file,
             'eis': analyze_eis_file, 'polcurve': analyze_polcurve_file}


@dataclass
class CharResult:
    """One characterization file's analysis."""
    file: StepFile
    series: str
    values: Dict[str, float] = field(default_factory=dict)
    curve: Optional[Tuple[np.ndarray, np.ndarray]] = None
    error: str = ''
    note: str = ''
    block: bool = False      # represents its block: the last good repeat of its series

    @property
    def kind(self) -> str:
        return self.file.role

    @property
    def interval(self) -> int:
        return self.file.interval

    @property
    def H(self) -> float:
        return self.file.H_start_h

    @property
    def headline(self) -> float:
        return self.values.get(METRICS[self.kind][0][0], math.nan)


@dataclass
class CharInterval:
    """A point on the hold clock where characterization files were logged."""
    index: int               # hold segments completed before it
    H: float                 # hold hours completed before it
    label: str               # 'BOL' before the first hold, else '150 h'


@dataclass
class XoverDeath:
    interval: CharInterval
    previous: Optional[CharInterval]   # last earlier interval with a crossover, if any
    result: CharResult

    @property
    def H(self) -> float:
        return self.interval.H

    @property
    def low(self) -> float:
        """Lower end of the bracket: the previous crossover, or the start of
        the test when none was measured before."""
        return self.previous.H if self.previous else 0.0


@dataclass
class CharAnalysis:
    settings: CharSettings
    results: List[CharResult] = field(default_factory=list)
    intervals: List[CharInterval] = field(default_factory=list)
    series: List[str] = field(default_factory=list)       # labels, in order of first use
    series_kind: Dict[str, str] = field(default_factory=dict)
    death_series: List[str] = field(default_factory=list)  # 100% RH crossover series
    xover_death: Optional[XoverDeath] = None
    xover_judged: bool = False      # some interval had a crossover the rule could use
    doe_xover: Optional[CharInterval] = None
    doe_r_short: Optional[CharInterval] = None
    notes: List[str] = field(default_factory=list)

    def block(self, series: str, index: int) -> Optional[CharResult]:
        return next((r for r in self.results
                     if r.block and r.series == series and r.interval == index), None)

    def blocks(self, series: str) -> List[CharResult]:
        return [r for r in self.results if r.block and r.series == series]

    @property
    def primary_death_series(self) -> Optional[str]:
        """The 100% RH crossover series the headline values follow: the one
        measured at the most characterizations (the first, on a tie)."""
        if not self.death_series:
            return None
        return max(self.death_series,
                   key=lambda s: (len(self.blocks(s)), -self.death_series.index(s)))

    def death_blocks(self) -> List[CharResult]:
        """The headline crossover series' block values, in clock order."""
        s = self.primary_death_series
        return self.blocks(s) if s else []

    def judged_at(self, index: int) -> List[CharResult]:
        """The 100% RH crossover blocks at an interval, one per series."""
        return [b for b in (self.block(s, index) for s in self.death_series) if b is not None]

    def label(self, index: int) -> str:
        return next((iv.label for iv in self.intervals if iv.index == index), '')


def _hours_label(H: float) -> str:
    return f'{H:.2f}'.rstrip('0').rstrip('.') + ' h'


def _series_labels(files: List[StepFile]) -> Dict[Tuple[Any, ...], str]:
    """A readable name per measurement series (kind and conditions): kind
    and RH, plus gases and flows for pol curves, which the stations run at
    two flow settings. Names that would still collide carry the temperature
    (and the gases, where not already named), then a count."""
    keys: Dict[Tuple[Any, ...], StepFile] = {}
    for f in files:
        keys.setdefault((f.role,) + f.conditions_key, f)
    names: Dict[Tuple[Any, ...], str] = {}
    for key, f in keys.items():
        base = f'{KIND_LABELS[f.role]}, {f.rh_label}'
        names[key] = f'{base}, {f.gases}' if f.role == 'polcurve' else base

    def collisions() -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for n in names.values():
            counts[n] = counts.get(n, 0) + 1
        return counts

    counts = collisions()
    for key, f in keys.items():
        if counts[names[key]] > 1:
            names[key] += (f', {f.T_C:g} °C' if f.role == 'polcurve'
                           else f', {f.T_C:g} °C, {f.gases}')
            if f.P_kPa is not None:
                names[key] += f', {f.P_kPa:g} kPa'
    counts, seen = collisions(), {}
    for key in keys:
        n = names[key]
        if counts[n] > 1:
            seen[n] = seen.get(n, 0) + 1
            names[key] = f'{n} ({seen[n]})'
    return names


def _analyze_file(f: StepFile, series: str, cs: CharSettings) -> CharResult:
    """One file through its analyzer. Only an analyzer that raises, or one
    that yields no number at all, makes an error: a pol curve that stops
    short of 1 A/cm² still has j @ 0.65 V and peak power."""
    res = CharResult(file=f, series=series)
    try:
        res.values, res.curve, res.note = ANALYZERS[f.role](f.path, cs)
        if not any(math.isfinite(x) for x in res.values.values()):
            res.error = 'no value could be computed'
    except Exception as e:                           # noqa: BLE001
        res.error = f'{type(e).__name__}: {e}'
    return res


def analyze_characterizations(tl: Timeline, settings: Optional[CharSettings] = None
                              ) -> CharAnalysis:
    """Every crossover, ECSA, EIS and pol-curve file, the block value of each
    measurement at each characterization interval, and the crossover rule."""
    cs = settings or CharSettings()
    files = [f for f in tl.files if f.role in CHAR_KINDS]
    names = _series_labels(files)
    results = [_analyze_file(f, names[(f.role,) + f.conditions_key], cs) for f in files]
    window_notes = []
    if ((cs.xover_lo, cs.xover_hi) != XOVER_WINDOW_V
            and any(r.kind == 'crossover' and 'Too few points' in r.error for r in results)):
        # A window the CVs do not span would leave the rule unjudged and
        # call a dead sample alive; the plan's window is the fallback.
        window_notes.append(f'crossover window {cs.xover_lo:g}–{cs.xover_hi:g} V lies outside '
                            f'the CVs; used {XOVER_WINDOW_V[0]:g}–{XOVER_WINDOW_V[1]:g} V')
        cs = dc_replace(cs, xover_lo=XOVER_WINDOW_V[0], xover_hi=XOVER_WINDOW_V[1])
        results = [_analyze_file(f, r.series, cs) if f.role == 'crossover' else r
                   for f, r in zip(files, results)]
    ca = CharAnalysis(settings=cs, results=results, notes=window_notes)
    for r in results:
        if r.series not in ca.series:
            ca.series.append(r.series)
            ca.series_kind[r.series] = r.kind
            if r.kind == 'crossover' and r.file.RH_pct == cs.death_rh:
                ca.death_series.append(r.series)
            elif r.kind == 'crossover' and not math.isfinite(r.file.RH_pct):
                ca.notes.append(f'{r.series}: the file names give no RH, so these crossovers are '
                                f'not judged by the {cs.death_rh:g}% RH crossover rule')

    by_index: Dict[int, float] = {}
    for r in ca.results:
        by_index.setdefault(r.interval, r.H)
    ca.intervals = [CharInterval(k, H, 'BOL' if k == 0 else _hours_label(H))
                    for k, H in sorted(by_index.items())]
    label = {iv.index: iv.label for iv in ca.intervals}

    # The last good repeat of each series in each block represents it.
    for series in ca.series:
        for iv in ca.intervals:
            mine = [r for r in ca.results if r.series == series and r.interval == iv.index]
            good = [r for r in mine if not r.error]
            if good:
                good[-1].block = True
            rated = [r for r in good if math.isfinite(r.headline)]
            if len(rated) > 1:
                vals = [abs(r.headline) for r in rated]
                if min(vals) > 0 and (max(vals) - min(vals)) / min(vals) > REPEAT_SPREAD:
                    listed = ', '.join(f'{r.file.step} {r.headline:.3g}' for r in rated)
                    ca.notes.append(f'{iv.label}: {series} repeats differ by '
                                    f'{(max(vals) - min(vals)) / min(vals):.0%} ({listed}); '
                                    f'the last, {good[-1].file.step}, represents the block')
            if mine and not good:
                ca.notes.append(f'{iv.label}: no {series} could be analyzed')
    for r in ca.results:
        if r.error:
            ca.notes.append(f'{label[r.interval]}: {r.file.step} ({KIND_LABELS[r.kind]}) could '
                            f'not be analyzed: {r.error}')
        elif r.note:
            ca.notes.append(f'{label[r.interval]}: {r.file.step} ({KIND_LABELS[r.kind]}) {r.note}')

    # The crossover rule (decisions 3, 5, 13): each 100% RH series' block
    # values against the threshold, bracketed back to that series' previous
    # measurement; the earliest trip of any series is the death.
    trips: List[XoverDeath] = []
    for series in ca.death_series:
        previous: Optional[CharInterval] = None
        for iv in ca.intervals:
            blk = ca.block(series, iv.index)
            if blk is None:
                continue
            if blk.values['j_xover'] > cs.xover_threshold:
                trips.append(XoverDeath(iv, previous, blk))
                break
            previous = iv
    if trips:
        ca.xover_death = min(trips, key=lambda d: (d.H, ca.death_series.index(d.result.series)))
    if len(ca.death_series) > 1:
        ca.notes.append(f'{len(ca.death_series)} {cs.death_rh:g}% RH crossover series '
                        f'({"; ".join(ca.death_series)}) are each judged on their own, and the '
                        f'earliest trip counts; j_xover initial and final follow '
                        f'{ca.primary_death_series}')

    # Notes and DOE breaches, characterization by characterization.
    for iv in ca.intervals:
        here = ca.judged_at(iv.index)
        if not here:
            ca.notes.append(f'{iv.label}: no {cs.death_rh:g}% RH crossover, so this '
                            f'characterization is skipped for the crossover rule')
            continue
        ca.xover_judged = True
        for blk in here:
            v = blk.values
            if ca.doe_xover is None and v['j_xover'] > DOE_XOVER:
                ca.doe_xover = iv
            if ca.doe_r_short is None and v['R_short'] < DOE_R_SHORT:
                ca.doe_r_short = iv
            if v['R_short'] < DOE_R_SHORT:
                ca.notes.append(f'{iv.label}: R_short {v["R_short"]:.0f} Ω·cm² ({blk.file.step}); '
                                f'the short adds '
                                f'{(cs.xover_lo + cs.xover_hi) / 2 / v["R_short"] * 1000:.2f} '
                                f'mA/cm² to the window mean')
            over = v['j_xover'] > cs.xover_threshold
            if over != (v['j_intercept'] > cs.xover_threshold):
                ca.notes.append(f'{iv.label}: crossover {v["j_xover"]:.2f} mA/cm² '
                                f'({blk.file.step}) is {"above" if over else "below"} '
                                f'{cs.xover_threshold:g} but the short-corrected intercept '
                                f'{v["j_intercept"]:.2f} is not; the verdict follows the window '
                                f'mean (decision 3)')
    if ca.intervals and not ca.xover_judged:
        ca.notes.append(f'no {cs.death_rh:g}% RH crossover was analyzed, so the crossover '
                        f'rule could not be applied')
    if ca.intervals and ca.intervals[0].index > 0:
        ca.notes.append(f'no BOL characterization in this upload; initial values are from '
                        f'{ca.intervals[0].label}')
    return ca


@dataclass
class Verdict:
    """The death verdict combining the OCV and crossover rules (plan: death
    rules). The earlier trip sets hours to death and cause; the cause is
    'both' when the OCV trip falls inside the crossover bracket. Neither
    tripping leaves the sample alive at the hours tested (censored)."""
    dead: bool
    hours: float                    # hours to death, or hours tested when alive
    cause: str = ''                 # 'OCV', 'crossover' or 'both'
    bracket: Optional[Tuple[float, float]] = None   # (low, high] of a crossover death
    at_bol: bool = False
    ocv_judged: bool = True
    xover_judged: bool = True
    ocv_trigger: str = ''           # 'station' when the station's own limit was the OCV death
    cycles: Optional[int] = None    # RH cycles to death, or tested when alive; None without

    @property
    def cause_text(self) -> str:
        """The cause, saying when the OCV death was the station's own limit."""
        if self.ocv_trigger == 'station' and self.cause in ('OCV', 'both'):
            return f'{self.cause} (station limit)'
        return self.cause

    @property
    def text(self) -> str:
        if not self.dead:
            return f'alive at {self.hours:.2f} h'
        if self.at_bol:
            return 'dead at BOL (0 h), crossover'
        if self.bracket and self.cause == 'crossover':
            return (f'dead at {self.hours:.2f} h, crossover, between '
                    f'{self.bracket[0]:.2f} and {self.bracket[1]:.2f} h')
        return f'dead at {self.hours:.2f} h, {self.cause_text}'

    @property
    def bracket_text(self) -> str:
        if not self.dead or self.cause != 'crossover' or not self.bracket:
            return ''
        return 'at BOL' if self.at_bol else f'({self.bracket[0]:.2f}, {self.bracket[1]:.2f}] h'


# Hold hours that differ by less than this (3.6 µs) are the same moment: sums
# of the same lengths in another order can differ in the last bit.
CLOCK_TOL_H = 1e-9


def decide(tl: Timeline, ocv: Optional['OcvResult'], ca: Optional[CharAnalysis]) -> Verdict:
    t_ocv = ocv.death_h if ocv is not None else None
    xd = ca.xover_death if ca is not None else None
    judged = dict(ocv_judged=bool(ocv is not None and ocv.judged),
                  xover_judged=bool(ca is not None and ca.xover_judged))
    # The two clocks meet where a hold the station ended is followed by the
    # crossover that trips: the OCV death is at the hold's last sample, the
    # crossover at the hold's end, the same hour summed two ways. Within
    # CLOCK_TOL_H they are the same time.
    if t_ocv is None and xd is None:
        v = Verdict(False, tl.hours_tested, **judged)
    elif xd is not None and (t_ocv is None or xd.H < t_ocv - CLOCK_TOL_H):
        v = Verdict(True, xd.H, 'crossover', (xd.low, xd.H), at_bol=xd.interval.index == 0,
                    **judged)
    else:
        both = xd is not None and xd.low < t_ocv <= xd.H + CLOCK_TOL_H
        v = Verdict(True, t_ocv, 'both' if both else 'OCV', (xd.low, xd.H) if both else None,
                    ocv_trigger=ocv.death.trigger, **judged)
    if tl.cycles_tested:
        v.cycles = tl.cycles_at(v.hours) if v.dead else tl.cycles_tested
    return v


# ═══════════════════════════════════════════════════════════════════
#  Outputs
# ═══════════════════════════════════════════════════════════════════

def _readout(ax, lines, x=1.0, y=1.02, ha='right', va='bottom'):
    """A 'key = value' box: the record layer reads text annotations, never
    titles or legends, so any number meant for the JSON record goes here."""
    lines = [ln for ln in (lines or []) if ln]
    if lines:
        ax.text(x, y, '\n'.join(lines), transform=ax.transAxes, ha=ha, va=va,
                fontsize=8.5, zorder=5,
                bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', alpha=0.9))


_LANES = (('hold', 'OCV hold', '#1f77b4'),
          ('air-step', 'Air OCV step', '#9ecae1'),
          ('h2n2', 'H₂/N₂ step', '#bdbdbd'),
          ('n2n2', 'N₂/N₂ step', '#d9d9d9'),
          ('ocv-unread', 'OCV, gases unread', '#fdae6b'),
          ('breakin', 'Break-in', '#8c6d31'),
          ('polcurve', 'Pol curve', '#2ca02c'),
          ('crossover', 'H₂ crossover', '#d62728'),
          ('ecsa', 'ECSA', '#9467bd'),
          ('eis', 'EIS', '#e377c2'),
          ('eclean', 'Electrode cleaning', '#7f7f7f'),
          ('other', 'Other', '#bcbd22'))


# A step that adds no hold time is drawn on the timeline as a tick this
# share of the test wide, centred on the hold hour it ran at; so is a hold
# segment shorter than that, in a darker blue.
TIMELINE_TICK = 0.006
SHORT_HOLD_COLOR = '#08306b'


def timeline_lanes(tl: Timeline) -> Tuple[Dict[str, List[Tuple[float, float]]],
                                          Dict[str, float], float, bool]:
    """What the timeline draws: each role's spans (h), where each folder
    starts, the length of the axis, and whether it is the hold clock.

    On the hold clock, OCV hold time only (Corey, 13 Oct), each hold segment
    spans its hold hours, and every other step takes no hold time: it is a
    tick centred on the hold hour it ran at. Without a hold there are no
    hold hours to draw on, so the
    steps are laid end to end over the time their logs ran (each its last
    stamp plus the step tail, as the clock counts a hold), which shows what
    did run.
    """
    lanes: Dict[str, List[Tuple[float, float]]] = {}
    folder_x: Dict[str, float] = {}
    if not tl.segments:
        t = 0.0
        for f in tl.files:
            folder_x.setdefault(f.folder, t)
            lanes.setdefault(f.role, []).append((t, t + f.length_s / 3600.0))
            t += f.length_s / 3600.0
        total = t if t > 0 else 1.0
        w = 0.002 * total
        return ({r: [(a, max(b, a + w)) for a, b in sp] for r, sp in lanes.items()},
                folder_x, total, False)
    total = tl.hours_tested if tl.hours_tested > 0 else 1.0
    tick = TIMELINE_TICK * total
    lanes['hold'] = [(sg.H_start_h, sg.H_end_h) for sg in tl.segments]
    for f in tl.files:
        folder_x.setdefault(f.folder, f.H_start_h)
        if f.role != 'hold':
            lanes.setdefault(f.role, []).append((f.H_start_h - tick / 2.0, f.H_start_h + tick / 2.0))
    return lanes, folder_x, total, True


def _verdict_lines(v: Optional['Verdict']) -> List[str]:
    """Readout lines for the verdict. Hours to death is the only number;
    the verdict, cause and bracket are text, as the plan's summary row has
    them."""
    if v is None:
        return []
    if not v.dead:
        lines = ['Verdict: alive']
        if not v.ocv_judged or not v.xover_judged:
            lines.append(' and '.join(
                n for n, ok in (('OCV rule', v.ocv_judged), ('crossover rule', v.xover_judged))
                if not ok) + ' not applied (no data)')
        return lines
    lines = ['Verdict: dead', f'Hours to death = {v.hours:.2f} h']
    if v.cycles is not None:
        lines.append(f'RH cycles to death = {v.cycles}')
    lines.append(f'Death cause: {v.cause_text}')
    if v.bracket_text:
        lines.append(f'Death bracket: {v.bracket_text}')
    return lines


def plot_timeline(tl: Timeline, save_path: Optional[str] = None, title: str = '',
                  verdict: Optional['Verdict'] = None):
    """The run as one lane per role: the check that holds were found where
    they should be, and where each characterization fell, before any number
    is trusted. The order of steps within a block is the workbook's
    Manifest.

    The x axis is the durability clock, OCV hold time only, as on every
    other plot (Corey, 13 Oct): holds fill their lane end to end, each
    segment parted from the next, and every other step is a tick at the hold
    hour it ran at, steps at one hour sharing it (timeline_lanes). A hold
    segment too short to see -- one the station ended within seconds -- is a
    dark blue tick at its hour. A run without a hold is drawn over the time
    its step logs ran instead. Folder boundaries are dotted lines, a death a
    red one. Each lane is drawn as a
    single square-wave line (no NaN breaks, which the sidecar writer
    refuses) over a filled band, so a stored copy re-renders as the same
    timing diagram; the hold lane dips to its base between segments, so the
    stored copy parts them too.
    """
    spans, folder_x, total, on_holds = timeline_lanes(tl)
    pad = TIMELINE_TICK * total if on_holds else 0.0
    x_lo, x_hi = -pad, total + pad
    lanes = [ln for ln in _LANES if ln[0] in spans]
    n = max(len(lanes), 1)
    fig, ax = plt.subplots(figsize=(12, 1.6 + 0.42 * n), dpi=120)

    for li, (role, label, color) in enumerate(lanes):
        base = n - 1 - li
        top = base + 0.7
        # Hold segments that abut stay apart; anything else that meets or
        # overlaps is one span.
        apart = on_holds and role == 'hold'
        merged: List[List[float]] = []
        for a, b in sorted(spans[role]):
            a, b = max(a, x_lo), min(b, x_hi)
            if merged and (a < merged[-1][1] if apart else a <= merged[-1][1]):
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        xs, ys = [x_lo], [base]
        for a, b in merged:
            xs += [a, a, b, b]
            ys += [base, top, top, base]
        xs.append(x_hi)
        ys.append(base)
        xs = [round(x, 5) for x in xs]
        ax.fill_between(xs, base, ys, color=color, alpha=0.85, linewidth=0)
        ax.plot(xs, ys, color=color, lw=0.8, label=label)

    # Folder boundaries, named along the top, and hold hours at the end of each
    # hold segment. Placed in the axes' data coordinates but held by the
    # figure: the sidecar keeps every text of an axes as readout, and a stored
    # copy then re-rendered the folder names and hours as a long readout list.
    # Folders that start at the same hold hour (a BOL folder holds no hold
    # time) share one boundary, named in order.
    named: List[Any] = []
    if len(folder_x) > 1:
        at: Dict[float, List[str]] = {}
        for folder, x in folder_x.items():
            at.setdefault(round(x, 6), []).append(folder.rsplit('/', 1)[-1])
        for x, names in at.items():
            if x > 0:
                ax.axvline(x, color='k', lw=0.8, ls=':', alpha=0.6)
            named.append(fig.text(x, n - 0.05, ' ' + ' → '.join(names), transform=ax.transData,
                                  fontsize=7.5, va='bottom', ha='left', color='0.25', rotation=0))
    if on_holds and 'hold' in spans:
        tick = TIMELINE_TICK * total
        for sg in tl.segments:
            if sg.H_end_h - sg.H_start_h < tick:
                a, b = round(sg.H_start_h - tick / 2.0, 5), round(sg.H_start_h + tick / 2.0, 5)
                ax.fill_between([a, b], n - 1, n - 0.3, color=SHORT_HOLD_COLOR, lw=0, zorder=2.5)
                ax.plot([a, a, b, b], [n - 1, n - 0.3, n - 0.3, n - 1], color=SHORT_HOLD_COLOR,
                        lw=0.8, zorder=2.6, label='hold shorter than a tick')

    # Holds abut on this clock, so each segment after the first is parted from
    # the one before by a white line across the hold lane (the figure's too:
    # a stored copy would redraw it as a reference line across every lane);
    # and hold hours mark each segment's end, the last always, the others
    # where they do not crowd it.
    if on_holds and lanes and lanes[0][0] == 'hold':
        for s in tl.segments[1:]:
            fig.add_artist(Line2D([s.H_start_h] * 2, [n - 1, n - 0.3], transform=ax.transData,
                                  color='white', lw=1.2, zorder=3))
        if len(tl.segments) <= 12:
            shown: List[float] = []
            for s in reversed(tl.segments):
                if shown and shown[-1] - s.H_end_h < 0.04 * total:
                    continue
                shown.append(s.H_end_h)
                fig.text(s.H_end_h, n - 1 + 0.78, f'{s.H_end_h:.2f} h', transform=ax.transData,
                         fontsize=7, ha='right', va='bottom', color='#1f77b4')
    if verdict is not None and verdict.dead:
        ax.axvline(round(verdict.hours, 4), color='#d62728', lw=1.6,
                   label=f'death at {verdict.hours:.2f} h')

    ax.set_yticks([n - 1 - i + 0.35 for i in range(len(lanes))])
    ax.set_yticklabels([ln[1] for ln in lanes], fontsize=9)
    ax.set_ylim(-0.3, n + 0.45)
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel('Hold hours (h): OCV hold time only; every other step is a tick at the hold '
                  'hour it ran at' if on_holds else
                  'No OCV hold found, so no hold hours: elapsed time over every step, from the '
                  'step logs (h)')
    ax.grid(True, axis='x', alpha=0.25)
    ax.set_title(title or 'Durability test timeline', fontsize=11, fontweight='bold', loc='left')
    _readout(ax, [f'Hours tested = {tl.hours_tested:.2f} h',
                  f'RH cycles tested = {tl.cycles_tested}' if tl.cycles_tested else None,
                  f'Hold files = {len(tl.holds)}',
                  f'Hold segments = {len(tl.segments)}'] + _verdict_lines(verdict))
    plt.tight_layout()
    # A folder named near the end of the axis would run off it: it is named
    # back from its boundary instead.
    renderer = fig.canvas.get_renderer()
    right = ax.get_window_extent(renderer).x1
    for txt in named:
        if txt.get_window_extent(renderer).x1 > right:
            txt.set_ha('right')
            txt.set_text(txt.get_text().strip() + ' ')
    if save_path:
        save_with_sidecar(fig, save_path, plot_type='fcdur_timeline', bbox_inches='tight')
    return fig


_COND_COLORS = ('#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#17becf',
                '#e377c2', '#7f7f7f')
_CHAR_ROLES = ('crossover', 'ecsa', 'eis', 'polcurve')


def _kv(key: str, x: Optional[float], spec: str, unit: str) -> Optional[str]:
    return f'{key} = {x:{spec}} {unit}' if x is not None and math.isfinite(x) else None


def _series(ax, x, y, **style):
    """ax.plot, except for two points level in x or y: the sidecar writer
    stores such a line as a reference line across the whole axes, so it is
    drawn as two markers instead. A single point gets a marker, or nothing
    would show (an OCV plot bin as long as a hold leaves it one)."""
    if len(x) == 2 and (x[0] == x[1] or y[0] == y[1]):
        ax.scatter(x, y, s=10, color=style.get('color'), label=style.get('label'),
                   alpha=style.get('alpha'), zorder=3)
    else:
        if len(x) == 1:
            style = {'marker': 'o', 'ms': 3, **style}
        ax.plot(x, y, **style)


def plot_ocv(tl: Timeline, res: OcvResult, save_path: Optional[str] = None, title: str = ''):
    """Hold OCV against hold hours, holds joined, over its rolling decay rate.

    Each hold is drawn from bins of its judged samples: the bin median, over a
    band shaded from each bin's lowest to its highest sample, so short dips
    and spikes survive the binning. The sidecar writer keeps lines but not
    fills, so the band's edges are also drawn, as faint lines labelled
    BAND_LOW and BAND_HIGH, which the portal shades between. Settled start and
    end values and dips are markers (scatters, so a pair of them never reads
    back as a reference line), and the threshold, characterization hours and
    the death trip are reference lines. Every plotted array is finite, since
    the sidecar writer refuses NaN.
    """
    s = res.settings
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(12, 7.5), dpi=120, sharex=True,
                                  gridspec_kw={'height_ratios': [3, 1]})
    colors: Dict[Tuple[Any, ...], str] = {}
    for h in res.holds:
        colors.setdefault(h.conditions_key, _COND_COLORS[len(colors) % len(_COND_COLORS)])

    ys: List[float] = [s.threshold_v]
    banded = False
    for h in res.holds:
        if h.plot is None or h.plot.empty:
            continue
        c = colors[h.conditions_key]
        x = h.plot['t_h'].round(4).to_numpy()
        low = h.plot['min'].round(5).to_numpy()
        high = h.plot['max'].round(5).to_numpy()
        mid = h.plot['median'].round(5).to_numpy()
        ax.fill_between(x, low, high, color=c, alpha=BAND_ALPHA, lw=0, zorder=1.2)
        # Every hold's edges carry their labels, so the stored copy pairs
        # them hold by hold; the legend below shows one key for the band.
        _series(ax, x, low, color=c, lw=0.4, alpha=0.3, label=BAND_LOW, zorder=1.3)
        _series(ax, x, high, color=c, lw=0.4, alpha=0.3, label=BAND_HIGH, zorder=1.3)
        # Every hold's median carries its conditions, not only the first: a
        # stored copy is read line by line (Comparisons draws each condition
        # as one series), and the legend below names each condition once.
        _series(ax, x, mid, color=c, lw=1.4, label=h.segment.conditions)
        banded = True
        ys += [float(low.min()), float(high.max())]

    marked = False
    for key, c in colors.items():
        pts = sorted([(h.start_at_h, h.start_v) for h in res.holds
                      if h.conditions_key == key and math.isfinite(h.start_v)]
                     + [(h.end_at_h, h.end_v) for h in res.holds
                        if h.conditions_key == key and math.isfinite(h.end_v)])
        if pts:
            ax.scatter([round(p[0], 4) for p in pts], [round(p[1], 5) for p in pts], s=34,
                       facecolors='white', edgecolors=c, linewidths=1.4, zorder=4,
                       label='settled start and end (30-min medians)' if not marked else '_nolegend_')
            marked = True

    ax.axhline(s.threshold_v, color='#d62728', ls='--', lw=1.1,
               label=f'{s.threshold_v:g} V death threshold')
    char_h = sorted({round(f.H_start_h, 4) for f in tl.files if f.role in _CHAR_ROLES})
    for j, xh in enumerate(char_h):
        ax.axvline(xh, color='0.55', ls=':', lw=1.0,
                   label='characterization' if j == 0 else '_nolegend_')
    dips = [r for r in res.runs if r.verdict == 'dip']
    if dips:
        ax.scatter([round(r.start_h, 4) for r in dips], [round(r.lowest_v, 5) for r in dips],
                   marker='v', s=36, color='#d62728', zorder=4,
                   label=f'dip under {s.threshold_v:g} V, shorter than {s.persist_min:g} min')
        ys += [r.lowest_v for r in dips]
    if res.death:
        how = ', the station\'s own limit' if res.death.trigger == 'station' else ''
        ax.axvline(round(res.death.start_h, 4), color='#d62728', lw=1.6,
                   label=f'OCV death at {res.death.start_h:.2f} h{how}')

    lo, hi = min(ys), max(ys)
    pad = max(0.01, 0.06 * (hi - lo))
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(0, max(tl.hours_tested, 1e-3))
    ax.set_ylabel('Hold OCV (V)')
    ax.grid(True, alpha=0.25)
    ax.set_title(title or 'Hold OCV', fontsize=11, fontweight='bold', loc='left')
    first: Dict[str, Any] = {}
    for handle, label in zip(*ax.get_legend_handles_labels()):
        if label not in (BAND_LOW, BAND_HIGH):
            first.setdefault(label, handle)
    if banded:
        # One neutral key for the bands, which take each condition's colour.
        first[f'high–low of the samples in each {res.plot_bin_s / 60:.4g}-min bin'] = Patch(
            facecolor='0.6', alpha=0.35, edgecolor='0.6', lw=0.4)
    ax.legend(list(first.values()), list(first), loc='best', fontsize=8, framealpha=0.9)
    lines = [_kv('OCV initial', res.initial_v, '.4f', 'V'),
             _kv('OCV final', res.final_v, '.4f', 'V'),
             _kv('OCV loss', res.loss_pct, '.2f', '%'),
             _kv('OCV decay rate', res.decay_uV_h, '.0f', 'μV/h'),
             _kv('OCV min (10-min median)', res.lowest_v, '.4f', 'V'),
             (_kv('OCV death', res.death_h, '.2f', 'h') if res.death
              else 'OCV death: none' if res.judged else 'OCV death: not judged'),
             _kv('Hours tested', tl.hours_tested, '.2f', 'h'),
             f'RH cycles tested = {tl.cycles_tested}' if tl.cycles_tested else None,
             _kv('OCV plot bin', res.plot_bin_s / 60.0, '.4g', 'min')]
    if len(colors) > 1:
        lines.append(f'Initial, final, loss and rate at\n{res.reference}')
    _readout(ax, lines, x=1.01, y=1.0, ha='left', va='top')

    for h in res.holds:
        rated = [r for r in h.rates if math.isfinite(r['decay'])]
        if rated:
            x = [round((r['start'] + r['end']) / 2.0, 4) for r in rated]
            y = [round(r['decay'], 1) for r in rated]
            if len(rated) >= 3:
                axr.plot(x, y, color=colors[h.conditions_key], marker='o', ms=3, lw=1.0)
            else:
                axr.scatter(x, y, s=12, color=colors[h.conditions_key], zorder=3)
    axr.axhline(0.0, color='0.5', lw=0.8)
    axr.set_ylabel(f'OCV decay (μV/h)\n{s.rate_window_h:g} h windows')
    axr.set_xlabel('Hold hours (h)')
    axr.grid(True, alpha=0.25)
    plt.tight_layout()
    if save_path:
        save_with_sidecar(fig, save_path, plot_type='fcdur_ocv_vs_time', bbox_inches='tight')
    return fig


def _finite_xy(x, y) -> Tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def _series_style(ca: CharAnalysis) -> Dict[str, str]:
    """One colour per series, the 100% RH crossover first."""
    order = ca.death_series + [s for s in ca.series if s not in ca.death_series]
    return {s: _COND_COLORS[i % len(_COND_COLORS)] for i, s in enumerate(order)}


def _plot_trend(ax, ca: CharAnalysis, series: str, key: str, color: str,
                label: Optional[str] = None, repeats: bool = True, **line) -> List[float]:
    """Block values of one series against hold hours as a line, and every
    other good repeat as hollow points beside it. Returns the y values drawn."""
    blocks = [r for r in ca.blocks(series) if math.isfinite(r.values.get(key, math.nan))]
    x, y = _finite_xy([r.H for r in blocks], [r.values[key] for r in blocks])
    if len(x):
        style = dict(color=color, marker='o', ms=6, lw=1.4, label=label or series)
        style.update(line)
        _series(ax, np.round(x, 4), np.round(y, 6), **style)
    reps = [r for r in ca.results if repeats and r.series == series and not r.error
            and not r.block and math.isfinite(r.values.get(key, math.nan))]
    rx, ry = _finite_xy([r.H for r in reps], [r.values[key] for r in reps])
    if len(rx):
        ax.scatter(np.round(rx, 4), np.round(ry, 6), s=30, facecolors='none', edgecolors=color,
                   linewidths=1.2, zorder=3, label=f'{series}, earlier repeats')
    return list(y) + list(ry)


def _time_axis(ax, tl: Timeline, ca: CharAnalysis):
    span = max(tl.hours_tested, max((iv.H for iv in ca.intervals), default=0.0), 1e-3)
    ax.set_xlim(-0.03 * span, 1.03 * span)
    ax.set_xlabel('Hold hours (h)')
    ax.grid(True, alpha=0.25)


def _first_last(ca: CharAnalysis, series: str, key: str) -> Tuple[Optional[CharResult],
                                                                    Optional[CharResult]]:
    blocks = [r for r in ca.blocks(series) if math.isfinite(r.values.get(key, math.nan))]
    return (blocks[0], blocks[-1]) if blocks else (None, None)


def plot_xover_vs_time(tl: Timeline, ca: CharAnalysis, save_path: Optional[str] = None,
                       title: str = ''):
    """Crossover against hold hours: the 100% RH window mean the rule judges,
    its short-corrected intercept, other RH series, the death threshold and
    the DOE target, and R_short on a log twin axis."""
    cs = ca.settings
    colors = _series_style(ca)
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=120)
    ys: List[float] = [cs.xover_threshold, DOE_XOVER]
    for s in [s for s in ca.series if ca.series_kind[s] == 'crossover']:
        ys += _plot_trend(ax, ca, s, 'j_xover', colors[s], label=f'{s}, window mean')
        if s in ca.death_series:
            ys += _plot_trend(ax, ca, s, 'j_intercept', '0.4', label=f'{s}, short-corrected',
                              repeats=False, ls='--', marker='s', ms=5, mfc='white', lw=1.0)
    ax.axhline(cs.xover_threshold, color='#d62728', ls='--', lw=1.1,
               label=f'{cs.xover_threshold:g} mA/cm² death threshold')
    ax.axhline(DOE_XOVER, color='0.45', ls=':', lw=1.1, label=f'DOE target {DOE_XOVER:g} mA/cm²')
    xd = ca.xover_death
    if xd:
        ax.axvline(round(xd.H, 4), color='#d62728', lw=1.6,
                   label=f'crossover death at {xd.interval.label}')
    ax.set_ylim(0, max(ys) * 1.12)
    ax.set_ylabel('H₂ crossover (mA/cm²)')
    _time_axis(ax, tl, ca)

    shorts = [r for s in ca.death_series for r in ca.blocks(s)
              if math.isfinite(r.values.get('R_short', math.nan))]
    if shorts:
        tw = ax.twinx()
        tw.scatter(np.round([r.H for r in shorts], 4), [round(r.values['R_short'], 1) for r in shorts],
                   marker='D', s=28, color='0.35', zorder=3, label='R_short')
        tw.axhline(DOE_R_SHORT, color='0.35', ls='-.', lw=0.9)
        tw.set_yscale('log')
        tw.set_ylabel('R_short (Ω·cm²), ◆; DOE target ≥ 1,000 (dash-dot)', color='0.35')
        lo = min(min(r.values['R_short'] for r in shorts), DOE_R_SHORT)
        hi = max(max(r.values['R_short'] for r in shorts), DOE_R_SHORT)
        tw.set_ylim(lo / 3, hi * 3)

    lines = []
    judged = ca.death_blocks()
    if judged:
        first, last = judged[0], judged[-1]
        lines += [f'j_xover initial = {first.values["j_xover"]:.3f} mA/cm²',
                  f'j_xover final = {last.values["j_xover"]:.3f} mA/cm²']
        if math.isfinite(last.values['R_short']):
            lines.append(f'R_short final = {last.values["R_short"]:.0f} Ω·cm²')
    lines.append(f'Crossover death = {xd.H:.2f} h' if xd else
                 'Crossover death: none' if ca.xover_judged else 'Crossover death: not judged')
    lines.append(f'DOE crossover breach = {ca.doe_xover.H:.2f} h' if ca.doe_xover
                 else 'DOE crossover breach: none')
    lines.append(f'DOE R_short breach = {ca.doe_r_short.H:.2f} h' if ca.doe_r_short
                 else 'DOE R_short breach: none')
    _readout(ax, lines, x=1.0, y=1.02)
    ax.set_title(title or 'H₂ crossover', fontsize=11, fontweight='bold', loc='left')
    # Below the axes: inside, it would hide R_short on the twin axis.
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=8,
              framealpha=0.9)
    plt.tight_layout()
    if save_path:
        save_with_sidecar(fig, save_path, plot_type='fcdur_xover_vs_time', bbox_inches='tight')
    return fig


def plot_metric_vs_time(tl: Timeline, ca: CharAnalysis, kind: str, key: str, ylabel: str,
                        name: str, unit: str, spec: str, save_path: Optional[str] = None,
                        title: str = '', plot_type: str = '', loss_name: str = '',
                        also: Optional[Tuple[str, str, str, str]] = None):
    """One measurement's block values (and repeats) against hold hours, one
    line per series, with its first and last value in the readout. `also`
    (key, name, unit, spec) adds a second value of the same blocks to the
    readout -- ECSA in m²/g beside the roughness factor."""
    colors = _series_style(ca)
    fig, ax = plt.subplots(figsize=(11, 5), dpi=120)
    ys: List[float] = []
    mine = [s for s in ca.series if ca.series_kind[s] == kind]
    for s in mine:
        ys += _plot_trend(ax, ca, s, key, colors[s])
    if ys:
        lo, hi = min(ys), max(ys)
        pad = max(0.08 * (hi - lo), 0.05 * abs(hi), 1e-6)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_ylabel(ylabel)
    _time_axis(ax, tl, ca)
    lines = []
    if mine:
        first, last = _first_last(ca, mine[0], key)
        if first:
            u = f' {unit}' if unit else ''
            lines = [f'{name} initial = {first.values[key]:{spec}}{u}',
                     f'{name} final = {last.values[key]:{spec}}{u}']
            if loss_name and first.values[key]:
                lines.append(f'{loss_name} = {(1 - last.values[key] / first.values[key]) * 100:.1f} %')
            if also:
                k2, n2, u2, s2 = also
                if math.isfinite(first.values.get(k2, math.nan)):
                    lines += [f'{n2} initial = {first.values[k2]:{s2}} {u2}',
                              f'{n2} final = {last.values[k2]:{s2}} {u2}']
            lines.append(f'Series {mine[0]}')
    _readout(ax, lines, x=1.0, y=1.02)
    ax.set_title(title, fontsize=11, fontweight='bold', loc='left')
    if mine:
        ax.legend(loc='best', fontsize=8, framealpha=0.9)
    plt.tight_layout()
    if save_path:
        save_with_sidecar(fig, save_path, plot_type=plot_type, bbox_inches='tight')
    return fig


def plot_perf_vs_time(tl: Timeline, ca: CharAnalysis, save_path: Optional[str] = None,
                      title: str = ''):
    """Pol-curve performance against hold hours: V @ 1 A/cm² above, j @ 0.65 V
    below, one line per flow setting."""
    colors = _series_style(ca)
    mine = [s for s in ca.series if ca.series_kind[s] == 'polcurve']
    fig, (ax, axj) = plt.subplots(2, 1, figsize=(11, 7), dpi=120, sharex=True)
    for target, key, ylabel in ((ax, 'V_1A', 'V @ 1 A/cm² (V)'),
                                (axj, 'j_0.65V', 'j @ 0.65 V (A/cm²)')):
        ys: List[float] = []
        for s in mine:
            ys += _plot_trend(target, ca, s, key, colors[s])
        if ys:
            lo, hi = min(ys), max(ys)
            pad = max(0.08 * (hi - lo), 0.02 * abs(hi), 1e-6)
            target.set_ylim(lo - pad, hi + pad)
        target.set_ylabel(ylabel)
        _time_axis(target, tl, ca)
    ax.set_xlabel('')
    lines = []
    for s in mine:
        tag = s.replace('Pol curve, ', '', 1)     # RH, gases and flows: unique per series
        blocks = ca.blocks(s)
        if not blocks:
            continue
        for key, name, unit, spec in (('V_1A', 'V @ 1 A/cm²', 'V', '.4f'),
                                      ('j_0.65V', 'j @ 0.65 V', 'A/cm²', '.4f')):
            # The first and last curves themselves: a curve that no longer
            # reaches 1 A/cm² is a result, not a gap to step over.
            for when, r in (('initial', blocks[0]), ('final', blocks[-1])):
                x = r.values.get(key, math.nan)
                lines.append(f'{name} {when}, {tag} = {x:{spec}} {unit}' if math.isfinite(x)
                             else f'{name} {when}, {tag}: not reached')
    _readout(ax, lines, x=1.0, y=1.02)
    ax.set_title(title or 'Pol-curve performance', fontsize=11, fontweight='bold', loc='left')
    if mine:
        ax.legend(loc='best', fontsize=8, framealpha=0.9)
    plt.tight_layout()
    if save_path:
        save_with_sidecar(fig, save_path, plot_type='fcdur_perf_vs_time', bbox_inches='tight')
    return fig


# The stored hours-to-cycles map keeps every this many cycle ends of a hold,
# besides its start and end, its first and last cycle end, and the cycle ends
# either side of a pause longer than RH_MAP_PAUSE cycles; read off linearly
# between them, it is then within a cycle of the count.
RH_MAP_EVERY = 25
RH_MAP_PAUSE = 2.0


def rh_cycle_map(tl: Timeline) -> Tuple[np.ndarray, np.ndarray]:
    """RH cycles completed against hold hours: each hold's start and end,
    its first and last cycle end, every RH_MAP_EVERY-th between, and both
    ends of any pause in the cycling. It is what Comparisons plots the
    durability charts against cycles with, reading it off linearly; a hold
    at a steady RH, or a pause, is a flat stretch."""
    xs: List[float] = [0.0]
    ys: List[int] = [0]
    period = tl.rh_period_s
    for f in tl.holds:
        xs.append(f.H_start_h)
        ys.append(f.C_start)
        ends = f.rh_ends_s
        if ends is not None and len(ends):
            keep = set(range(RH_MAP_EVERY - 1, len(ends), RH_MAP_EVERY)) | {0, len(ends) - 1}
            if period:
                for i in np.flatnonzero(np.diff(ends) > RH_MAP_PAUSE * period):
                    keep |= {int(i), int(i) + 1}
            for i in sorted(keep):
                xs.append(f.H_start_h + float(ends[i]) / 3600.0)
                ys.append(f.C_start + i + 1)
        xs.append(f.H_end_h)
        ys.append(f.C_end)
    x, y = np.round(np.asarray(xs, float), 5), np.asarray(ys, float)
    keep = np.r_[True, (np.diff(x) > 0) | (np.diff(y) != 0)]
    return x[keep], y[keep]


def plot_rh_cycles(tl: Timeline, verdict: Optional[Verdict] = None,
                   save_path: Optional[str] = None, title: str = ''):
    """RH cycles completed against hold hours (rh_cycle_map), with the
    characterizations and the death. Its stored copy is the map Comparisons
    reads to plot every durability chart against RH cycles."""
    x, y = rh_cycle_map(tl)
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=120)
    ax.plot(x, y, color='#1f77b4', lw=1.4, label='RH cycles completed')
    char_h = sorted({round(f.H_start_h, 4) for f in tl.files if f.role in _CHAR_ROLES})
    for j, xh in enumerate(char_h):
        ax.axvline(xh, color='0.55', ls=':', lw=1.0,
                   label='characterization' if j == 0 else '_nolegend_')
    if verdict is not None and verdict.dead:
        ax.axvline(round(verdict.hours, 4), color='#d62728', lw=1.6,
                   label=f'death at {verdict.hours:.2f} h, RH cycle {verdict.cycles}')
    ax.set_xlim(0, max(tl.hours_tested, 1e-3))
    ax.set_ylim(0, max(tl.cycles_tested, 1) * 1.05)
    ax.set_xlabel('Hold hours (h)')
    ax.set_ylabel('RH cycles')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.set_title(title or 'RH cycles', fontsize=11, fontweight='bold', loc='left')
    _readout(ax, [f'RH cycles tested = {tl.cycles_tested}',
                  _kv('RH cycle', tl.rh_period_s, '.4g', 's') if tl.rh_steady
                  else 'RH cycle: no steady pace' if tl.rh_period_s is not None else None,
                  _kv('Hours tested', tl.hours_tested, '.2f', 'h'),
                  f'RH cycles to death = {verdict.cycles}'
                  if verdict is not None and verdict.dead and verdict.cycles is not None else None],
             x=1.0, y=1.02)
    plt.tight_layout()
    if save_path:
        save_with_sidecar(fig, save_path, plot_type='fcdur_rh_cycles', bbox_inches='tight')
    return fig


def _pick_curves(n: int, k: int = EVOLUTION_MAX_CURVES) -> List[int]:
    """Indices of at most k of n curves: the first, the last, and evenly
    between."""
    if n <= k:
        return list(range(n))
    return sorted(set(int(i) for i in np.linspace(0, n - 1, k).round()))


def _thin(x, y, n: int = EVOLUTION_MAX_POINTS) -> Tuple[np.ndarray, np.ndarray]:
    x, y = _finite_xy(x, y)
    if len(x) > n:
        idx = np.unique(np.linspace(0, len(x) - 1, n).round().astype(int))
        x, y = x[idx], y[idx]
    return np.round(x, 6), np.round(y, 6)


EVOLUTION = {   # kind -> (plot type, title, x label, y label)
    'polcurve': ('fcdur_polcurve_evolution', 'Pol curves', 'j (A/cm²)', 'V (V)'),
    'crossover': ('fcdur_xover_evolution', 'Crossover CV, anodic and cathodic averaged',
                  'V (V)', 'j (mA/cm²)'),
    'eis': ('fcdur_nyquist_evolution', 'Nyquist', "Z' (mΩ·cm²)", "−Z'' (mΩ·cm²)"),
    'ecsa': ('fcdur_ecsa_evolution', 'ECSA CV, cycle analyzed', 'V (V)', 'j (mA/cm²)'),
}


def plot_evolution(ca: CharAnalysis, kind: str, save_path: Optional[str] = None, title: str = ''):
    """One panel per series of a measurement (at most two), one curve per
    characterization (BOL, then by hold hours; at most four), each thinned
    to EVOLUTION_POINTS[kind] points."""
    plot_type, name, xl, yl = EVOLUTION[kind]
    mine = [s for s in ca.series if ca.series_kind[s] == kind and ca.blocks(s)]
    mine = mine[:EVOLUTION_MAX_SERIES]
    n = max(len(mine), 1)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n + 0.8, 4.6), dpi=120, squeeze=False)
    for ax, s in zip(axes[0], mine):
        blocks = [r for r in ca.blocks(s) if r.curve is not None]
        picked = [blocks[i] for i in _pick_curves(len(blocks))]
        cmap = plt.get_cmap('viridis')
        arc_top = 0.0
        for i, r in enumerate(picked):
            x, y = _thin(*r.curve, n=EVOLUTION_POINTS[kind])
            if len(x) >= 3:
                color = cmap(0.85 * i / max(len(picked) - 1, 1))
                ax.plot(x, y, color=color, lw=1.3, label=f'{ca.label(r.interval)} ({r.file.step})')
                if kind == 'eis':
                    arc_top = max(arc_top, float(y.max()))
                    ax.scatter([round(r.values['HFR'], 3)], [0.0], marker='o', s=42,
                               facecolors='white', edgecolors=color, linewidths=1.6, zorder=4,
                               label=f'HFR {r.values["HFR"]:.1f}')
        if kind == 'eis' and arc_top > 0:
            # The inductive tail below the axis runs far down at high
            # frequency; the view keeps the arc and the HFR crossing.
            ax.set_ylim(-0.3 * arc_top, 1.15 * arc_top)
        if kind == 'crossover':
            for j, v in enumerate((ca.settings.xover_lo, ca.settings.xover_hi)):
                ax.axvline(v, color='0.5', ls=':', lw=1.0,
                           label='averaging window' if j == 0 else '_nolegend_')
        ax.set_title(s, fontsize=9.5)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7.5, loc='best')
    fig.suptitle(title or name, fontsize=11, fontweight='bold', x=0.01, ha='left')
    plt.tight_layout()
    if save_path:
        save_with_sidecar(fig, save_path, plot_type=plot_type, bbox_inches='tight')
    return fig


def _breach(label: Optional[str]) -> str:
    return label if label else 'none'


def verdict_rows(v: Verdict, ocv: Optional[OcvResult],
                 ca: Optional[CharAnalysis]) -> List[Tuple[str, Any]]:
    """The verdict block that heads the Summary sheet."""
    xd = ca.xover_death if ca is not None else None
    if ocv is None or not ocv.judged:
        ocv_rule = 'not judged'
    else:
        ocv_rule = f'death at {ocv.death_h:.2f} h' if ocv.death else 'no death'
    if ca is None or not ca.xover_judged:
        xover_rule = 'not judged'
    elif xd:
        xover_rule = (f'death at {xd.interval.label}: {xd.result.values["j_xover"]:.2f} mA/cm² '
                      f'({xd.result.file.step}), previous crossover '
                      f'{xd.previous.label if xd.previous else "none"}')
    else:
        xover_rule = 'no death'
    rows = [
        ('Verdict', v.text),
        ('Hours to death (h)', round(v.hours, 4) if v.dead else ''),
    ]
    if v.cycles is not None:
        rows.append(('RH cycles to death', v.cycles if v.dead else ''))
    return rows + [
        ('Death cause', v.cause_text),
        ('Death bracket', v.bracket_text),
        ('OCV rule', ocv_rule),
        ('Crossover rule', xover_rule),
    ]


def char_summary_rows(ca: CharAnalysis, ocv: Optional[OcvResult]) -> List[Tuple[str, Any]]:
    cs = ca.settings
    judged = ca.death_blocks()
    first, last = (judged[0], judged[-1]) if judged else (None, None)
    loss_h = ocv.loss_breach_h if ocv is not None else None
    return [
        ('Characterizations', ', '.join(iv.label for iv in ca.intervals) or 'none'),
        ('j_xover initial (mA/cm²)', _r(first.values['j_xover'], 3) if first else None),
        ('j_xover final (mA/cm²)', _r(last.values['j_xover'], 3) if last else None),
        ('R_short final (Ω·cm²)', _r(last.values['R_short'], 0) if last else None),
        ('Crossover rule settings', f'{cs.death_rh:g}% RH crossover, window mean of the averaged '
                                    f'CV current over {cs.xover_lo:g}–{cs.xover_hi:g} V, above '
                                    f'{cs.xover_threshold:g} mA/cm²; the last repeat in a block '
                                    f'represents it'),
        (f'DOE crossover ≤ {DOE_XOVER:g} mA/cm², first breach', _breach(
            ca.doe_xover.label if ca.doe_xover else None)),
        (f'DOE OCV loss ≤ {DOE_OCV_LOSS_PCT:g}%, first breach',
         _breach(_hours_label(loss_h) if loss_h is not None else None)),
        (f'DOE R_short ≥ {DOE_R_SHORT:,.0f} Ω·cm², first breach', _breach(
            ca.doe_r_short.label if ca.doe_r_short else None)),
        ('Geometric area and loading used', f'{cs.geo_area:g} cm², {cs.loading:g} mg Pt/cm²'),
    ]


def cycle_rows(ca: CharAnalysis, tl: Optional[Timeline] = None) -> List[Dict[str, Any]]:
    """The Cycles sheet: one row per characterization, each measurement's
    block value across; with RH cycling, the RH cycles before it too."""
    skip = {'fit_R2', 'scan_rate', 'cycle', 'Q_H'}
    rows = []
    for iv in ca.intervals:
        row: Dict[str, Any] = {'Characterization': iv.label, 'Interval': iv.index,
                               'Hold hours (h)': round(iv.H, 4)}
        if tl is not None and tl.cycles_tested:
            row['RH cycles before'] = tl.cycles_at(iv.H)
        for s in ca.series:
            blk = ca.block(s, iv.index)
            for key, heading, nd in METRICS[ca.series_kind[s]]:
                if key not in skip:
                    row[f'{s}: {heading}'] = _r(blk.values.get(key), nd) if blk else None
        row['Notes'] = '; '.join(n.split(': ', 1)[1] for n in ca.notes
                                 if n.startswith(iv.label + ': '))
        rows.append(row)
    return rows


CHAR_COLUMNS = (['Characterization', 'Interval', 'Hold hours (h)', 'Step', 'Kind', 'Series',
                 'Represents block']
                + [heading for kind in CHAR_KINDS for _, heading, _ in METRICS[kind]]
                + ['Note', 'Error', 'File'])


def characterization_rows(ca: CharAnalysis, tl: Optional[Timeline] = None) -> pd.DataFrame:
    """The Characterizations sheet: every file, repeats included; with RH
    cycling, the RH cycles before each too."""
    cycled = tl is not None and tl.cycles_tested
    columns = list(CHAR_COLUMNS)
    if cycled:
        columns.insert(columns.index('Hold hours (h)') + 1, 'RH cycles before')
    rows = []
    for r in ca.results:
        row: Dict[str, Any] = {'Characterization': ca.label(r.interval), 'Interval': r.interval,
                               'Hold hours (h)': round(r.H, 4), 'Step': r.file.step,
                               'Kind': KIND_LABELS[r.kind], 'Series': r.series,
                               'Represents block': 'yes' if r.block else ''}
        if cycled:
            row['RH cycles before'] = r.file.C_start
        for key, heading, nd in METRICS[r.kind]:
            row[heading] = _r(r.values.get(key), nd)
        row['Note'] = r.note
        row['Error'] = r.error
        row['File'] = r.file.name
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def step_formats(tl: Timeline) -> str:
    """Which formats the run's step files came in."""
    kinds = [('csv', 'station CSV'), ('fcd', 'Scribner FuelCell (.fcd)')]
    present = {f.source for f in tl.files}
    return ' and '.join(label for key, label in kinds if key in present) or 'none'


def step_tail_text(tl: Timeline) -> Any:
    """The tail the clock adds per hold file: the station CSVs' 30 s; none
    for a .fcd step, which ends with its last sample. A run without holds
    says what its files would have had."""
    sources = {f.source for f in tl.holds} or {f.source for f in tl.files}
    if sources == {'fcd'}:
        return '0 (.fcd steps end with their last sample)'
    if sources == {'csv', 'fcd'}:
        return f'{tl.step_tail_s:g} (station CSVs); 0 (.fcd)'
    return tl.step_tail_s


def _uncounted(f: StepFile) -> bool:
    """A hold whose RH cycles could not be counted: unreadable (rh_unread), or
    named for RH cycling with no cycle seen in it."""
    return f.rh_unread or ('↔' in f.rh_text and f.rh_ends_s is None)


def _rh_uncounted(tl: Timeline) -> Tuple[List[str], List[str]]:
    """The holds whose RH cycles could not be counted (_uncounted), by why:
    unreadable (Wet flags or an OCV that could not be read, or an OCV that
    swings but whose cycles could not be counted); named for RH cycling, with
    no cycle seen in them. Steps once each, in run order."""
    def steps(fs: List[StepFile]) -> List[str]:
        return list(dict.fromkeys(f.step for f in fs))
    unread = [f for f in tl.holds if f.rh_unread]
    cycling = [f for f in tl.holds if _uncounted(f) and not f.rh_unread]
    return steps(unread), steps(cycling)


def _step_list(steps: List[str], most: int = 6) -> str:
    shown = ', '.join(steps[:most])
    return shown + (f' and {len(steps) - most} more' if len(steps) > most else '')


def rh_cycles_uncounted(tl: Timeline) -> Optional[str]:
    """The holds whose RH cycles could not be counted and why, for the
    Summary; None when there are none."""
    unread, cycling = _rh_uncounted(tl)
    parts = []
    if unread:
        parts.append(f'{_step_list(unread)}: unreadable (see the notes)')
    if cycling:
        parts.append(f'{_step_list(cycling)}: named for RH cycling, but no cycle is seen in '
                     f'{"them" if len(cycling) > 1 else "it"}')
    return '; '.join(parts) or None


def rh_cycles_absent(tl: Timeline) -> Optional[str]:
    """Why a test with holds counts no RH cycles, for its Summary; None when
    it counts some, or has no holds. A .fcd hold's cycles are its Wet flag's
    falls; a hold that logs none has them read off its OCV
    (ocv_rh_cycle_ends)."""
    if tl.cycles_tested or not tl.holds:
        return None
    uncounted = rh_cycles_uncounted(tl)
    if uncounted:
        rest = any(not _uncounted(f) for f in tl.holds)
        return 'none counted: ' + uncounted + ('; the other holds give none' if rest else '')
    flagged = [f.wet_logged for f in tl.holds]
    ocv = f'the OCV {OCV_RH_STEADY}'
    if all(flagged):
        return 'none: the Wet flags never fall from wet to dry in a hold (RH held steady)'
    if not any(flagged):
        return f'none: no Wet flag is logged, and {ocv} (RH held steady)'
    return (f'none: the Wet flags never fall from wet to dry, and where none is logged, {ocv} '
            f'(RH held steady)')


RH_RULE_FLAG = ("every fall from wet to dry of the station's Wet flag (the one that falls most, "
                "the cathode's on a tie)")
RH_RULE_OCV = (f"every rise of the OCV from its low (wet) level to its high (dry) one, where it "
               f"swings {OCV_RH_SWING_MIN_V * 1000:g} mV or more at a steady pace for "
               f"{OCV_RH_MIN_CYCLES} cycles or more, logged every {OCV_RH_LOG_S:g} s or finer")


def rh_cycle_rule(tl: Timeline) -> str:
    """How a test's RH cycles were counted: from the Wet flags, off the OCV
    of holds that log none, or both."""
    flag = any(f.rh_from == 'Wet flag' for f in tl.holds)
    ocv = any(f.rh_from.startswith('OCV') for f in tl.holds)
    if flag and ocv:
        how = f'{RH_RULE_FLAG}, in a hold that logs one; in a hold that logs none, {RH_RULE_OCV}'
    elif ocv:
        how = f'{RH_RULE_OCV}, in a hold that logs no Wet flag'
    else:
        how = f'{RH_RULE_FLAG}, in any hold'
    return f'a cycle counts once its wet half is over: {how}; cycles, like hours, count in OCV holds only'


def export_workbook(tl: Timeline, path: str, params: Dict[str, Any],
                    ocv: Optional[OcvResult] = None, ca: Optional[CharAnalysis] = None,
                    verdict: Optional[Verdict] = None) -> None:
    """Summary (verdict first), Cycles, Holds and Manifest sheets, the OCV
    sheets when there are holds, Characterizations, and Notes.

    The OCV trace is kept as 1-min bins (median, minimum, maximum) rather than
    every sample: 150 h at 1 s is 540,000 rows, and a run past 291 h would
    not fit a sheet. The step files hold the samples.
    """
    summary = [('Hours tested (h)', round(tl.hours_tested, 4)),
               ('Time counted', 'OCV hold time only: characterizations and every other step '
                                'between holds add none')]
    if tl.cycles_tested:
        summary += [
            ('RH cycles tested', tl.cycles_tested),
            ('RH cycle (s)', _r(tl.rh_period_s, 3) if tl.rh_steady or tl.rh_period_s is None
             else f'{tl.rh_period_s:.4g}, the median: the cycles keep no steady pace'),
            ('RH cycle rule', rh_cycle_rule(tl)),
        ]
    absent = rh_cycles_absent(tl)
    if absent:
        summary.append(('RH cycles tested', absent))
    elif tl.cycles_tested and rh_cycles_uncounted(tl):
        summary.append(('RH cycles not counted in', rh_cycles_uncounted(tl)))
    summary += [
        ('Hold files', len(tl.holds)),
        ('Hold segments', len(tl.segments)),
        ('Step files', len(tl.files)),
        ('Folders, in order', ' → '.join(tl.folders)),
        ('Folders left out at upload', ', '.join(tl.left_out) or 'none'),
        ('Hold rule', f'OCV, air or O2 cathode, running ≥ {tl.hold_min_h:g} h'),
        ('Step file format', step_formats(tl)),
        ('Step tail added per hold file (s)', step_tail_text(tl)),
        ('Geometric area (cm²)', params.get('geo_area')),
        ('Pt loading (mg/cm²)', params.get('loading')),
        ('Ignored files', ', '.join(f'{v} {k}' for k, v in sorted(tl.ignored.items())) or 'none'),
    ]
    has_ocv = ocv is not None and bool(ocv.holds)
    if has_ocv:
        s, d = ocv.settings, ocv.death
        a, b = ocv.decay_window_h
        summary += [
            ('OCV reference conditions', ocv.reference),
            ('OCV initial (V)', _r(ocv.initial_v, 5)),
            ('OCV initial from', f'{_hold_label(ocv.initial_from.segment)}, settled start'
                                 if ocv.initial_from else 'no judged OCV'),
            ('OCV final (V)', _r(ocv.final_v, 5)),
            ('OCV final from', f'{_hold_label(ocv.final_from.segment)}, last 30 min'
                               if ocv.final_from else 'no judged OCV'),
            ('OCV loss (%)', _r(ocv.loss_pct, 3)),
            ('OCV decay rate (μV/h): latest window, ending by the death if any',
             _r(ocv.decay_uV_h, 1)),
            ('OCV decay rate window (h)', f'{a:.2f}–{b:.2f}' if math.isfinite(a) else 'none'),
            ('OCV min (V): lowest 10-min median', _r(ocv.lowest_v, 5)),
            ('OCV min at (h)', _r(ocv.lowest_h, 4)),
            ('OCV death (h)', _r(d.start_h, 4) if d else 'none' if ocv.judged else 'not judged'),
            ('OCV death step', f'{d.step} (hold {d.segment})' if d else ''),
            ('OCV recovered after death at (h)',
             '' if ocv.recovered_h is None else _r(ocv.recovered_h, 4)
             if not (d and d.trigger == 'station' and not d.median_fell) else
             f'{ocv.recovered_h:.4f}, after the station\'s limit last acted: the 10-min median '
             f'never fell under the threshold'),
            ('OCV dips (under threshold, too short for death)',
             sum(r.verdict == 'dip' for r in ocv.runs)),
            ('OCV death rule', s.describe() + (
                ". Also a death, however short: a .fcd hold's own voltage limit (Term:E<...) "
                "acting -- the station ending a step whose RH is steady on it, or, in an "
                "RH-cycling loop, the limit tripping in a half of its sub-step: a reading under "
                "it ending the half, or the half ending a sample and a half or more short of the "
                f"header's time with a reading within {TRIP_NEAR_V * 1000:g} mV of the limit at "
                "its end. Dated from the stretch under the limit; a trip counts once the OCV has "
                "reached the limit in the hold and settled, or from the first reading if it never "
                "does. Samples above the current limit are left out of it too"
                if any(f.term_v is not None and not f.gas_change for f in tl.holds) else '')),
            ('OCV decay rule', f'straight line through 1-min medians over {s.rate_window_h:g} h '
                               f'windows ending every {RATE_STEP_H:g} h of each hold, '
                               f'≥ {RATE_MIN_COVERAGE:.0%} of minutes with data; '
                               f'positive while the OCV falls'),
            ('OCV trace sheet', f'{TRACE_BIN_S:g} s bins of every open-circuit sample; '
                                f'the 10-min median is blank where nothing was judged'),
            ('OCV plot bin (min)', f'{ocv.plot_bin_s / 60:.4g} ({ocv.plot_bin_rule}); '
                                   f'each bin is drawn as its median, shaded from its lowest '
                                   f'to its highest sample'),
        ]
    if ca is not None:
        summary += char_summary_rows(ca, ocv)
    if verdict is not None:
        summary = verdict_rows(verdict, ocv, ca) + summary
    notes = (tl.notes + (ocv.notes if ocv is not None else [])
             + (ca.notes if ca is not None else []))
    with pd.ExcelWriter(path, engine='openpyxl') as xw:
        pd.DataFrame(summary, columns=['Item', 'Value']).to_excel(
            xw, sheet_name='Summary', index=False)
        if ca is not None and ca.results:
            pd.DataFrame(cycle_rows(ca, tl)).to_excel(xw, sheet_name='Cycles', index=False)
        pd.DataFrame(hold_rows(tl, ocv)).to_excel(xw, sheet_name='Holds', index=False)
        if has_ocv:
            pd.DataFrame(ocv.by_condition).to_excel(
                xw, sheet_name='OCV by condition', index=False)
            pd.DataFrame(ocv_event_rows(ocv, tl), columns=EVENT_COLUMNS + (
                ['Start (RH cycles)'] if tl.cycles_tested else [])).to_excel(
                xw, sheet_name='OCV events', index=False)
            pd.DataFrame(decay_rows(ocv), columns=DECAY_COLUMNS).to_excel(
                xw, sheet_name='Decay rate', index=False)
            traces = [h.trace for h in ocv.holds if h.trace is not None and not h.trace.empty]
            trace = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
            if tl.cycles_tested and not trace.empty:
                trace.insert(1, 'RH cycles', [tl.cycles_at(x) for x in trace['Hold hours (h)']])
            trace.to_excel(xw, sheet_name='OCV trace', index=False)
            pd.DataFrame([row for h in ocv.holds for row in h.files]).to_excel(
                xw, sheet_name='Hold files', index=False)
        if ca is not None and ca.results:
            characterization_rows(ca, tl).to_excel(xw, sheet_name='Characterizations', index=False)
        pd.DataFrame(manifest_rows(tl)).to_excel(xw, sheet_name='Manifest', index=False)
        pd.DataFrame({'Note': notes or ['none']}).to_excel(
            xw, sheet_name='Notes', index=False)


def build_summary(tl: Timeline, params: Dict[str, Any],
                  ocv: Optional[OcvResult] = None, ca: Optional[CharAnalysis] = None,
                  verdict: Optional[Verdict] = None) -> Dict[str, Any]:
    """Tier 1 scalars for the record layer: one row for the whole test.

    The Label is a string every output name of this script contains, so the
    row matches its unit by name. A label-less row is annotated with step ''
    on merge and is duplicated by a re-push whenever plot names happen to
    yield a step (V29). A value the run could not produce is left out rather
    than stored as a placeholder: 'Hours to death' exists only for a sample
    that died (plan decision 7: alive is Hours to death being absent), and
    'OCV death' only when the OCV rule tripped. Verdict, cause and bracket
    are text.
    """
    row: Dict[str, Any] = {
        'Analysis': 'fc_durability',
        'Label': 'fcdur',
        'Hours tested': float(tl.hours_tested),
        'Hold files': len(tl.holds),
        'Hold segments': len(tl.segments),
        'Step files': len(tl.files),
        'geo_area': float(params['geo_area']),
        'loading': float(params['loading']),
    }
    if tl.left_out:
        row['Folders left out'] = ', '.join(tl.left_out)
    if ocv is not None:
        for key, x, nd in (('OCV initial', ocv.initial_v, 5), ('OCV final', ocv.final_v, 5),
                           ('OCV loss', ocv.loss_pct, 3), ('OCV decay rate', ocv.decay_uV_h, 1),
                           ('OCV min', ocv.lowest_v, 5)):
            if math.isfinite(x):
                row[key] = round(float(x), nd)
        if ocv.death is not None:
            row['OCV death'] = round(ocv.death.start_h, 4)
    if ca is not None:
        judged = ca.death_blocks()
        if judged:
            row['j_xover initial'] = round(judged[0].values['j_xover'], 4)
            row['j_xover final'] = round(judged[-1].values['j_xover'], 4)
            if math.isfinite(judged[-1].values['R_short']):
                row['R_short final'] = round(judged[-1].values['R_short'], 1)
    if tl.cycles_tested:
        row['RH cycles tested'] = int(tl.cycles_tested)
    if verdict is not None:
        row['Verdict'] = 'dead' if verdict.dead else 'alive'
        if verdict.dead:
            row['Hours to death'] = round(verdict.hours, 4)
            if verdict.cycles is not None:
                row['RH cycles to death'] = int(verdict.cycles)
            row['Death cause'] = verdict.cause_text
            if verdict.bracket_text:
                row['Death bracket'] = verdict.bracket_text
    return row


def _num_param(p: Dict[str, Any], key: str, default: float, low: float, high: float,
               notes: Optional[List[str]] = None) -> float:
    """A numeric parameter within [low, high]. Blank input gives the default
    quietly; unreadable or out-of-range input gives it with a note, so 700
    typed for a 0.7 V threshold is caught rather than killing every sample."""
    v = p.get(key, '')
    if v in ('', None):
        return default
    try:
        x = float(v)
    except (TypeError, ValueError):
        x = math.nan
    if math.isfinite(x) and low <= x <= high:
        return x
    if notes is not None:
        notes.append(f'parameter {key} = {v!r} is not a number from {low:g} to {high:g}; '
                     f'used {default:g}')
    return default


# (default, low, high) of each numeric parameter.
PARAM_RANGES: Dict[str, Tuple[float, float, float]] = {
    'geo_area': (DEFAULT_GEO_AREA, 0.01, 1000.0),          # cm²
    'loading': (DEFAULT_LOADING, 0.001, 10.0),             # mg Pt/cm²
    'hold_min_h': (HOLD_MIN_H, 0.0, 1000.0),
    'step_tail_s': (STEP_TAIL_S, 0.0, 3600.0),
    'ocv_threshold': (OCV_THRESHOLD_V, 0.0, 1.3),          # V
    'ocv_window_min': (OCV_WINDOW_MIN, 0.1, 1440.0),
    'ocv_persist_min': (OCV_PERSIST_MIN, 0.0, 10080.0),
    'settle_min': (SETTLE_MIN, 0.0, 1440.0),
    'rate_window_h': (RATE_WINDOW_H, 0.1, 1000.0),
    'ocv_plot_bin_min': (0.0, 0.0, 1440.0),               # 0 = automatic
    'xover_threshold': (XOVER_THRESHOLD, 0.0, 1000.0),     # mA/cm²
    # Inside the stations' 0.12-0.80 V crossover CVs.
    'xover_window_lo': (XOVER_WINDOW_V[0], 0.15, 0.75),    # V
    'xover_window_hi': (XOVER_WINDOW_V[1], 0.20, 0.80),
}
XOVER_MIN_WINDOW_V = 0.05    # a narrower window averages too few points to judge
LEFT_OUT_MAX = 50            # folder names kept from skipped_folders


def run(input_dir: str, output_dir: str, params: dict = None) -> dict:
    """The FC durability analysis: timeline, OCV, characterizations, verdict."""
    p = params or {}
    param_notes: List[str] = []
    num = {key: _num_param(p, key, *rng, notes=param_notes) for key, rng in PARAM_RANGES.items()}
    if num['xover_window_hi'] - num['xover_window_lo'] < XOVER_MIN_WINDOW_V - 1e-9:
        param_notes.append(f'crossover window {num["xover_window_lo"]:g}–{num["xover_window_hi"]:g} V '
                           f'is empty or narrower than {XOVER_MIN_WINDOW_V:g} V; used '
                           f'{XOVER_WINDOW_V[0]:g}–{XOVER_WINDOW_V[1]:g} V')
        num['xover_window_lo'], num['xover_window_hi'] = XOVER_WINDOW_V
    settings = {k: num[k] for k in ('geo_area', 'loading', 'hold_min_h', 'step_tail_s')}
    ocv_settings = OcvSettings(threshold_v=num['ocv_threshold'],
                               window_min=num['ocv_window_min'],
                               persist_min=num['ocv_persist_min'],
                               settle_min=num['settle_min'],
                               rate_window_h=num['rate_window_h'],
                               plot_bin_min=num['ocv_plot_bin_min'])
    char_settings = CharSettings(geo_area=num['geo_area'], loading=num['loading'],
                                 xover_threshold=num['xover_threshold'],
                                 xover_lo=num['xover_window_lo'], xover_hi=num['xover_window_hi'])
    order = p.get('folder_order') or None
    tl = build_timeline(input_dir, hold_min_h=settings['hold_min_h'],
                        step_tail_s=settings['step_tail_s'],
                        folder_order=order if isinstance(order, list) else None)
    tl.notes[:0] = param_notes
    # The upload page leaves skipped folders out of the upload and names them
    # here. Their hours are missing from the clock, so the outputs say so.
    # Names only, as the page sends them: control characters cannot go into
    # a workbook cell, and a folder that was read after all is not left out.
    left = p.get('skipped_folders')
    if isinstance(left, list):
        names = [re.sub(r'[\x00-\x1f\x7f]', '', x).strip().strip('/')
                 for x in left if isinstance(x, str)]
        names = [n[:200] for n in names if n and n not in tl.folders]
        tl.left_out = names[:LEFT_OUT_MAX] + ([f'and {len(names) - LEFT_OUT_MAX} more']
                                             if len(names) > LEFT_OUT_MAX else [])
    if tl.left_out:
        tl.notes.insert(len(param_notes),
                        'left out at upload (skipped in the folder list): '
                        + ', '.join(tl.left_out)
                        + '; their steps and hold hours are not in this analysis')
    if not tl.files:
        # Raised rather than returned: a returned error status still shows the
        # job as complete (V22).
        # What was there instead, so a format it does not read says so.
        seen = []
        if tl.skipped:
            seen.append(f'{len(tl.skipped)} file(s) skipped, e.g. {tl.skipped[0][0]}: '
                        f'{tl.skipped[0][1]}')
        if tl.ignored:
            seen.append('ignored: ' + ', '.join(f'{v} {k}' for k, v in sorted(tl.ignored.items())))
        raise RuntimeError('No step files found. FC Durability reads station CSVs named like '
                           'c5_<sample>_OCV_95Cell-30RH_0.1WH2-0.2WA.csv and Scribner '
                           'FuelCell files named like c4_OCV_90C-100RH_0o1H2-0o1Air_0kPa.fcd.'
                           + (' Found ' + '; '.join(seen) + '.' if seen else ''))
    areas = sorted({f.area_cm2 for f in tl.files if f.area_cm2 is not None})
    if areas and any(abs(a - num['geo_area']) > 1e-6 for a in areas):
        tl.notes.insert(len(param_notes),
                        f'the .fcd headers give an active area of '
                        f'{", ".join(f"{a:g}" for a in areas)} cm² (SurfaceArea); this analysis '
                        f'used {num["geo_area"]:g} cm², the geometric area parameter')
    if {f.source for f in tl.files} == {'csv', 'fcd'}:
        # Their names give conditions differently (W/D flags and no pressure
        # against an RH token and a pressure), so their holds never share a
        # conditions key.
        tl.notes.insert(len(param_notes),
                        'this run mixes station CSVs and .fcd files; their names give '
                        'conditions differently, so holds of the two never count as the same '
                        'conditions')
    ocv = analyze_ocv(tl, ocv_settings)
    ca = analyze_characterizations(tl, char_settings)
    verdict = decide(tl, ocv, ca)
    if verdict.dead and (ocv.death is None or verdict.hours < ocv.death.start_h):
        # Died on crossover before any OCV trip: the headline OCV rate stops there too.
        stale = ocv.rate_note
        set_rate_cutoff(ocv, verdict.hours, verdict.cause)
        ocv.notes = [n for n in ocv.notes if n != stale] + ([ocv.rate_note] if ocv.rate_note else [])

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    from scripts.helpers.conditions import img_ext_from_params
    image_ext = img_ext_from_params(p)
    if image_ext:
        # A .fcd name carries no sample; the run's folder does.
        sample = (p.get('sample_name') or tl.files[0].sample
                  or Path(tl.files[0].folder).name or 'sample')
        kinds = {ca.series_kind[s] for s in ca.series if ca.blocks(s)}

        def save(name: str, draw) -> None:
            draw(str(out / f'{name}.{image_ext}'))
            plt.close('all')

        save('fcdur_timeline', lambda path: plot_timeline(
            tl, path, title=f'Durability test timeline — {sample}', verdict=verdict))
        if tl.cycles_tested:
            save('fcdur_rh_cycles', lambda path: plot_rh_cycles(
                tl, verdict, path, title=f'RH cycles — {sample}'))
        if ocv.holds:
            save('fcdur_ocv_vs_time', lambda path: plot_ocv(
                tl, ocv, path, title=f'Hold OCV — {sample}'))
        if 'crossover' in kinds:
            save('fcdur_xover_vs_time', lambda path: plot_xover_vs_time(
                tl, ca, path, title=f'H₂ crossover — {sample}'))
        if 'eis' in kinds:
            save('fcdur_hfr_vs_time', lambda path: plot_metric_vs_time(
                tl, ca, 'eis', 'HFR', 'HFR (mΩ·cm²)', 'HFR', 'mΩ·cm²', '.1f', path,
                title=f'HFR — {sample}', plot_type='fcdur_hfr_vs_time'))
        if 'ecsa' in kinds:
            save('fcdur_ecsa_vs_time', lambda path: plot_metric_vs_time(
                tl, ca, 'ecsa', 'RF', 'Roughness factor (cm² Pt per cm²)', 'RF', '', '.1f', path,
                title=f'ECSA — {sample}', plot_type='fcdur_ecsa_vs_time', loss_name='ECSA loss',
                also=('ECSA', 'ECSA', 'm²/g', '.2f')))
        if 'polcurve' in kinds:
            save('fcdur_perf_vs_time', lambda path: plot_perf_vs_time(
                tl, ca, path, title=f'Pol-curve performance — {sample}'))
        for kind in ('polcurve', 'crossover', 'eis', 'ecsa'):
            if kind in kinds:
                save(EVOLUTION[kind][0], lambda path, kind=kind: plot_evolution(
                    ca, kind, path, title=f'{EVOLUTION[kind][1]} — {sample}'))
    export_workbook(tl, str(out / 'fc_durability.xlsx'), settings, ocv, ca, verdict)

    notes = tl.notes + ocv.notes + ca.notes
    for note in notes:
        print(f'  note: {note}')
    print(f'  verdict: {verdict.text}')
    return {'status': 'success',
            'files_processed': len(tl.files),
            'hours_tested': tl.hours_tested,
            'ocv_death_h': ocv.death_h,
            'verdict': verdict.text,
            'hours_to_death': verdict.hours if verdict.dead else None,
            'notes': notes,
            'summary': [build_summary(tl, settings, ocv, ca, verdict)]}
