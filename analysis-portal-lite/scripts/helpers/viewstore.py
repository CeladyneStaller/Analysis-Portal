"""
View tab data layer.
=====================

Read-only access to the JSONBin store for the viewer. Transport lives in
`jsonbin.py`; this module adds the read paths the viewer needs — filtering,
caching, and materialising stored sidecars back onto disk.

Why materialisation matters: `compare_polcurves.run()` locates each source with
`find_sidecar(output_dir, filename)`, reading from disk. Writing decoded
sidecars into the same `{dir}/_plot_data/{plot}.json` layout the analysis
scripts produce means the existing comparison script runs unchanged against
historical data — grouping modes, clean labels, condition subtitles, readout
boxes and the Excel Metrics sheet all come along for free.

The viewer is read-only (fork F). Nothing here writes to JSONBin.
"""

import bisect
import json
import math
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.helpers import jsonbin, registry
from scripts.helpers.record import (
    STAND_OPTIONS, decode_sidecars, plot_bucket, stand_matches,
)

# Detail bins used to be immutable, which is why this cache has no expiry.
# Sample-keyed merging makes them mutable: a later run rewrites the same bin.
# The cache key therefore carries the index entry's timestamp, which advances
# on every merge — a stale record simply stops being addressable and ages out
# by LRU, so nothing needs to invalidate it explicitly.
# 32 x ~360 KB is roughly 11 MB.
DETAIL_CACHE_SIZE = 32

# The index is appended to as runs complete, so it gets a short TTL rather than
# being cached indefinitely.
INDEX_TTL_S = 30

_detail_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_cache_lock = threading.Lock()
# `gen` counts the times the copy was dropped (forget_index, clear_cache): a
# read that was out across one is not kept.
_index_cache: Dict[str, Any] = {'at': 0.0, 'data': None, 'gen': 0}


# ─────────────────────────────────────────────────────────────────────
#  Index
# ─────────────────────────────────────────────────────────────────────

def fetch_index(force: bool = False) -> Dict[str, Any]:
    """Read the index, cached for INDEX_TTL_S."""
    with _cache_lock:
        fresh = (not force
                 and _index_cache['data'] is not None
                 and (time.time() - _index_cache['at']) < INDEX_TTL_S)
        if fresh:
            return _index_cache['data']
        gen = _index_cache.get('gen', 0)

    index = jsonbin.fetch_index()

    with _cache_lock:
        # Dropped while this read was out (a push, say): what it read may be
        # from before, so it is returned but not kept.
        if _index_cache.get('gen', 0) == gen:
            _index_cache['data'] = index
            _index_cache['at'] = time.time()
    return index


def _entry_analyses(entry: Dict[str, Any]) -> List[str]:
    return sorted({d.get('Analysis', '') for d in entry.get('Data', [])
                   if d.get('Analysis')})


def list_runs(*, sample: Optional[str] = None,
              script: Optional[str] = None,
              analysis: Optional[str] = None,
              stand: Optional[str] = None,
              since: Optional[str] = None,
              until: Optional[str] = None,
              limit: Optional[int] = None,
              force: bool = False) -> Dict[str, Any]:
    """Return index entries, newest first, optionally filtered.

    Filters: sample name substring, script, analysis type, test stand, and an
    ISO date range. Condition filtering is deferred.

    Timestamps are ISO-8601 Z strings, so lexicographic comparison is also
    chronological and no parsing is needed.
    """
    index = fetch_index(force=force)
    runs = list(index.get('runs', []))

    if sample:
        needle = sample.lower()
        runs = [r for r in runs
                if needle in str(r.get('sample_name', '')).lower()]
    if script:
        runs = [r for r in runs if r.get('script') == script]
    if analysis:
        runs = [r for r in runs if analysis in _entry_analyses(r)]
    if stand:
        # Entries predating numbered stands carry a bare family and match any
        # stand within it — there is no way to tell which one they came from,
        # and hiding them from every numbered filter would be worse.
        runs = [r for r in runs if stand_matches(r.get('stand'), stand)]
    if since:
        runs = [r for r in runs if str(r.get('timestamp', '')) >= since]
    if until:
        runs = [r for r in runs if str(r.get('timestamp', '')) <= until]

    # Order by experiment date where it is known, falling back to the analysis
    # date. Both are ISO, so the leading YYYY-MM-DD compares lexicographically
    # even though one is a date and the other a datetime.
    runs.sort(key=_sort_key, reverse=True)
    total = len(runs)
    if limit:
        runs = runs[:limit]

    return {
        'runs': runs,
        'total': total,
        'returned': len(runs),
        'facets': index_facets(index),
        # What each analysis and value is: labels, units, which direction is
        # better. Sent with the runs so the views read one table rather than
        # keeping their own copies of it.
        'registry': registry.as_json(),
        # Old per-plot-type buckets still named in the index, and the analysis
        # each belongs to. Empty once the electrolyzer rebuild has run.
        'analysis_aliases': registry.aliases(
            sorted({a for e in index.get('runs', []) for a in _entry_analyses(e)}),
            plot_bucket),
    }


def _stand_facets(present: set) -> List[str]:
    """Filter options: the canonical stands plus any unnumbered value in use.

    Ordered so an unnumbered family sits with its own numbered stands rather
    than trailing the whole list, where 'FCTS' after 'FCTS 4' reads like a
    mistake instead of the catch-all it is.
    """
    from scripts.helpers.record import stand_family
    extras = sorted(present - set(STAND_OPTIONS))
    out: List[str] = []
    for family in ('Scribner', 'FCTS'):
        # The aggregate heads its own group, so the broad choice reads as the
        # heading for the narrow ones rather than an extra entry among them.
        out.append(f'All {family}')
        out += [s for s in STAND_OPTIONS if stand_family(s) == family]
        out += [e for e in extras if stand_family(e) == family]
    # Anything whose family cannot be read at all still needs an option.
    out += [e for e in extras if stand_family(e) is None]
    return out


def _sort_key(entry: Dict[str, Any]) -> str:
    """Experiment date when recoverable, else the analysis date."""
    return str(entry.get('run_date') or entry.get('timestamp', ''))[:10]


def index_facets(index: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
    """Distinct values available for filtering, for populating the UI controls."""
    if index is None:
        index = fetch_index()
    runs = index.get('runs', [])
    samples, scripts, analyses, stands = set(), set(), set(), set()
    for r in runs:
        if r.get('sample_name'):
            samples.add(r['sample_name'])
        if r.get('script'):
            scripts.add(r['script'])
        if r.get('stand'):
            stands.add(r['stand'])
        analyses.update(_entry_analyses(r))
    return {
        'samples': sorted(samples),
        'scripts': sorted(scripts),
        'analyses': sorted(analyses),
        # The canonical stands, plus any other value actually recorded — a
        # bare 'Scribner' or 'FCTS' from before stands were numbered. Matching
        # is exact, so without these those entries would match no filter at
        # all. The extra options vanish as the data is backfilled.
        'stands': _stand_facets(stands),
    }


# ─────────────────────────────────────────────────────────────────────
#  Detail bins
# ─────────────────────────────────────────────────────────────────────

def _entry_for(key: str, index: Optional[Dict[str, Any]] = None
               ) -> Optional[Dict[str, Any]]:
    """Resolve an index entry from a bin id, sample name, or job id.

    `bin_id` is tried first because it is the only one guaranteed unambiguous.
    A sample may still have more than one entry — anything written before
    merging was enabled — and resolving those by sample name would silently
    hand back whichever is newest, so a caller holding a specific entry gets
    the bin it actually asked for.

    Resolving through the index also validates the key: an arbitrary bin id
    cannot be used to read a bin the index does not reference.
    """
    runs = (index or fetch_index()).get('runs', [])
    for field in ('bin_id', 'sample_name', 'job_id'):
        matches = [e for e in runs if e.get(field) == key]
        if len(matches) == 1:
            return matches[0]
        if matches:
            # Ambiguous only for sample_name pre-migration; take the newest.
            matches.sort(key=lambda e: str(e.get('timestamp', '')))
            return matches[-1]
    return None


def fetch_detail(key: str) -> Dict[str, Any]:
    """Fetch a run's detail bin by bin id, sample name, or job id.

    Cached against the index entry's timestamp, so a merged bin is re-read
    rather than served stale.
    """
    entry = _entry_for(key)
    if entry is None:
        # Could have been written moments ago. Retry once against a fresh index.
        entry = _entry_for(key, fetch_index(force=True))
    if entry is None or not entry.get('bin_id'):
        raise KeyError(f'no indexed run matching {key!r}')

    bin_id = entry['bin_id']
    cache_key = f"{bin_id}@{entry.get('timestamp', '')}"

    with _cache_lock:
        if cache_key in _detail_cache:
            _detail_cache.move_to_end(cache_key)
            return _detail_cache[cache_key]

    payload = jsonbin._request(
        f'{jsonbin._JSONBIN_BASE}/{bin_id}/latest', method='GET')
    record = payload.get('record') if isinstance(payload, dict) else None
    if not isinstance(record, dict):
        raise RuntimeError(f'detail bin {bin_id} returned no usable record')

    with _cache_lock:
        _detail_cache[cache_key] = record
        _detail_cache.move_to_end(cache_key)
        while len(_detail_cache) > DETAIL_CACHE_SIZE:
            _detail_cache.popitem(last=False)
    return record


def cache_stats() -> Dict[str, Any]:
    """Cache state, for the diagnostics endpoint."""
    with _cache_lock:
        return {
            'detail_cached': len(_detail_cache),
            'detail_capacity': DETAIL_CACHE_SIZE,
            'detail_keys': list(_detail_cache.keys()),
            'index_age_s': (round(time.time() - _index_cache['at'], 1)
                            if _index_cache['data'] is not None else None),
            'index_ttl_s': INDEX_TTL_S,
        }


def clear_cache() -> None:
    with _cache_lock:
        _detail_cache.clear()
        _index_cache['data'] = None
        _index_cache['at'] = 0.0
        _index_cache['gen'] = _index_cache.get('gen', 0) + 1


def forget_index() -> None:
    """Drop the cached index, so the next read sees a run pushed a moment ago.

    Called after every push. The detail cache needs nothing: it is keyed by
    the index entry's timestamp, which the push advances. A read already out
    is not kept when it answers (fetch_index).
    """
    with _cache_lock:
        _index_cache['data'] = None
        _index_cache['at'] = 0.0
        _index_cache['gen'] = _index_cache.get('gen', 0) + 1


# ─────────────────────────────────────────────────────────────────────
#  Plot inventory
# ─────────────────────────────────────────────────────────────────────

def run_plots(key: str) -> List[Dict[str, Any]]:
    """List a run's plots, flagging which can be re-rendered.

    A plot is renderable only if its sidecar was stored. Cleaning sidecars are
    excluded at write time because they dominate the transport budget, so
    cleaning plots appear here with `renderable: False` — their metrics are
    intact, only the plot data is absent.

    `plot_type` comes from the sidecar, so it is '' for a plot without one.
    Run detail orders an analysis's main plots by it.
    """
    detail = fetch_detail(key)
    sidecars = decode_sidecars(detail)

    out: List[Dict[str, Any]] = []
    for bucket, plots in (detail.get('metrics') or {}).items():
        for plot_name, entry in plots.items():
            sc = sidecars.get(plot_name)
            out.append({
                'plot': plot_name,
                'analysis': bucket,
                'conditions': entry.get('conditions') or {},
                'values': entry.get('values') or {},
                'renderable': plot_name in sidecars,
                'plot_type': str((sc or {}).get('plot_type') or '')
                             if isinstance(sc, dict) else '',
            })
    out.sort(key=lambda p: (p['analysis'], p['plot']))
    return out


# ─────────────────────────────────────────────────────────────────────
#  Materialisation
# ─────────────────────────────────────────────────────────────────────

# A chart is under a thousand pixels wide, so a curve carrying thousands of
# points spends bytes on detail no screen can show. Measured: a polcurve set of
# 24 curves is 15 kB and an EIS set 27 kB — both irrelevant — but an ECSA scan
# at ~900 points per cycle reaches roughly 1 MB for the same chart. Subsampling
# to this many points is visually identical — the chart is 760 px wide, so 400
# points is already about one every two pixels — and cuts the payload by more
# than half.
#
# Only the browser sees decimated data. Export re-renders server-side from the
# stored sidecar at full resolution, so the figure and workbook are unaffected.
MAX_SERIES_POINTS = 400
# A chart that computes from the points rather than only drawing them asks for
# them whole (`full`): Comparisons' OCV smoothing takes medians of the stored
# points, and medians of bucket extremes are not those; its RH cycles axis
# reads hours off a test's map to cycles (RH_CYCLES_PLOT_TYPE), whose kinks a
# thinned copy can cut. Only for the plot types named here, and still bounded,
# far above the most points an FC Durability OCV line holds (about 3,000 bins
# per test) or its cycle map (a point every 25 cycles).
MAX_FULL_SERIES_POINTS = 20_000
FULL_SERIES_PLOT_TYPES = {'fcdur_ocv_vs_time', 'fcdur_rh_cycles'}
# FC Durability stores, with a test that cycles RH, a plot of RH cycles
# completed against hold hours: this line, a point at every 25th cycle and at
# each hold's start and end (fc_durability_analysis.rh_cycle_map). Comparisons
# can draw a durability plot against cycles off it (materialize_for_compare).
RH_CYCLES_PLOT_TYPE = 'fcdur_rh_cycles'
RH_CYCLES_LINE = 'RH cycles completed'
RH_CYCLES_XLABEL = 'RH cycles'
# The durability plots drawn against hold hours, which Comparisons offers
# against RH cycles; the timeline, the evolution plots and the map are not.
ON_CYCLES_PLOT_TYPES = {'fcdur_ocv_vs_time', 'fcdur_xover_vs_time', 'fcdur_hfr_vs_time',
                        'fcdur_ecsa_vs_time', 'fcdur_perf_vs_time'}
ON_CYCLES_SUFFIX = '~rh-cycles'          # the staged copy against RH cycles
# The labels of a high–low band's edges (plot_compare.BAND_LOW and BAND_HIGH,
# not imported, which would load matplotlib here).
BAND_EDGE_LABELS = ('bin low', 'bin high')


def _decimate(xs: List[Any], ys: List[Any], limit: int = MAX_SERIES_POINTS) -> Dict[str, List[Any]]:
    """Thin a series while keeping its extremes, endpoints included.

    Min/max per bucket rather than plain subsampling. Taking every nth point
    clips peaks, and on a cyclic voltammogram the hydrogen adsorption and
    desorption peaks *are* the measurement — a decimation that rounds them off
    changes what the chart says. Measured on a sharply peaked trace, plain
    subsampling moved the curve by 0.8 % of its span, about three pixels;
    keeping the extreme of each bucket is exact at every peak.

    Endpoints are kept too: for a polarization curve the last point is the
    limiting current and for a Nyquist it is the low-frequency end.

    Only the browser sees this. Export re-renders server-side from the stored
    sidecar at full resolution, so the figure and workbook are unaffected.
    """
    n = min(len(xs), len(ys))
    if n <= limit:
        return {'x': list(xs[:n]), 'y': list(ys[:n])}

    buckets = max(1, limit // 2)
    width = n / float(buckets)
    keep = {0, n - 1}
    for b in range(buckets):
        lo, hi = int(b * width), min(int((b + 1) * width), n)
        if hi <= lo:
            continue
        window = range(lo, hi)
        keep.add(min(window, key=lambda i: ys[i]))
        keep.add(max(window, key=lambda i: ys[i]))
    idx = sorted(keep)
    return {'x': [xs[i] for i in idx], 'y': [ys[i] for i in idx],
            'decimated': True}


def _whole(xs: List[Any], ys: List[Any]) -> Dict[str, List[Any]]:
    """A series as stored, for an export: nothing thinned."""
    n = min(len(xs), len(ys))
    return {'x': list(xs[:n]), 'y': list(ys[:n])}


def _read_record(key: str, memo: Optional[Dict[str, Any]] = None):
    """A run's detail record and its decoded sidecars, read once per `memo`.

    A batch asks for several plots of one record, and decoding is of the whole
    record every time; a key that cannot be read fails once per memo too (an
    unknown key reads the index again before it fails). Without a memo, as
    before: read and decoded on every call.
    """
    if memo is not None and key in memo:
        got = memo[key]
        if isinstance(got, BaseException):
            raise got
        return got
    try:
        detail = fetch_detail(key)
        got = (detail, decode_sidecars(detail))
    except Exception as e:                                  # noqa: BLE001
        if memo is not None:
            memo[key] = e
        raise
    if memo is not None:
        memo[key] = got
    return got


def _unit_plot(detail: Dict[str, Any], sidecars: Dict[str, Any],
               analysis: str, step: str, plot_type: str = '') -> str:
    """The plot a request by (analysis, step) gets, or '' for none.

    With `plot_type`, the first plot of that type in the unit. Without it, the
    analysis's main plot: an electrolyzer unit holds a dozen plots — the curve,
    its fits, iR correction, per-spectrum fits — and the first one stored is
    whichever the script happened to write first ('ir_correction' for a pol
    curve). The registry lists each analysis's main plots in order; the first
    of them that is stored is the one returned, and its type comes back in
    the payload. A unit none of whose main plots is stored returns nothing
    rather than a plot of some other kind.

    Analyses that list no main plots — the fuel-cell ones — keep the rule they
    have always had: the first plot filed under the unit.
    """
    names = [name for bucket, plots in (detail.get('metrics') or {}).items()
             if not analysis or bucket == analysis
             for name, entry in plots.items()
             if str((entry.get('conditions') or {}).get('step') or '') == step]

    def typed(pt: str) -> str:
        return next((n for n in names
                     if (sidecars.get(n) or {}).get('plot_type') == pt), '')

    if plot_type:
        return typed(plot_type)
    main = registry.main_plots(analysis)
    if not main:
        return names[0] if names else ''
    return next((n for n in map(typed, main) if n), '')


def plot_series(key: str, analysis: str = '', step: str = '',
                plot: str = '', plot_type: str = '',
                full: bool = False, points: bool = False, labels: bool = False,
                stored: bool = False,
                memo: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """The plotted series for one stored plot, for charting in the browser.

    Every other read path returns a rendered PNG. The numbers behind it are in
    the sidecar all along — this hands them over so a chart can be drawn client
    side and re-drawn on every toggle without another server round trip.

    A plot is addressed either by name, or by (analysis, step), which is what
    the index carries and therefore what a picker built from the index can ask
    for. `plot_type` narrows (analysis, step) to one type of plot: a step-less
    analysis such as FC Durability stores ten plots under its one unit, and
    the first of them is not the one a chart of OCV against hours wants. A
    name, when given, still wins. See `_unit_plot` for which plot an
    (analysis, step) request gets without one.

    Lines longer than MAX_SERIES_POINTS come back thinned to bucket extremes
    and marked `decimated`; with `full`, on a plot type in
    FULL_SERIES_PLOT_TYPES, only past MAX_FULL_SERIES_POINTS; with `stored`
    (an export's read), never.

    Asked for (the custom plot), each axis also carries its `points`, the
    scatters the plot marks its values with (an FC Durability test's R_short,
    or a hold's few OCV decay windows), thinned the same way, and an axis with
    only points is kept; and with `labels` every line and point is its label
    and its stored count `n`, without the numbers, which is what a list of
    what a sample stores needs. Without them the answer is as it always was.
    `memo` (a dict the caller keeps for one batch) reads each record once.

    Returns None when the plot is not stored — cleaning sidecars are excluded
    at write time, so this is an ordinary outcome rather than an error.
    """
    detail, sidecars = _read_record(key, memo)
    if not sidecars:
        return None

    name = plot or _unit_plot(detail, sidecars, analysis, step, plot_type)
    sc = sidecars.get(name)
    if not sc:
        return None

    # Axes are passed through with their labels but without the annotation
    # text and reference lines: those are readout furniture for a rendered
    # figure, and a comparison of several curves cannot show one curve's
    # readout box without implying it applies to all of them.
    whole = full and sc.get('plot_type') in FULL_SERIES_PLOT_TYPES
    limit = MAX_FULL_SERIES_POINTS if whole else MAX_SERIES_POINTS

    def send(label: Any, xs: List[Any], ys: List[Any]) -> Dict[str, Any]:
        if labels:
            return {'label': label or '', 'n': min(len(xs), len(ys))}
        return {'label': label or '',
                **(_whole(xs, ys) if stored else _decimate(xs, ys, limit))}

    axes = []
    for ax in (sc.get('data') or {}).get('axes', []):
        lines = [send(ln.get('label'), ln.get('x') or [], ln.get('y') or [])
                 for ln in ax.get('lines', []) if ln.get('x')]
        # A band edge level across a hold's two bins is stored as two points,
        # not a line (the sidecar writer reads a level two-point line as a
        # reference line). It is sent as a line all the same, so a chart pairs
        # it with its trace as the server-side redraw does.
        lines += [send(pts['label'], pts.get('x') or [], pts.get('y') or [])
                  for pts in ax.get('scatters', [])
                  if pts.get('label') in BAND_EDGE_LABELS and pts.get('x')]
        marks = [send(pts.get('label'), pts.get('x') or [], pts.get('y') or [])
                 for pts in ax.get('scatters', [])
                 if points and pts.get('x') and pts.get('label') not in BAND_EDGE_LABELS]
        if lines or marks:
            axes.append({'title': ax.get('title') or '',
                         'xlabel': ax.get('xlabel') or '',
                         'ylabel': ax.get('ylabel') or '',
                         'is_twin': bool(ax.get('is_twin')),
                         'lines': lines,
                         **({'points': marks} if points else {})})
    if not axes:
        return None
    return {'key': key, 'plot': name, 'plot_type': sc.get('plot_type', ''),
            'analysis': analysis or plot_bucket(sc.get('plot_type', 'unknown')),
            'step': step, 'sample_name': detail.get('sample_name', ''),
            'axes': axes}


def plot_series_batch(selections: List[Dict[str, str]]) -> Dict[str, Any]:
    """Series for several plots in one call.

    A chart of twenty curves is twenty sidecars; asking for them one at a time
    would be twenty round trips against a cache that already holds the detail
    records after the first. Selections that are not stored come back in
    `missing` rather than failing the batch — one unavailable curve should not
    cost the other nineteen. Each record is read and decoded once per call,
    however many of its plots the batch asks for.
    """
    out, missing = [], []
    memo: Dict[str, Any] = {}
    for sel in selections or []:
        key = str(sel.get('key') or '')
        try:
            got = plot_series(key, str(sel.get('analysis') or ''),
                              str(sel.get('step') or ''),
                              str(sel.get('plot') or ''),
                              str(sel.get('plot_type') or ''),
                              full=sel.get('full') is True,
                              points=sel.get('points') is True,
                              labels=sel.get('labels') is True,
                              memo=memo)
        except Exception as e:
            missing.append({**sel, 'reason': f'{type(e).__name__}: {e}'})
            continue
        if got:
            out.append(got)
        else:
            missing.append({**sel, 'reason': 'no stored plot data'})
    return {'series': out, 'missing': missing}


def materialize_sidecars(key: str, dest_dir: Path,
                         plots: Optional[List[str]] = None) -> List[str]:
    """Write a run's stored sidecars to disk in analysis-output layout.

    Produces `{dest_dir}/_plot_data/{plot}.json`, which is exactly what
    `find_sidecar()` expects — so anything that reads sidecars off disk works
    against historical runs without modification.

    Returns the plot names actually written. Requested plots whose sidecars were
    not stored are skipped silently; callers should compare against the returned
    list rather than assume.
    """
    detail = fetch_detail(key)
    sidecars = decode_sidecars(detail)
    if plots is not None:
        wanted = set(plots)
        sidecars = {k: v for k, v in sidecars.items() if k in wanted}

    dest = Path(dest_dir) / '_plot_data'
    dest.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for name, sidecar in sidecars.items():
        (dest / f'{name}.json').write_text(
            json.dumps(sidecar, ensure_ascii=False))
        written.append(name)
    return sorted(written)


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def rh_cycle_map(sidecars: Dict[str, Any]) -> Optional[Tuple[List[float], List[float]]]:
    """A stored test's map from hold hours to RH cycles completed, or None.

    The points of its RH_CYCLES_LINE with both numbers, in order of hours (one
    that steps back in hours is dropped). None for a test that holds RH
    steady, or one stored before RH cycles were counted.
    """
    for sc in (sidecars or {}).values():
        if not isinstance(sc, dict) or sc.get('plot_type') != RH_CYCLES_PLOT_TYPE:
            continue
        for ax in (sc.get('data') or {}).get('axes', []):
            for ln in ax.get('lines', []):
                if ln.get('label') != RH_CYCLES_LINE:
                    continue
                hours: List[float] = []
                cycles: List[float] = []
                for h, c in zip(ln.get('x') or [], ln.get('y') or []):
                    if _finite(h) and _finite(c) and not (hours and h < hours[-1]):
                        hours.append(float(h))
                        cycles.append(float(c))
                return (hours, cycles) if len(hours) >= 2 else None
    return None


def cycles_at(hours: List[float], cycles: List[float], h: Any) -> Any:
    """RH cycles completed at `h` hold hours on a map (rh_cycle_map): straight
    between its points and held at its ends, as the Comparisons chart reads
    it. Anything that is not a number passes through."""
    if not _finite(h):
        return h
    if h <= hours[0]:
        return cycles[0]
    if h >= hours[-1]:
        return cycles[-1]
    i = bisect.bisect_right(hours, h)            # hours[i - 1] <= h < hours[i]
    h0, h1, c0, c1 = hours[i - 1], hours[i], cycles[i - 1], cycles[i]
    return c0 + (c1 - c0) * (h - h0) / (h1 - h0) if h1 > h0 else c1


def sidecar_on_cycles(sidecar: Dict[str, Any], hours: List[float],
                      cycles: List[float]) -> Dict[str, Any]:
    """A durability plot's sidecar redrawn against RH cycles, in place.

    Every x on its axes is read off the test's map: its lines and points (a
    band's edges with them, so they still pair), its vertical lines (the
    characterizations and a death), text placed by data, and its x range; the
    x label says RH cycles.
    """
    def at(v: Any) -> Any:
        return cycles_at(hours, cycles, v)

    for ax in (sidecar.get('data') or {}).get('axes', []):
        for tr in list(ax.get('lines') or []) + list(ax.get('scatters') or []):
            tr['x'] = [at(v) for v in tr.get('x') or []]
        for vl in ax.get('axvlines') or []:
            vl['x'] = at(vl.get('x'))
        for tx in ax.get('texts') or []:
            if tx.get('in_axes_fraction') is False:
                tx['x'] = at(tx.get('x'))
        if isinstance(ax.get('xlim'), list):
            ax['xlim'] = [at(v) for v in ax['xlim']]
        ax['xlabel'] = RH_CYCLES_XLABEL
    return sidecar


def materialize_for_compare(selections: List[Dict[str, str]],
                            root: Path) -> List[Dict[str, str]]:
    """Stage historical plots for the comparison script.

    `selections` is [{key, plot, label?}, ...] where `key` identifies a run —
    a bin id, sample name or job id. `job_id` is still accepted for callers
    that have not been updated.

    Each run gets its own directory under `root`, mirroring how live jobs are
    laid out, so the returned list can be passed straight through as the
    comparison script's `sources` parameter.

    A durability plot against hold hours (ON_CYCLES_PLOT_TYPES) selected with
    `x: 'cycles'` is staged against RH cycles (sidecar_on_cycles), off its own
    test's map; a test without one is left out, as Comparisons leaves it off
    its chart. A plot is staged once for every selection of it.

    Selections whose sidecar was not stored are omitted.
    """
    def _key(sel: Dict[str, str]) -> str:
        return str(sel.get('key') or sel.get('job_id') or '')

    by_run: Dict[str, List[str]] = {}
    for sel in selections:
        by_run.setdefault(_key(sel), []).append(sel['plot'])

    available: Dict[str, set] = {}
    dirs: Dict[str, Path] = {}
    for i, (key, plots) in enumerate(by_run.items()):
        # Directory names come from the position rather than the key: a key may
        # be a sample name, and those contain characters that are awkward on a
        # filesystem.
        run_dir = Path(root) / f'run{i}'
        dirs[key] = run_dir
        available[key] = set(materialize_sidecars(key, run_dir, plots))

    sources: List[Dict[str, Any]] = []
    for sel in selections:
        key, plot = _key(sel), sel['plot']
        if plot not in available.get(key, set()):
            continue
        detail = fetch_detail(key)
        sidecars = decode_sidecars(detail)
        plot_type = str((sidecars.get(plot) or {}).get('plot_type', ''))
        name = plot
        if sel.get('x') == 'cycles' and plot_type in ON_CYCLES_PLOT_TYPES:
            cmap = rh_cycle_map(sidecars)
            if cmap is None:
                continue
            # A copy of its own beside the stored one, made from the stored
            # one, so the same plot asked for on hold hours is still staged
            # as stored.
            name = plot + ON_CYCLES_SUFFIX
            staged = dirs[key] / '_plot_data'
            (staged / f'{name}.json').write_text(json.dumps(
                sidecar_on_cycles(json.loads((staged / f'{plot}.json').read_text()), *cmap),
                ensure_ascii=False))
        source: Dict[str, Any] = {
            'job_id': detail.get('job_id', key),
            'label': sel.get('label', ''),
            # The comparison script strips the extension to find the sidecar,
            # so the suffix here is nominal.
            'filename': f'{name}.png',
            'output_dir': str(dirs[key]),
            'sample_name': detail.get('sample_name', ''),
            # Carried so the caller can check the selection is comparable
            # before spending an executor slot on it. The comparison script
            # ignores keys it does not use.
            'plot_type': plot_type,
        }
        # Whether the plot's high–low band is drawn, chosen per sample on the
        # Comparisons chart; only a literal true or false is passed on, and
        # without one the comparison draws the band as stored.
        if isinstance(sel.get('band'), bool):
            source['band'] = sel['band']
        sources.append(source)
    return sources


# The most curves one custom-plot workbook is asked for: six quantities, a
# dozen samples, a few series each; and the most samples, each a record read.
CUSTOM_EXPORT_MAX = 400
CUSTOM_EXPORT_MAX_SAMPLES = 64
CUSTOM_CURVE_FIELDS = ('name', 'stored', 'quantity', 'unit', 'axis', 'series', 'cond')


def custom_plot_curves(curves: List[Dict[str, Any]], x: str = '') -> List[Dict[str, Any]]:
    """The custom plot's curves, read again with every stored point, for its
    workbook.

    Each curve names the stored plot the chart drew it from, as the chart asked
    for it (`src`: analysis, step, plot type, and whether it asked for points),
    and what it took (`pick`): one axis by position, and lines and points on it
    by position, each with the label the chart found there. A curve that is
    the power of a polarization curve (`power`) is its current density times
    its voltage, in mW/cm². Against RH cycles (`x` 'cyc'), each test's hold
    hours are read off its own map, as the chart reads them.

    Returns the curves in order, each with its descriptive fields as sent,
    `parts` (lists of [x, y] with both numbers, in order of their first x, as
    the chart lists them) and `reason`: why a curve has no parts — its plot
    not stored, its parts no longer where the chart found them (stored again
    since), or, against RH cycles, its test counting none. Each record is read
    once, and each plot once, however many curves come from it.
    """
    maps: Dict[str, Any] = {}
    memo: Dict[str, Any] = {}
    reads: Dict[Tuple[Any, ...], Any] = {}
    out: List[Dict[str, Any]] = []
    for c in curves:
        c = c if isinstance(c, dict) else {}
        key = str(c.get('key') or '')
        row: Dict[str, Any] = {'key': key, **{f: str(c.get(f) or '') for f in CUSTOM_CURVE_FIELDS},
                               'parts': [], 'reason': ''}
        out.append(row)
        src = c.get('src') if isinstance(c.get('src'), dict) else {}
        pick = c.get('pick') if isinstance(c.get('pick'), dict) else {}
        want = (key, str(src.get('analysis') or ''), str(src.get('step') or ''),
                str(src.get('plot_type') or ''), src.get('points') is True)
        if want not in reads:
            try:
                reads[want] = plot_series(want[0], want[1], want[2], '', want[3],
                                          points=want[4], stored=True, memo=memo)
            except Exception as e:                      # noqa: BLE001
                reads[want] = e
        sr = reads[want]
        if isinstance(sr, Exception):
            row['reason'] = f'could not be read ({type(sr).__name__}: {sr})'
            continue
        if not sr:
            row['reason'] = 'no stored plot data'
            continue
        axes, ai = sr['axes'], pick.get('axis')
        if not isinstance(ai, int) or isinstance(ai, bool) or not 0 <= ai < len(axes):
            row['reason'] = 'not found in its stored plot (stored again since?)'
            continue
        found: List[Dict[str, Any]] = []
        for kind in ('lines', 'points'):
            have = axes[ai].get(kind) or []
            want, labs = pick.get(kind) or [], pick.get(kind + '_labels') or []
            for j, i in enumerate(want):
                ok = (isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(have)
                      and j < len(labs) and have[i].get('label', '') == str(labs[j]))
                if not ok:
                    found = []
                    row['reason'] = 'not found in its stored plot (stored again since?)'
                    break
                found.append(have[i])
            if row['reason']:
                break
        if row['reason']:
            continue
        parts = []
        for ln in found:
            pts = [[a, b] for a, b in zip(ln.get('x') or [], ln.get('y') or [])
                   if _finite(a) and _finite(b)]
            if pick.get('power') is True:
                pts = [[a, a * b * 1000.0] for a, b in pts]
            parts.append(pts)
        # Lines come before points in a pick: put back in x order (a short
        # hold's decay windows, stored as points, before a long one's line).
        parts = sorted((p for p in parts if p), key=lambda p: p[0][0])
        if x == 'cyc':
            if key not in maps:
                try:
                    maps[key] = rh_cycle_map(_read_record(key, memo)[1])
                except Exception as e:                  # noqa: BLE001
                    maps[key] = e
            if isinstance(maps[key], Exception):
                row['reason'] = f'could not be read ({type(maps[key]).__name__}: {maps[key]})'
                continue
            if maps[key] is None:
                row['reason'] = 'no RH cycles counted'
                continue
            parts = [[[cycles_at(*maps[key], a), b] for a, b in p] for p in parts]
        row['parts'] = parts
        if not row['parts']:
            row['reason'] = 'no points stored'
    return out


def comparable_groups(sources: List[Dict[str, str]]) -> Dict[str, int]:
    """{plot_type: count} for a staged selection.

    The comparison script overlays plots of the same type and skips any type
    with fewer than two, so a selection spanning four analyses with one plot
    each produces nothing. Knowing that up front turns a failed job into an
    immediate, explicable rejection.
    """
    counts: Dict[str, int] = {}
    for s in sources:
        pt = s.get('plot_type') or 'unknown'
        counts[pt] = counts.get(pt, 0) + 1
    return counts
