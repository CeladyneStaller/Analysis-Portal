"""
Duplicate detection by exact summary matching.
===============================================

Pure functions. No network, no environment. Nothing here writes anything.

The rule, from DUPLICATE_DETECTION_ANALYSIS.md:

    Two records are the same measurement when every (analysis, step) unit
    present in BOTH has every numeric summary field equal exactly, and the
    overlap contains at least MIN_MATCHED_FIELDS such fields.

Three measured properties this rests on:

  * Re-analysis of the same source is bit-identical — the pipeline is
    deterministic, so identical input gives identical floats.
  * Genuinely different cells share zero exact matches, even 0.2 mV apart.
    A polcurve unit carries ten independent doubles; coincidence is not a
    risk worth modelling.
  * Per-file analysis is batch-independent, so a partial run and a full run
    agree exactly on the files they share. This is why the comparison is over
    the OVERLAP rather than the union — requiring identical unit sets would
    reject the partial-versus-full case, which is the main thing being solved.

What this detects is "the same measurement analysed twice", not "the same
physical cell measured twice". The second cannot be established from analysis
output and needs something the operator knows.
"""

import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Set, Tuple

from scripts.helpers.record import _annotate_summary_rows

# Floor on how much evidence counts as a match. Some units carry no numeric
# summary at all — OCV emits none, cleaning few — so two records overlapping
# only on those would match VACUOUSLY: nothing compared, no contradiction
# found, duplicate declared. Three is comfortable against ten fields in a
# polcurve unit and five in an EIS one; one field is already strong.
MIN_MATCHED_FIELDS = 3

# Not measurements.
_SKIP_FIELDS = {'Label', 'label', 'Analysis'}

UnitKey = Tuple[str, str]           # (analysis, step)
Fingerprint = Dict[UnitKey, Dict[str, float]]


# ─────────────────────────────────────────────────────────────────────
#  Fingerprint
# ─────────────────────────────────────────────────────────────────────

def summary_fingerprint(detail: Dict[str, Any]) -> Fingerprint:
    """{(analysis, step): {field: value}} at full precision.

    Values are taken from the detail record's `summary`, not from the index's
    `key_values`: the index rounds to six significant figures, which makes
    collisions strictly more likely. Two values a nanovolt apart collide at
    6 sf and stay distinct as doubles.

    Rows are attributed to units by `_annotate_summary_rows`, which uses the
    row's `Analysis` when the Full Analysis orchestrator stamped one and
    otherwise recovers it from the plot name containing the row's label.

    A field appearing twice within one unit with *different* values is dropped
    rather than resolved. That happens when two rows attribute to the same
    unit — two files sharing a step, say — and a value that is ambiguous inside
    one record cannot mean anything compared across two.
    """
    fields: Fingerprint = {}
    ambiguous: Dict[UnitKey, Set[str]] = {}

    for ann in _annotate_summary_rows(detail):
        key: UnitKey = (str(ann.get('_bucket') or ''), str(ann.get('_step') or ''))
        row = ann.get('row') or {}
        bucket_fields = fields.setdefault(key, {})
        amb = ambiguous.setdefault(key, set())
        for name, value in row.items():
            if name in _SKIP_FIELDS or isinstance(value, bool):
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                continue
            v = float(value)
            if name in bucket_fields and bucket_fields[name] != v:
                amb.add(name)
            else:
                bucket_fields[name] = v

    for key, names in ambiguous.items():
        for name in names:
            fields.get(key, {}).pop(name, None)

    return {k: v for k, v in fields.items() if v}


def values_fingerprint(detail: Dict[str, Any]) -> Fingerprint:
    """{(analysis, step): {field: value}} from per-plot `values`.

    Records written before the tier-1 summary existed — every Full Analysis run
    up to the orchestrator fix — carry `summary: []`. Their metrics are still
    present as `values` parsed from the plot annotations, so they are not
    unmatchable, just less precise: annotation text is rounded for display, so
    an OCV reads 0.9 rather than 0.9001322.

    Lower precision means weaker evidence, which is why this is a fallback and
    not the primary source. The MIN_MATCHED_FIELDS floor still applies, so a
    match needs several display-rounded fields to agree.
    """
    out: Fingerprint = {}
    for bucket, plots in (detail.get('metrics') or {}).items():
        for _name, entry in plots.items():
            step = str((entry.get('conditions') or {}).get('step') or '')
            key: UnitKey = (bucket, step)
            fields = out.setdefault(key, {})
            for name, value in (entry.get('values') or {}).items():
                if isinstance(value, dict):
                    value = value.get('value')
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                if not math.isfinite(value):
                    continue
                v = float(value)
                if name in fields and fields[name] != v:
                    fields[name] = math.nan      # ambiguous; dropped below
                else:
                    fields.setdefault(name, v)
    cleaned = {k: {n: v for n, v in f.items() if math.isfinite(v)}
               for k, f in out.items()}
    return {k: v for k, v in cleaned.items() if v}


def best_fingerprints(a: Dict[str, Any], b: Dict[str, Any]
                      ) -> Tuple[Fingerprint, Fingerprint, str]:
    """The most precise fingerprint pair the two records can both support.

    Summary values are full-precision doubles; annotation values are rounded
    for display. Comparing one against the other would be meaningless — the
    field names differ too ('OCV' versus 'OCV', but 'V_at_1Acm2' versus
    'V @ 1 A/cm²') — so both sides must come from the same source.
    """
    fa, fb = summary_fingerprint(a), summary_fingerprint(b)
    if fa and fb:
        return fa, fb, 'summary'
    return values_fingerprint(a), values_fingerprint(b), 'plot values'


def fingerprint_digest(fp: Fingerprint) -> str:
    """Stable hash of a whole fingerprint. Equal digests mean equal content."""
    payload = sorted(
        (a, s, sorted(f.items())) for (a, s), f in fp.items())
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────
#  Comparison
# ─────────────────────────────────────────────────────────────────────

class MatchResult:
    """Why two records did or did not match.

    Deliberately reports rather than just deciding: a report that cannot show
    which fields agreed, or which one field disagreed, is not reviewable — and
    reviewability is the whole basis for trusting this before it merges
    anything.
    """

    __slots__ = ('is_duplicate', 'matched_fields', 'overlap_units',
                 'contradiction', 'reason', 'source')

    def __init__(self, is_duplicate: bool, matched_fields: int,
                 overlap_units: List[UnitKey],
                 contradiction: Optional[Tuple[str, str, str, float, float]],
                 reason: str):
        self.is_duplicate = is_duplicate
        self.matched_fields = matched_fields
        self.overlap_units = overlap_units
        self.contradiction = contradiction   # analysis, step, field, a, b
        self.reason = reason
        # Which fingerprint the comparison used. 'plot values' is the rounded
        # fallback for records predating the tier-1 summary — weaker evidence,
        # and worth showing in a report so the reader can weight it.
        self.source = 'summary'

    def __repr__(self):
        return (f'MatchResult(duplicate={self.is_duplicate}, '
                f'matched={self.matched_fields}, reason={self.reason!r})')

    def describe(self) -> str:
        if self.is_duplicate:
            units = ', '.join(f'{a}/{s or "—"}' for a, s in self.overlap_units)
            via = '' if self.source == 'summary' else f' via {self.source}'
            return (f'duplicate — {self.matched_fields} field(s) matched '
                    f'exactly across {units}{via}')
        if self.contradiction:
            a, s, f, va, vb = self.contradiction
            return (f'not a duplicate — {a}/{s or "—"} {f} differs '
                    f'({va!r} vs {vb!r})')
        return f'not a duplicate — {self.reason}'


def compare_records(a: Dict[str, Any], b: Dict[str, Any],
                    min_fields: int = MIN_MATCHED_FIELDS) -> MatchResult:
    """Compare two detail records under the exact-overlap rule."""
    fa, fb, source = best_fingerprints(a, b)
    res = compare_fingerprints(fa, fb, min_fields=min_fields)
    res.source = source
    return res


def compare_fingerprints(fa: Fingerprint, fb: Fingerprint,
                         min_fields: int = MIN_MATCHED_FIELDS) -> MatchResult:
    overlap = sorted(set(fa) & set(fb))
    if not overlap:
        return MatchResult(False, 0, [], None, 'no shared analysis units')

    matched = 0
    compared: List[UnitKey] = []
    for key in overlap:
        shared = sorted(set(fa[key]) & set(fb[key]))
        if not shared:
            continue
        for name in shared:
            va, vb = fa[key][name], fb[key][name]
            if va != vb:
                # One disagreement is decisive. Different measurements of
                # different cells never agree exactly, so a single differing
                # field means these are not the same source data.
                return MatchResult(False, matched, compared,
                                   (key[0], key[1], name, va, vb),
                                   'field values differ')
        matched += len(shared)
        compared.append(key)

    if matched < min_fields:
        return MatchResult(
            False, matched, compared, None,
            f'only {matched} field(s) in the overlap, below the floor of '
            f'{min_fields} — too little evidence to call it a duplicate')

    return MatchResult(True, matched, compared, None, 'exact match on overlap')


# ─────────────────────────────────────────────────────────────────────
#  Index prefilter
# ─────────────────────────────────────────────────────────────────────

def normalised_name(name: Any) -> str:
    """Sample name reduced to alphanumerics, lowercased.

    '260819_60h-6EG_CT2o1-FCS6' and '260819_60h-6EG_CT2o1_FCS6' differ by a
    single punctuation character and are the same cell typed two ways. Data
    matching cannot see that — it compares measurements, and a re-analysis that
    included a different ECSA file produces genuinely different numbers — but
    the names say it plainly.

    Checked against the live index: nine replicate-style names
    (BM1-Qual1, GSMA-Qual-1, GSMA-Qual2_, …) all stay distinct under this,
    so it separates spelling variants from real replicates.
    """
    if not name:
        return ''          # str(None) would normalise to 'none' and collide
    return ''.join(c for c in str(name).lower() if c.isalnum())


def name_prefilter(index: Dict[str, Any], require_same_date: bool = True
                   ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Entry pairs whose sample names differ only in punctuation or case.

    A far stronger signal than measurement matching for the case it covers, and
    it needs no detail bins at all.

    `require_same_date` additionally demands the same `run_date`, which is
    parsed from the name's own YYMMDD prefix. Two entries agreeing on both the
    normalised name and the date are the same cell to any useful certainty.
    """
    runs = index.get('runs') or []
    buckets: Dict[Any, List[Dict[str, Any]]] = {}
    for e in runs:
        name = normalised_name(e.get('sample_name'))
        if not name:
            continue
        key = (name, e.get('run_date')) if require_same_date else name
        buckets.setdefault(key, []).append(e)

    pairs = []
    for group in buckets.values():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                # Identical spellings are the sample-keyed case, handled by
                # merging on push; only variants need a decision here.
                if group[i].get('sample_name') != group[j].get('sample_name'):
                    pairs.append((group[i], group[j]))
    return pairs


def field_agreement(a: Dict[str, Any], b: Dict[str, Any]
                    ) -> Tuple[int, List[Tuple[str, str, str, float, float]]]:
    """(fields agreeing, fields differing) across the overlap.

    For a pair matched on name rather than data, this is the diagnostic: it
    shows what the two analyses agreed on and exactly where they parted, which
    is what tells you whether the second run changed inputs or parameters.
    """
    fa, fb, _src = best_fingerprints(a, b)
    agreed = 0
    differed: List[Tuple[str, str, str, float, float]] = []
    for key in sorted(set(fa) & set(fb)):
        for name in sorted(set(fa[key]) & set(fb[key])):
            va, vb = fa[key][name], fb[key][name]
            if va == vb:
                agreed += 1
            else:
                differed.append((key[0], key[1], name, va, vb))
    return agreed, differed


def _unit_hashes(entry: Dict[str, Any]) -> Set[str]:
    """One hash per (Analysis, step, field, value) in the index entry.

    Per field, deliberately, not per unit. Hashing the whole `key_values` dict
    means a record carrying fewer keys can never match one carrying more — and
    that happens routinely, because the whitelist has grown: entries written
    before `j @ V` was added hold two polcurve keys where newer ones hold seven.
    Requiring the dicts to be equal made those invisible to the prefilter even
    though the keys they share agree exactly, so the confirm step never got to
    see them.

    Recall is what matters here. A candidate that survives is confirmed at full
    precision afterwards, so an extra candidate costs one cached fetch, while a
    missed one is a duplicate that is never found.
    """
    out: Set[str] = set()
    for unit in entry.get('Data') or []:
        analysis = str(unit.get('Analysis', ''))
        step = str(unit.get('step', ''))
        for name, value in (unit.get('key_values') or {}).items():
            payload = [analysis, step, str(name), value]
            out.add(hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16])
    return out


def index_prefilter(index: Dict[str, Any],
                    skip_same_sample: bool = True
                    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Candidate pairs sharing at least one index unit.

    Entries already sharing a sample name are skipped by default: sample-keyed
    merging handles those, and including them would flood the report with pairs
    that need no decision.
    """
    runs = index.get('runs') or []
    hashes = [(_unit_hashes(e), e) for e in runs]

    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for i in range(len(hashes)):
        hi, ei = hashes[i]
        if not hi:
            continue
        for j in range(i + 1, len(hashes)):
            hj, ej = hashes[j]
            if not hj or not (hi & hj):
                continue
            if skip_same_sample and ei.get('sample_name') == ej.get('sample_name'):
                continue
            pairs.append((ei, ej))
    return pairs


def group_matches(pairs: List[Tuple[str, str]]) -> List[List[str]]:
    """Collapse matching pairs into groups by transitive closure.

    Three records matching each other should be merged in one pass rather than
    pairwise, which would rewrite the same bin repeatedly.
    """
    parent: Dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    groups: Dict[str, List[str]] = {}
    for node in parent:
        groups.setdefault(find(node), []).append(node)
    return [sorted(v) for v in groups.values() if len(v) > 1]
