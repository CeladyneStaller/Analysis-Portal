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
                 'contradiction', 'reason')

    def __init__(self, is_duplicate: bool, matched_fields: int,
                 overlap_units: List[UnitKey],
                 contradiction: Optional[Tuple[str, str, str, float, float]],
                 reason: str):
        self.is_duplicate = is_duplicate
        self.matched_fields = matched_fields
        self.overlap_units = overlap_units
        self.contradiction = contradiction   # analysis, step, field, a, b
        self.reason = reason

    def __repr__(self):
        return (f'MatchResult(duplicate={self.is_duplicate}, '
                f'matched={self.matched_fields}, reason={self.reason!r})')

    def describe(self) -> str:
        if self.is_duplicate:
            units = ', '.join(f'{a}/{s or "—"}' for a, s in self.overlap_units)
            return (f'duplicate — {self.matched_fields} field(s) matched '
                    f'exactly across {units}')
        if self.contradiction:
            a, s, f, va, vb = self.contradiction
            return (f'not a duplicate — {a}/{s or "—"} {f} differs '
                    f'({va!r} vs {vb!r})')
        return f'not a duplicate — {self.reason}'


def compare_records(a: Dict[str, Any], b: Dict[str, Any],
                    min_fields: int = MIN_MATCHED_FIELDS) -> MatchResult:
    """Compare two detail records under the exact-overlap rule."""
    fa, fb = summary_fingerprint(a), summary_fingerprint(b)
    return compare_fingerprints(fa, fb, min_fields=min_fields)


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

def _unit_hashes(entry: Dict[str, Any]) -> Set[str]:
    """Hashes over each index unit's (Analysis, step, key_values).

    Cheap: the index is already fetched, and this needs no detail bins. Recall
    is what matters — a candidate that survives here is confirmed at full
    precision afterwards, so false candidates cost only a fetch.
    """
    out: Set[str] = set()
    for unit in entry.get('Data') or []:
        kv = unit.get('key_values') or {}
        if not kv:
            continue      # nothing to match on; the confirm step would reject
        payload = [unit.get('Analysis', ''), unit.get('step', ''),
                   sorted((str(k), v) for k, v in kv.items())]
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
