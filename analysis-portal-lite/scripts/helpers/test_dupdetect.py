"""
Offline harness for dupdetect.py. No network.

Run:  python3 scripts/helpers/test_dupdetect.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.helpers.dupdetect import (  # noqa: E402
    MIN_MATCHED_FIELDS, compare_records, compare_fingerprints,
    fingerprint_digest, group_matches, index_prefilter, summary_fingerprint,
)

_passed = 0
_failed = 0


def check(name, got, want):
    global _passed, _failed
    if got == want:
        _passed += 1
    else:
        _failed += 1
        print(f"  FAIL {name}\n       got:  {got!r}\n       want: {want!r}")


def check_true(name, cond, detail=''):
    global _passed, _failed
    if cond:
        _passed += 1
    else:
        _failed += 1
        print(f"  FAIL {name} {detail}")


def rec(summary, metrics=None):
    """A minimal detail record carrying summary rows."""
    return {'schema': 2, 'sample_name': 's', 'summary': summary,
            'metrics': metrics or {'polcurve': {
                'polcurve_b4': {'conditions': {'step': 'b4'}, 'values': {}}}}}


POL = lambda **kw: dict({'Label': 'b4', 'Analysis': 'polcurve',
                         'OCV': 0.9001322, 'V_at_1Acm2': 0.6074109,
                         'peak_power_W_cm2': 0.6766823}, **kw)

print("fingerprint")
fp = summary_fingerprint(rec([POL()]))
check('one unit', sorted(fp), [('polcurve', 'b4')])
check('numeric fields kept', sorted(fp[('polcurve', 'b4')]),
      ['OCV', 'V_at_1Acm2', 'peak_power_W_cm2'])
check_true('Label and Analysis are not fields',
           'Label' not in fp[('polcurve', 'b4')]
           and 'Analysis' not in fp[('polcurve', 'b4')])

# Full precision, not the index's six significant figures — rounding makes
# collisions strictly more likely.
check('value stored unrounded',
      summary_fingerprint(rec([POL(OCV=0.90013222222222219)]))
      [('polcurve', 'b4')]['OCV'], 0.90013222222222219)

# Non-finite values cannot be compared and are not measurements.
nf = summary_fingerprint(rec([POL(OCV=float('nan'), extra=float('inf'))]))
check('non-finite excluded', sorted(nf[('polcurve', 'b4')]),
      ['V_at_1Acm2', 'peak_power_W_cm2'])

# Booleans are not measurements either.
check_true('bool excluded',
           'flag' not in summary_fingerprint(rec([POL(flag=True)]))
           [('polcurve', 'b4')])

# A field that is ambiguous inside one record cannot mean anything across two.
amb = summary_fingerprint({'schema': 2, 'summary': [
    {'Label': 'b4_one', 'Analysis': 'polcurve', 'OCV': 0.900, 'P': 0.67, 'V': 0.6},
    {'Label': 'b4_two', 'Analysis': 'polcurve', 'OCV': 0.950, 'P': 0.67, 'V': 0.6}],
    'metrics': {'polcurve': {
        'polcurve_b4_one': {'conditions': {'step': 'b4'}, 'values': {}},
        'polcurve_b4_two': {'conditions': {'step': 'b4'}, 'values': {}}}}})
check('ambiguous field dropped', sorted(amb[('polcurve', 'b4')]), ['P', 'V'])

check('digest is stable',
      fingerprint_digest(summary_fingerprint(rec([POL()]))),
      fingerprint_digest(summary_fingerprint(rec([POL()]))))
check_true('digest differs on different content',
           fingerprint_digest(summary_fingerprint(rec([POL()])))
           != fingerprint_digest(summary_fingerprint(rec([POL(OCV=0.5)]))))

print("matching")
same = compare_records(rec([POL()]), rec([POL()]))
check_true('identical records match', same.is_duplicate)
check('all fields counted', same.matched_fields, 3)
check('overlap reported', same.overlap_units, [('polcurve', 'b4')])
check_true('no contradiction on a match', same.contradiction is None)

# One differing field is decisive: different cells never agree exactly.
diff = compare_records(rec([POL()]), rec([POL(OCV=0.9001323)]))
check_true('one differing field rejects', not diff.is_duplicate)
check('contradiction names the field', diff.contradiction[2], 'OCV')
check('contradiction carries both values',
      (diff.contradiction[3], diff.contradiction[4]), (0.9001322, 0.9001323))
check_true('describe explains the rejection',
           'OCV differs' in diff.describe(), f'({diff.describe()})')

# Overlap, not union — this is what makes partial-versus-full work.
partial = rec([POL()])
full = rec([POL(), {'Label': 'c6', 'Analysis': 'polcurve', 'OCV': 0.88,
                    'V_at_1Acm2': 0.59, 'peak_power_W_cm2': 0.64}],
           metrics={'polcurve': {
               'polcurve_b4': {'conditions': {'step': 'b4'}, 'values': {}},
               'polcurve_c6': {'conditions': {'step': 'c6'}, 'values': {}}}})
pf = compare_records(partial, full)
check_true('partial matches full on the shared unit', pf.is_duplicate)
check('only the shared unit compared', pf.overlap_units, [('polcurve', 'b4')])

print("evidence floor")
# Units carrying no numeric summary must not match vacuously.
vac = compare_records(rec([{'Label': 'b4', 'Analysis': 'ocv'}]),
                      rec([{'Label': 'b4', 'Analysis': 'ocv'}]))
check_true('vacuous overlap rejected', not vac.is_duplicate)
check('vacuous overlap counts nothing', vac.matched_fields, 0)

thin = rec([{'Label': 'b4', 'Analysis': 'eis', 'HFR': 0.045}])
check_true(f'below the floor of {MIN_MATCHED_FIELDS} rejected',
           not compare_records(thin, thin).is_duplicate)
check_true('floor is configurable',
           compare_records(thin, thin, min_fields=1).is_duplicate)
check_true('rejection explains the floor',
           'below the floor' in compare_records(thin, thin).reason)

print("no overlap")
a = rec([POL()])
b = rec([{'Label': 'a2', 'Analysis': 'eis', 'HFR': 0.045, 'R': 0.05, 'X': 1.0}],
        metrics={'eis': {'eis_a2': {'conditions': {'step': 'a2'}, 'values': {}}}})
no = compare_records(a, b)
check_true('different analyses do not match', not no.is_duplicate)
check('reason is the absent overlap', no.reason, 'no shared analysis units')
check('empty fingerprints do not match',
      compare_fingerprints({}, {}).is_duplicate, False)

print("index prefilter")


def ent(job, sample, kv, step='b4'):
    return {'job_id': job, 'sample_name': sample, 'bin_id': 'B' + job,
            'Data': [{'Analysis': 'polcurve', 'step': step, 'key_values': kv}]}


idx = {'runs': [ent('1', 'A', {'OCV': 0.9}), ent('2', 'B', {'OCV': 0.9}),
                ent('3', 'C', {'OCV': 0.9}), ent('4', 'D', {'OCV': 0.8}),
                ent('5', 'A', {'OCV': 0.9})]}
pairs = index_prefilter(idx)
names = [(a['sample_name'], b['sample_name']) for a, b in pairs]
check_true('matching key_values pair up', ('A', 'B') in names)
check_true('same-sample pairs skipped',
           not any(x == y for x, y in names))
check_true('different key_values excluded',
           not any('D' in p for p in names))
check_true('same-sample pairs included when asked',
           any(a['sample_name'] == b['sample_name']
               for a, b in index_prefilter(idx, skip_same_sample=False)))

# A unit with no key_values has nothing to match on and must not be a candidate.
bare = {'runs': [ent('1', 'A', {}), ent('2', 'B', {})]}
check('units without key_values are not candidates', index_prefilter(bare), [])

# Differing step is a different unit.
steps = {'runs': [ent('1', 'A', {'OCV': 0.9}, step='b4'),
                  ent('2', 'B', {'OCV': 0.9}, step='c6')]}
check('different steps are different units', index_prefilter(steps), [])

print("grouping")
check('transitive closure', group_matches([('A', 'B'), ('B', 'C')]), [['A', 'B', 'C']])
check('disjoint groups stay separate',
      sorted(group_matches([('A', 'B'), ('C', 'D')])), [['A', 'B'], ['C', 'D']])
check('no pairs, no groups', group_matches([]), [])

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
