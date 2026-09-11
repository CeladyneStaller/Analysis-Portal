#!/usr/bin/env python3
"""Offline tests for scripts/helpers/admin.py — no network, mocked JSONBin."""

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.helpers import admin, jsonbin                       # noqa: E402
from scripts.helpers.record import encode_sidecars                # noqa: E402

_passed = _failed = 0


def check(label, got, want):
    global _passed, _failed
    if got == want:
        _passed += 1
    else:
        _failed += 1
        print(f"  FAIL {label}\n       got:  {got!r}\n       want: {want!r}")


def check_true(label, cond, extra=''):
    global _passed, _failed
    if cond:
        _passed += 1
    else:
        _failed += 1
        print(f"  FAIL {label} {extra}")


def raises(label, fn, fragment):
    global _passed, _failed
    try:
        fn()
    except Exception as e:
        if fragment.lower() in str(e).lower():
            _passed += 1
        else:
            _failed += 1
            print(f"  FAIL {label}\n       raised: {e}\n       wanted: ...{fragment}...")
        return
    _failed += 1
    print(f"  FAIL {label} — did not raise")


# ── mock store ───────────────────────────────────────────────────────
SC = {'plot_type': 'polcurve', 'data': {'axes': [{
    'title': 't', 'xlabel': 'j', 'ylabel': 'V', 'texts': [],
    'axhlines': [], 'axvlines': [],
    'lines': [{'label': 'Cell voltage', 'x': [0, 1], 'y': [0.9, 0.6]}]}]}}

BINS, INDEX, DELETED, CREATED = {}, {}, [], []
INDEX_BIN = 'INDEXBIN'


def detail(job, name, ocv, stand=None):
    d = {'schema': 2, 'job_id': job, 'sample_name': name, 'script': 'FCA',
         'timestamp': f'2026-08-{job[-2:]}T10:00:00Z',
         'input_files': [f'{job}.fcd'],
         'metrics': {'polcurve': {'polcurve_b4': {
             'conditions': {'step': 'b4'}, 'values': {}}}},
         'summary': [{'Label': 'b4', 'Analysis': 'polcurve', 'OCV': ocv,
                      'V_at_1Acm2': ocv - 0.29, 'peak_power_W_cm2': ocv - 0.22}],
         'sidecars': encode_sidecars({'polcurve_b4': SC})}
    if stand:
        d['stand'] = stand
    return d


def entry(job, name, bid, ocv, date, stand=None):
    e = {'job_id': job, 'sample_name': name, 'script': 'FCA',
         'timestamp': f'2026-08-{job[-2:]}T10:00:00Z', 'run_date': date,
         'bin_id': bid,
         'Data': [{'Analysis': 'polcurve', 'step': 'b4', 'Conditions': {},
                   'key_values': {'OCV': ocv}}]}
    if stand:
        e['stand'] = stand
    return e


def reset():
    global BINS, INDEX, DELETED, CREATED
    BINS = {
        'B01': detail('j01', '260819_Cell-A', 0.9012, 'Scribner'),
        'B02': detail('j02', '260819_Cell_A', 0.9012, 'Scribner'),
        'B03': detail('j03', '260820_Cell-B', 0.8974, 'FCTS'),
        'B04': detail('j04', '260821_Cell-C', 0.8931),
    }
    INDEX = {'schema': 2, 'runs': [
        entry('j01', '260819_Cell-A', 'B01', 0.9012, '2026-08-19', 'Scribner'),
        entry('j02', '260819_Cell_A', 'B02', 0.9012, '2026-08-19', 'Scribner'),
        entry('j03', '260820_Cell-B', 'B03', 0.8974, '2026-08-20', 'FCTS'),
        entry('j04', '260821_Cell-C', 'B04', 0.8931, '2026-08-21'),
    ]}
    DELETED[:] = []
    CREATED[:] = []


def mock(url, method='GET', body=None, extra_headers=None):
    global INDEX
    if method == 'DELETE':
        for b in list(BINS):
            if url.endswith('/' + b):
                del BINS[b]
                DELETED.append(b)
                return {}
        return {}
    if method == 'POST':
        bid = f'BNEW{len(CREATED) + 1}'
        BINS[bid] = copy.deepcopy(body)
        CREATED.append(bid)
        return {'metadata': {'id': bid}}
    if method == 'PUT':
        for b in BINS:
            if url.endswith('/' + b):
                BINS[b] = copy.deepcopy(body)
                return {'record': body}
        INDEX = copy.deepcopy(body)
        return {'record': body}
    for b in BINS:
        if f'/{b}/' in url or url.endswith('/' + b):
            return {'record': copy.deepcopy(BINS[b])}
    # A detail-bin URL that matches nothing is a deleted bin, and has to fail
    # like one — returning the index instead made every purged sample look as
    # though its bin were still there.
    if url.endswith('/latest') and not url.rstrip('/latest').endswith(INDEX_BIN):
        raise RuntimeError('404 bin not found')
    return {'record': copy.deepcopy(INDEX)}


jsonbin._request = mock
reset()

# ── inventory ────────────────────────────────────────────────────────
print("inventory")
inv = admin.list_samples()
check('every sample listed', inv['total'], 4)
check_true('sorted newest first',
           inv['samples'][0]['sample_name'] == '260821_Cell-C')
check_true('carries what the panels need',
           all(k in inv['samples'][0]
               for k in ('key', 'sample_name', 'run_date', 'stand', 'units')))

# ── duplicates ───────────────────────────────────────────────────────
print("duplicates")
reset()
dups = admin.find_duplicates()
check('the spelling pair is found', len(dups['groups']), 1)
g = dups['groups'][0]
check('both members', sorted(g['names']), ['260819_Cell-A', '260819_Cell_A'])
check_true('reported with its evidence', g['evidence'] in ('name', 'measurements'))

reset()
prev = admin.merge_duplicate_group(['B01', 'B02'], '260819_Cell-A')
check('preview writes nothing', prev['applied'], False)
check('index untouched by a preview', len(INDEX['runs']), 4)
check('names the surviving bin', prev['survivor_bin'], 'B02')

# The name has to come from the group. Renaming during a merge is almost
# always a mistyped command, and it would discard the right name silently.
raises('a name from outside the group is refused',
       lambda: admin.merge_duplicate_group(['B01', 'B02'], 'something else'),
       'not one of this group')
raises('a merge needs two survivors',
       lambda: admin.merge_duplicate_group(['B01'], '260819_Cell-A'),
       'at least two')

reset()
res = admin.merge_duplicate_group(['B01', 'B02'], '260819_Cell-A', apply=True)
check('applied', res['applied'], True)
check('one entry fewer', len(INDEX['runs']), 3)
check('under the chosen name',
      sorted(e['sample_name'] for e in INDEX['runs'])[0], '260819_Cell-A')
check_true('the superseded bin is left in place, not destroyed',
           'B01' in BINS and DELETED == [])

# ── rename ───────────────────────────────────────────────────────────
print("rename")
reset()
prev = admin.rename_sample('B04', '260821_Cell-C-R2')
check('preview writes nothing', prev['applied'], False)
check('run date survives a kept prefix', prev['run_date_to'], '2026-08-21')
check_true('and is not flagged as lost', prev['run_date_lost'] is False)

prev = admin.rename_sample('B04', 'Cell-C-R2')
check_true('dropping the date prefix is flagged', prev['run_date_lost'] is True)
check('with no new date', prev['run_date_to'], None)

raises('a Windows-forbidden character is refused',
       lambda: admin.rename_sample('B04', 'Cell/C'), 'windows does not allow')
raises('so are control characters',
       lambda: admin.rename_sample('B04', 'Cell\x01C'), 'control characters')
raises('an empty name is refused',
       lambda: admin.rename_sample('B04', '   '), 'required')
raises('renaming to the current name is refused',
       lambda: admin.rename_sample('B04', '260821_Cell-C'), 'already the name')
raises('an unknown key is refused',
       lambda: admin.rename_sample('NOPE', 'x'), 'no stored sample')

# The collision guard: sample-keyed merging identifies a sample by name, so
# two entries sharing one are merged into a single bin on the next push.
raises('renaming onto an existing name is refused',
       lambda: admin.rename_sample('B04', '260820_Cell-B'), 'already used')

reset()
res = admin.rename_sample('B04', '260821_Cell-C-R2', apply=True)
check('applied', res['applied'], True)
check('the index entry is renamed',
      [e['sample_name'] for e in INDEX['runs'] if e['bin_id'] == 'B04'],
      ['260821_Cell-C-R2'])
# The index entry is rebuilt from the detail record on the next push, so an
# index-only rename would silently revert.
check('the detail record is renamed too',
      BINS['B04']['sample_name'], '260821_Cell-C-R2')

reset()
admin.rename_sample('B04', 'Cell-C-nodate', apply=True)
kept = [e for e in INDEX['runs'] if e['bin_id'] == 'B04'][0]
check_true('a name with no date prefix drops run_date rather than keeping a stale one',
           'run_date' not in kept)

# ── test stand ───────────────────────────────────────────────────────
print("test stand")
reset()
prev = admin.set_stand(['B01', 'B02'], 'Scribner 1')
check('preview writes nothing', prev['applied'], False)
check('both would change', prev['total'], 2)
check('no family crossing here', prev['crossing_total'], 0)

prev = admin.set_stand(['B03'], 'Scribner 1')
check('an FCTS entry moving to Scribner is flagged', prev['crossing_total'], 1)
raises('and refused without acknowledgement',
       lambda: admin.set_stand(['B03'], 'Scribner 1', apply=True),
       'change family')

reset()
res = admin.set_stand(['B03'], 'Scribner 1', apply=True,
                      allow_family_change=True)
check('applied when acknowledged', res['applied'], True)
check('index updated',
      [e['stand'] for e in INDEX['runs'] if e['bin_id'] == 'B03'], ['Scribner 1'])
check('detail record updated too', BINS['B03']['stand'], 'Scribner 1')

raises('an unknown stand is refused',
       lambda: admin.set_stand(['B01'], 'Scribner 9'), 'not a known stand')
raises('an unknown key is refused',
       lambda: admin.set_stand(['NOPE'], 'Scribner 1'), 'no stored sample')

reset()
res = admin.set_stand(['B01'], 'Scribner', apply=True) \
    if 'Scribner' in admin.STAND_OPTIONS else None

# ── backup and delete ────────────────────────────────────────────────
print("backup and delete")
reset()
bak = admin.build_backup(['B04'])
check('a backup names itself', bak['kind'], 'analysis-portal-backup')
check('and carries the entry', len(bak['removed']), 1)
# An index entry alone cannot bring a sample back: the measurements are in
# the bin, so the record travels with it.
check('and the full detail record', len(bak['purged']), 1)
check_true('which holds the measurements',
           'sidecars' in bak['purged'][0]['record'])

reset()
prev = admin.delete_samples(['B04'])
check('preview writes nothing', prev['applied'], False)
check('index untouched', len(INDEX['runs']), 4)
raises('an unknown key aborts rather than deleting the rest',
       lambda: admin.delete_samples(['B04', 'NOPE'], apply=True),
       'no stored sample')

reset()
res = admin.delete_samples(['B04'], apply=True)
check('unlisted', len(INDEX['runs']), 3)
check_true('the bin survives an unlist', 'B04' in BINS and DELETED == [])

reset()
res = admin.delete_samples(['B04'], purge=True, apply=True)
check('purged from the index', len(INDEX['runs']), 3)
check('and the bin destroyed', DELETED, ['B04'])
check_true('a backup was taken before destroying anything',
           res['backup']['purged'][0]['bin_id'] == 'B04')

# ── restore ──────────────────────────────────────────────────────────
print("restore")
reset()
bak = admin.build_backup(['B04'])
admin.delete_samples(['B04'], purge=True, apply=True)
info = admin.inspect_backup(bak)
check('the backup lists its sample', len(info['items']), 1)
check('recognised as purged', info['items'][0]['state'], 'purged')
check_true('and not blocked', info['items'][0]['blocked'] is False)

res = admin.restore_backup(bak, ['260821_Cell-C'])
check('preview writes nothing', res['applied'], False)
check('index still short', len(INDEX['runs']), 3)

res = admin.restore_backup(bak, ['260821_Cell-C'], apply=True)
check('restored', len(INDEX['runs']), 4)
# JSONBin will not resurrect a deleted bin, so the record is written as a new
# one and the entry repointed. The old bin_id in the backup is a label.
check('into a new bin', len(CREATED), 1)
restored = [e for e in INDEX['runs'] if e['sample_name'] == '260821_Cell-C'][0]
check('with the entry repointed', restored['bin_id'], CREATED[0])
check_true('and the measurements back',
           'sidecars' in BINS[CREATED[0]])

# The file also holds the whole index as it stood. Applying that wholesale
# would revert every analysis pushed since.
reset()
bak = admin.build_backup(['B04'])
admin.delete_samples(['B04'], apply=True)
INDEX['runs'].append(entry('j09', '260901_Later', 'B09', 0.88, '2026-09-01'))
admin.restore_backup(bak, ['260821_Cell-C'], apply=True)
check_true('a sample pushed after the backup survives the restore',
           any(e['sample_name'] == '260901_Later' for e in INDEX['runs']))
check('and the restored one is back', len(INDEX['runs']), 5)

# An unlisted sample only needs its entry back.
reset()
bak = admin.build_backup(['B04'])
admin.delete_samples(['B04'], apply=True)
info = admin.inspect_backup(bak)
check('recognised as unlisted', info['items'][0]['state'], 'unlisted')
admin.restore_backup(bak, ['260821_Cell-C'], apply=True)
check('re-indexed without a new bin', len(CREATED), 0)

# A name that exists again is the rename collision by another route.
reset()
bak = admin.build_backup(['B04'])
admin.delete_samples(['B04'], purge=True, apply=True)
INDEX['runs'].append(entry('j10', '260821_Cell-C', 'B10', 0.87, '2026-08-21'))
info = admin.inspect_backup(bak)
check_true('a reused name blocks the restore', info['items'][0]['blocked'] is True)
raises('and applying it is refused',
       lambda: admin.restore_backup(bak, ['260821_Cell-C'], apply=True),
       'stored under that name')

raises('a file that is not a backup is refused',
       lambda: admin.inspect_backup({'hello': 'world'}), 'lists no samples')
raises('and neither is a foreign document',
       lambda: admin.inspect_backup({'kind': 'something-else', 'removed': [1]}),
       'not an analysis portal backup')
raises('restoring a name the backup does not hold',
       lambda: admin.restore_backup(bak, ['nope']), 'none of those samples')

print("name-extension duplicates")
# The pair that motivated this: one name extends the other with a qualifier,
# which neither the equality prefilter nor the measurement prefilter can see
# unless the two runs agree on a stored value exactly.
from scripts.helpers.dupdetect import name_extension_prefilter
def _e(n, d='2026-08-07'):
    return {'sample_name': n, 'run_date': d, 'bin_id': n[:6], 'Data': []}
IDXN = {'runs': [_e('260807_Volvo-B2-(VB2-3)_Half-CCM'), _e('260807_Volvo-B2-(VB2-3)'),
                 _e('260807_Volvo-B2-(VB2-4)'), _e('260801_Volvo-B2-(VB2-3)', '2026-08-01')]}
got = name_extension_prefilter(IDXN)
check('the extension pair is proposed', len(got), 1)
check_true('and it is the right one',
           {got[0][0]['sample_name'], got[0][1]['sample_name']} ==
           {'260807_Volvo-B2-(VB2-3)_Half-CCM', '260807_Volvo-B2-(VB2-3)'})
# A sibling build shares the prefix legitimately and is not an extension.
check_true('a sibling name is not proposed',
           all('VB2-4' not in a['sample_name'] and 'VB2-4' not in b['sample_name']
               for a, b in got))
check('a different date is not proposed', len(name_extension_prefilter(IDXN)), 1)

reset()
# Found when the measurements agree.
res = admin.find_duplicates()
check_true('the spelling pair is still found', len(res['groups']) >= 1)
check_true('the report says how many pairs it compared', res['considered'] >= 1)

# A rejection is reported rather than dropped: a pair that was looked at and
# turned down must not look the same as one never considered.
reset()
BINS['B02']['summary'][0]['OCV'] = 0.90121          # differs below display precision
res = admin.find_duplicates()
# These two are also a spelling pair, so the name route still finds them —
# what matters is that the measurement route's refusal is now on the record
# instead of vanishing.
check_true('a pair rejected on measurements is reported',
           any(r['reason'] == 'field values differ' for r in res['rejected']))
diff = next(r for r in res['rejected'] if r['reason'] == 'field values differ')
check('naming the field that differed', diff['differing']['field'], 'OCV')
check_true('and both values, so the size of the gap is visible',
           diff['differing']['a'] != diff['differing']['b'])
check_true('the group that survives is on name evidence',
           res['groups'][0]['evidence'] in ('name', 'name extension'))

print("stand repair")
# The index could be demoted to the bare family while the detail record kept
# the number. This puts it back, and only where that exact demotion happened.
reset()
INDEX['runs'][0]['stand'] = 'Scribner'          # B01's record says 'Scribner'
BINS['B01']['stand'] = 'Scribner 1'
INDEX['runs'][2]['stand'] = 'FCTS 2'            # already correct
BINS['B03']['stand'] = 'FCTS 2'
prev = admin.repair_stands()
check('finds only the demoted entry', len(prev['repairs']), 1)
check('naming what the index lost', prev['repairs'][0]['index'], 'Scribner')
check('and what the record still holds', prev['repairs'][0]['record'], 'Scribner 1')
check('preview writes nothing', prev['applied'], False)
check('index untouched by a preview', INDEX['runs'][0]['stand'], 'Scribner')

res = admin.repair_stands(apply=True)
check('applied', res['applied'], True)
check('the number is back', INDEX['runs'][0]['stand'], 'Scribner 1')
check('a correct entry is left alone', INDEX['runs'][2]['stand'], 'FCTS 2')
check_true('and a backup was taken first', bool(res.get('backup')))

# A genuine disagreement is not this tool's business.
reset()
INDEX['runs'][0]['stand'] = 'FCTS 1'
BINS['B01']['stand'] = 'Scribner 2'
check('a cross-family disagreement is not "repaired"',
      len(admin.repair_stands()['repairs']), 0)

print("automatic backup")
# An offered backup is only as good as the habit of taking it. Every apply
# takes one first, so the guarantee is that a written change is a recoverable
# one — this asserts it for each operation rather than trusting the pattern.
reset()
r = admin.merge_duplicate_group(['B01', 'B02'], '260819_Cell-A', apply=True)
check_true('merge carries a backup', bool(r.get('backup')))
reset()
r = admin.rename_sample('B04', '260821_Cell-C-R2', apply=True)
check_true('rename carries a backup', bool(r.get('backup')))
reset()
r = admin.set_stand(['B01'], 'Scribner 1', apply=True)
check_true('stand change carries a backup', bool(r.get('backup')))
reset()
r = admin.delete_samples(['B04'], apply=True)
check_true('an unlist carries one too, not just a purge', bool(r.get('backup')))
check_true('and it holds the record, not only the entry',
           bool(r['backup']['purged']))
reset()
bak = admin.build_backup(['B04'])
admin.delete_samples(['B04'], apply=True)
r = admin.restore_backup(bak, ['260821_Cell-C'], apply=True)
check_true('restore carries one as well', bool(r.get('backup')))

# A preview must not take one: it writes nothing, and a backup on every
# keystroke of the rename field would be a fetch per character.
reset()
check_true('a preview takes no backup',
           'backup' not in admin.rename_sample('B04', 'x-y-z'))
check_true('nor does a delete preview',
           'backup' not in admin.delete_samples(['B04']))

# A record that cannot be backed up is one that cannot be restored, so the
# operation stops rather than proceeding without it.
reset()
_real = admin.build_backup
admin.build_backup = lambda *a, **k: (_ for _ in ()).throw(
    RuntimeError('detail bin unreadable'))
raises('a failed backup stops the write',
       lambda: admin.delete_samples(['B04'], apply=True), 'nothing was changed')
admin.build_backup = _real
check('and the index is untouched', len(INDEX['runs']), 4)
check_true('and the bin survives', 'B04' in BINS and DELETED == [])

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
