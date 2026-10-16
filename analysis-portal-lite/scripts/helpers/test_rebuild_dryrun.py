"""The offline rehearsal of the electrolyzer rebuild (rebuild_dryrun.py).

Run against backup files made the way the admin modal's ⭳ Backup makes them,
from the old-era store in fixtures/elx_eras_store.json: a whole one, one
missing records, and files that are not backups. It must never make an HTTP
call, whatever the environment holds.

Run:  python3 scripts/helpers/test_rebuild_dryrun.py
"""

import contextlib
import copy
import io
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# What a shell with the real settings exported would hold. The tool must
# ignore it.
os.environ.update(JSONBIN_API_KEY='real-looking-key', JSONBIN_COLLECTION_ID='real',
                  JSONBIN_INDEX_BIN_ID='REAL-INDEX')

from scripts.helpers import admin, jsonbin                          # noqa: E402
from scripts.helpers.record import attach_sidecars                  # noqa: E402

passed = failed = 0


def check(name, got, want):
    global passed, failed
    if got == want:
        passed += 1
    else:
        failed += 1
        print(f'  FAIL {name}\n       got  {got!r}\n       want {want!r}')


FIX = json.loads((Path(__file__).resolve().parent / 'fixtures'
                  / 'elx_eras_store.json').read_text(encoding='utf-8'))
bins = {'REAL-INDEX': copy.deepcopy(FIX['index'])}
for bid, rec in FIX['records'].items():
    r = copy.deepcopy(rec)
    attach_sidecars(r, r.pop('plots'))
    bins[bid] = r


def serve(url, method='GET', body=None, extra_headers=None):
    assert method == 'GET', method
    return {'record': copy.deepcopy(bins[url.rstrip('/').split('/')[-2]])}


jsonbin._request = serve
backup = admin.build_backup(None)            # what ⭳ Backup downloads
tmp = Path(tempfile.mkdtemp())

# Any HTTP call from here on is a failure, however it is made.
calls = []


def no_network(*a, **k):
    calls.append(a)
    raise RuntimeError('network used')


jsonbin._ureq.urlopen = no_network
from scripts.helpers import rebuild_dryrun                           # noqa: E402


def run(payload, *extra):
    f = tmp / f'b{len(list(tmp.iterdir()))}.json'
    f.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = rebuild_dryrun.main([str(f), *extra])
    return code, out.getvalue()


print('a whole backup')
code, out = run(backup, '--json', str(tmp / 'report.json'))
check('checks out', code, 0)
check('reports the samples it would rebuild', '11 to rebuild' in out, True)
check('and finds nothing left after applying', 'the rebuild is idempotent' in out, True)
rep = json.loads((tmp / 'report.json').read_text())
check('writes the full reports', sorted(rep), ['applied', 'missed', 'not_in_backup',
                                              'preview', 'second_preview'])
check('nothing was missed by the scan', rep['missed'], [])
check('the real settings were overwritten in-process',
      os.environ['JSONBIN_INDEX_BIN_ID'], 'OFFLINE-INDEX')

print('a backup missing records')
part = copy.deepcopy(backup)
part['purged'] = part['purged'][2:]
code, out = run(part)
check('is incomplete', code, 3)
check('and says which samples were not checked', 'cannot be checked' in out, True)

print('an entry with no bin, which no backup can supply')
odd = copy.deepcopy(backup)
odd['index']['runs'].append({'sample_name': 'NOBIN', 'script': 'Electrolyzer Pol Curve',
                             'Data': []})
code, out = run(odd)
check('is reported but leaves the status alone', (code, 'NOBIN: not checked' in out), (0, True))

print('files that are not backups')
check('not JSON', run('{nope')[0], 2)
check('JSON but no index', run({'kind': 'analysis-portal-backup'})[0], 2)
check('something else entirely', run({'kind': 'other', 'index': {'runs': []}})[0], 2)

check('no HTTP call was made, from any of it', calls, [])

print(f'\n{passed} passed, {failed} failed')
sys.exit(1 if failed else 0)
