"""Admin writes and pushes, interleaved.

The admin handlers run in worker threads, so an admin change can land while a
push is between reading the index and writing it, and the other way round.
Each race below is staged deterministically: the in-memory JSONBin runs the
other writer at the exact moment the first one is mid-way.

  * a push keeps an admin change written while it was merging;
  * an admin write keeps a push that landed while it was working;
  * two admin writes keep each other's changes;
  * an admin write waits for a push holding its sample, a preview does not;
  * the rebuild holds every sample it rewrites;
  * a push re-reads the index and finds its own entry, not a namesake;
  * main.py: the admin handlers are plain functions — FastAPI runs those in
    worker threads instead of on the event loop — an applied change empties
    the View tab's caches, and pushes run on a thread of their own, so a
    push waiting for a sample holds up no job.

Run:  python3 scripts/helpers/test_admin_locks.py
"""

import copy
import inspect
import json
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.update(JSONBIN_API_KEY='k', JSONBIN_COLLECTION_ID='C',
                  JSONBIN_INDEX_BIN_ID='INDEX', ADMIN_CODE='letmein',
                  JOBS_DIR=tempfile.mkdtemp(), MPLBACKEND='Agg')

from scripts.helpers import admin, jsonbin, viewstore               # noqa: E402
from scripts.helpers.record import encode_sidecars                  # noqa: E402

passed = failed = 0


def check(name, got, want):
    global passed, failed
    if got == want:
        passed += 1
    else:
        failed += 1
        print(f'  FAIL {name}\n       got  {got!r}\n       want {want!r}')


SC = {'plot_type': 'polcurve', 'data': {'axes': [{
    'texts': [{'text': 'OCV = 0.950 V'}], 'axhlines': [], 'axvlines': [],
    'lines': [{'label': 'V', 'x': [0, 1], 'y': [0.9, 0.6]}]}]}}


def record(job, name):
    return {'schema': 2, 'job_id': job, 'sample_name': name,
            'script': 'FC Polarization Curve', 'timestamp': '2026-08-01T00:00:00Z',
            'input_files': [f'{job}.fcd'],
            'metrics': {'polcurve': {f'polcurve_{name}_b4': {
                'conditions': {'step': 'b4'}, 'values': {}}}},
            'summary': [{'Analysis': 'polcurve', 'Label': 'b4', 'OCV': 0.9}],
            'sidecars': encode_sidecars({f'polcurve_{name}_b4': SC})}


def entry(job, name, bid):
    return {'job_id': job, 'sample_name': name, 'script': 'FC Polarization Curve',
            'timestamp': '2026-08-01T00:00:00Z', 'bin_id': bid,
            'Data': [{'Analysis': 'polcurve', 'step': 'b4'}]}


class Store:
    """JSONBin in memory. `on[(method, bin)]` runs once, just before that call."""

    def __init__(self):
        self.bins = {'INDEX': {'schema': 2, 'runs': [
            entry('j1', 'SAMPLE-A', 'B1'), entry('j2', 'SAMPLE-B', 'B2')]},
            'B1': record('j1', 'SAMPLE-A'), 'B2': record('j2', 'SAMPLE-B')}
        self.on, self.n = {}, 0
        self.lock = threading.Lock()

    def __call__(self, url, method='GET', body=None, extra_headers=None):
        parts = url.rstrip('/').split('/')
        bid = parts[-2] if parts[-1] == 'latest' else parts[-1]
        hook = self.on.pop((method, bid), None)
        if hook:
            hook()
        with self.lock:
            if method == 'POST':
                self.n += 1
                bid = f'NEW{self.n}'
                self.bins[bid] = copy.deepcopy(body)
                return {'metadata': {'id': bid}}
            if method == 'PUT':
                self.bins[bid] = json.loads(json.dumps(jsonbin.json_safe(body)))
                return {}
            return {'record': copy.deepcopy(self.bins[bid])}

    def entry(self, name):
        return next((e for e in self.bins['INDEX']['runs'] if e['sample_name'] == name),
                    None)


def install():
    s = Store()
    jsonbin._request = s
    return s


def push(sample, job):
    out = Path(tempfile.mkdtemp()) / 'output'
    (out / '_plot_data').mkdir(parents=True)
    (out / '_plot_data' / f'polcurve_{sample}_b9.json').write_text(json.dumps(SC))
    return jsonbin.push_job_metrics(job_id=job, sample_name=sample,
                                    script='FC Polarization Curve', output_dir=out,
                                    input_files=['x.fcd'], script_short='PolCurve')


# ════════════════════════════════════════════════════════════════════
print('a push keeps an admin change written while it was merging')
store = install()
# Sample A's push has read the index and is writing A's bin when the stand of
# sample B is changed. Writing back the index it read would undo that.
store.on[('PUT', 'B1')] = lambda: admin.set_stand(['B2'], 'Scribner 1', apply=True)
r = push('SAMPLE-A', 'j3')
check('the push went through', r['pushed'], True)
check('B keeps the stand set meanwhile', store.entry('SAMPLE-B').get('stand'), 'Scribner 1')
check('and A has its merged entry', store.entry('SAMPLE-A').get('n_jobs'), 2)
check('no entry duplicated', len(store.bins['INDEX']['runs']), 2)

print('a push that is new keeps it too')
store = install()
store.on[('POST', 'b')] = lambda: admin.delete_samples(['B2'], apply=True)
r = push('SAMPLE-C', 'j4')
check('C added', store.entry('SAMPLE-C') is not None, True)
check('B unlisted meanwhile stays unlisted', store.entry('SAMPLE-B'), None)

print('an admin write keeps a push that landed while it was working')
store = install()
# Renaming B: between reading the index and writing it, sample C is pushed.
store.on[('PUT', 'B2')] = lambda: push('SAMPLE-C', 'j5')
admin.rename_sample('B2', 'SAMPLE-B2', apply=True)
check('the rename is written', store.entry('SAMPLE-B2') is not None, True)
check('and C, pushed meanwhile, is still indexed', store.entry('SAMPLE-C') is not None, True)
check('three entries', len(store.bins['INDEX']['runs']), 3)

store = install()
# Unlisting B: while its backup is being read, sample C is pushed.
store.on[('GET', 'B2')] = lambda: push('SAMPLE-C', 'j6')
admin.delete_samples(['B2'], apply=True)
check('an unlist keeps a push that landed meanwhile',
      sorted(e['sample_name'] for e in store.bins['INDEX']['runs']),
      ['SAMPLE-A', 'SAMPLE-C'])

store = install()
store.on[('GET', 'B2')] = lambda: push('SAMPLE-C', 'j7')
admin.set_stand(['B2'], 'FCTS 1', apply=True)
check('so does a stand change', (store.entry('SAMPLE-B').get('stand'),
                                 store.entry('SAMPLE-C') is not None), ('FCTS 1', True))

print('two admin writes keep each other\'s changes')
store = install()
store.on[('PUT', 'B1')] = lambda: admin.set_stand(['B2'], 'FCTS 2', apply=True)
admin.set_stand(['B1'], 'Scribner 2', apply=True)
check('both stands', (store.entry('SAMPLE-A').get('stand'),
                      store.entry('SAMPLE-B').get('stand')), ('Scribner 2', 'FCTS 2'))

store = install()
store.on[('PUT', 'B1')] = lambda: admin.delete_samples(['B2'], apply=True)
admin.rename_sample('B1', 'SAMPLE-A2', apply=True)
check('a rename and an unlist both land',
      sorted(e['sample_name'] for e in store.bins['INDEX']['runs']), ['SAMPLE-A2'])

# ════════════════════════════════════════════════════════════════════
print('an admin write waits for a push holding its sample')


def blocked(fn, hold='SAMPLE-A'):
    """Run fn in a thread while `hold`'s push lock is held; report whether it
    finished before the lock was released, and after."""
    done = threading.Event()
    lk = jsonbin._sample_lock(hold)
    lk.acquire()
    try:
        t = threading.Thread(target=lambda: (fn(), done.set()), daemon=True)
        t.start()
        early = done.wait(1.5)        # long enough for an unblocked call, under load
    finally:
        lk.release()
    late = done.wait(5)
    return early, late


store = install()
check('a stand change waits, then completes',
      blocked(lambda: admin.set_stand(['B1'], 'Scribner 1', apply=True)), (False, True))
check('and is written', store.entry('SAMPLE-A').get('stand'), 'Scribner 1')
store = install()
check('a preview does not wait', blocked(lambda: admin.set_stand(['B1'], 'Scribner 1')),
      (True, True))
store = install()
check('a rename waits for the name it is taking, too',
      blocked(lambda: admin.rename_sample('B2', 'SAMPLE-NEW', apply=True),
              hold='SAMPLE-NEW'), (False, True))
store = install()
check('an unrelated sample does not hold it up',
      blocked(lambda: admin.set_stand(['B2'], 'Scribner 1', apply=True)), (True, True))

print('the rebuild holds every sample it rewrites')
store = install()
# A unit the old rules left behind, with no plots under it: A needs a rebuild.
store.bins['INDEX']['runs'][0]['Data'].append({'Analysis': 'elx_model_fit', 'step': ''})
prep = admin.rebuild_electrolyzer(prepare=True)
check('A is to be rebuilt', prep['keys'], ['B1'])
check('its apply waits for a push on it',
      blocked(lambda: admin.rebuild_electrolyzer(apply=True, keys=prep['keys'],
                                                 token=prep['token'])), (False, True))
check('and then rebuilds it', store.entry('SAMPLE-A')['Data'],
      [{'Analysis': 'polcurve', 'step': 'b4'}])
check('its preview and its prepare do not wait',
      (blocked(lambda: admin.rebuild_electrolyzer()),
       blocked(lambda: admin.rebuild_electrolyzer(prepare=True))),
      ((True, True), (True, True)))

print('a push re-reads the index and finds its own entry, not a namesake')
# Only something outside this process — a CLI tool run over railway ssh —
# can add an entry under a pushing sample's name; the push must neither
# overwrite it nor be written over it.
store = install()
stray = dict(entry('jX', 'SAMPLE-C', 'BX'), run_date='2026-12-31')
store.on[('POST', 'b')] = lambda: store.bins['INDEX']['runs'].append(copy.deepcopy(stray))
push('SAMPLE-C', 'j8')
c = [e for e in store.bins['INDEX']['runs'] if e['sample_name'] == 'SAMPLE-C']
check('a new sample\'s entry is added beside it, not over it',
      sorted(e['bin_id'] for e in c), ['BX', 'NEW1'])

store = install()
store.on[('PUT', 'B1')] = lambda: store.bins['INDEX']['runs'].append(
    dict(entry('jY', 'SAMPLE-A', 'BY'), run_date='2026-12-31'))
push('SAMPLE-A', 'j9')
a = {e['bin_id']: e for e in store.bins['INDEX']['runs'] if e['sample_name'] == 'SAMPLE-A'}
check('a merge replaces the entry it merged into', a['B1'].get('n_jobs'), 2)
check('and leaves the newer namesake alone', a['BY'].get('job_id'), 'jY')


def meddle():
    """Change A's entry, and put another sample's entry on A's bin ahead of it."""
    runs = store.bins['INDEX']['runs']
    runs[0]['note'] = 'touched'
    runs.insert(0, entry('jZ', 'SAMPLE-Z', 'B1'))


store = install()
store.on[('PUT', 'B1')] = meddle
push('SAMPLE-A', 'j10')
check('an entry changed meanwhile is found by its sample and bin',
      store.entry('SAMPLE-A').get('n_jobs'), 2)
check('not by its bin alone', store.entry('SAMPLE-Z').get('job_id'), 'jZ')

print('a name changed while the locks were being taken is locked as well')
store = install()
taken = []
real_lock = jsonbin._sample_lock
jsonbin._sample_lock = lambda n: (taken.append(n or '__unnamed__'), real_lock(n))[1]
try:
    # Renamed after the first read of the index and before the second, which
    # is the one made once the first name's lock is held.
    def rename_b():
        store.bins['INDEX']['runs'][1]['sample_name'] = 'SAMPLE-B-RENAMED'
    store.on[('GET', 'INDEX')] = lambda: store.on.__setitem__(('GET', 'INDEX'), rename_b)
    admin.set_stand(['B2'], 'Scribner 1', apply=True)
finally:
    jsonbin._sample_lock = real_lock
check('both names locked', {'SAMPLE-B', 'SAMPLE-B-RENAMED'} <= set(taken), True)
check('the change landed', store.entry('SAMPLE-B-RENAMED').get('stand'), 'Scribner 1')

# ════════════════════════════════════════════════════════════════════
print('main.py: admin handlers run off the event loop, and writes empty the caches')
import io, contextlib                                               # noqa: E401,E402
with contextlib.redirect_stdout(io.StringIO()):
    from app import main                                            # noqa: E402
from fastapi import HTTPException                                   # noqa: E402

routes = {r.path: r.endpoint for r in main.app.routes
          if getattr(r, 'path', '').startswith('/api/admin/')}
check('every admin handler is a plain function',
      sorted(p for p, fn in routes.items() if inspect.iscoroutinefunction(fn)), [])
check('the rebuild is one of them', '/api/admin/rebuild-electrolyzer' in routes, True)

store = install()
viewstore.fetch_index()
check('the index is cached', viewstore.cache_stats()['index_age_s'] is not None, True)
prev = main.admin_rebuild_electrolyzer({'code': 'letmein'})
check('a preview answers', (prev['applied'], prev['scanned']), (False, 0))
check('and leaves the cache', viewstore.cache_stats()['index_age_s'] is not None, True)
main.admin_backup({'code': 'letmein'})
check('so does a backup, which changes nothing',
      viewstore.cache_stats()['index_age_s'] is not None, True)
main.admin_stand({'code': 'letmein', 'keys': ['B1'], 'stand': 'Scribner 1', 'apply': True})
check('an applied change empties it', viewstore.cache_stats()['index_age_s'], None)
try:
    main.admin_rebuild_electrolyzer({'code': 'wrong', 'apply': True})
    check('a wrong code is refused', 'not refused', 403)
except HTTPException as e:
    check('a wrong code is refused', e.status_code, 403)

print('main.py: pushes run on their own thread, one at a time, in order')
order, gate = [], threading.Event()
real_push = jsonbin.push_job_metrics


def slow_push(**kw):
    gate.wait(5)
    order.append(kw['job_id'])
    return {'pushed': True, 'reason': None, 'bin_id': 'X'}


class Done:
    def result(self):
        return {'output_files': ['out.png'], 'output_groups': {},
                'script_result': {}, 'sample_name': 'S'}


jsonbin.push_job_metrics = slow_push
try:
    for jid in ('jA', 'jB'):
        main.jobs[jid] = {'job_id': jid, 'status': 'running', 'script': 'FC Polarization Curve',
                          'input_files': [], 'params': {}}
    t0 = time.time()
    main._on_job_done('jA', Done())
    main._on_job_done('jB', Done())
    check('completion does not wait for the push', time.time() - t0 < 1, True)
    check('the jobs are complete already',
          [main.jobs[j]['status'] for j in ('jA', 'jB')], ['complete', 'complete'])
    check('with their pushes still to come, which the page polls for',
          [main.jobs[j].get('push_pending') for j in ('jA', 'jB')], [True, True])
    gate.set()
    main.push_executor.submit(lambda: None).result(5)
    check('the pushes ran, in the order the jobs finished', order, ['jA', 'jB'])
    check('and are recorded on the jobs',
          [(main.jobs[j].get('metrics_push', {}).get('pushed'), main.jobs[j].get('push_pending'))
           for j in ('jA', 'jB')], [(True, False), (True, False)])

    # Shutting down, the push thread is stopped before the process pool has
    # finished its jobs. A job finishing then is pushed where it finishes.
    live = main.push_executor
    main.push_executor = ThreadPoolExecutor(max_workers=1)
    main.push_executor.shutdown()
    main.jobs['jC'] = {'job_id': 'jC', 'status': 'running', 'script': 'FC Polarization Curve',
                       'input_files': [], 'params': {}}
    try:
        main._on_job_done('jC', Done())
    finally:
        main.push_executor = live
    check('a job finishing during shutdown is still pushed',
          (order[-1], main.jobs['jC'].get('push_pending')), ('jC', False))
finally:
    jsonbin.push_job_metrics = real_push
    gate.set()

print(f'\n{passed} passed, {failed} failed')
sys.exit(1 if failed else 0)
