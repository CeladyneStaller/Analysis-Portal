"""
Analysis Portal (Lite) — FastAPI application.

Self-contained: no database, no Redis, no cloud storage.
Jobs run in a ProcessPoolExecutor and results live in /tmp until auto-cleaned.

Routes:
  GET  /                              → Frontend UI
  GET  /api/scripts                   → Available analysis scripts
  POST /api/upload                    → Upload CSVs + start analysis
  GET  /api/jobs/{id}                 → Job status
  GET  /api/jobs                      → All jobs (for page reload recovery)
  GET  /api/download/{id}/{filename}  → Download a result file
"""

import os
import uuid
import shutil
import time
import traceback
import threading
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

# ═══════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════

MAX_WORKERS = int(os.getenv("MAX_WORKERS", "3"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
JOB_TTL_HOURS = int(os.getenv("JOB_TTL_HOURS", "24"))
JOBS_DIR = Path(os.getenv("JOBS_DIR", "/tmp/analysis-portal-jobs"))
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# In-memory job store
# ═══════════════════════════════════════════════════════════════════

jobs: dict = {}  # job_id → job metadata dict
jobs_lock = threading.Lock()

app = FastAPI(title="Analysis Portal")
executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)

TEMPLATE_DIR = Path(__file__).parent / "templates"


# ═══════════════════════════════════════════════════════════════════
# Worker function (runs in a separate process)
# ═══════════════════════════════════════════════════════════════════

def _run_job(job_id: str, script_name: str, input_dir: str, output_dir: str,
             params: dict = None) -> dict:
    """
    Executed in the process pool. Imports the script, runs it,
    returns a result dict. Exceptions propagate back to the future.
    """
    import importlib
    import scripts

    registry = scripts.SCRIPT_REGISTRY

    if script_name not in registry:
        raise ValueError(f"Unknown script: {script_name}")

    run_fn = registry[script_name]
    result = run_fn(input_dir=input_dir, output_dir=output_dir, params=params or {})

    # Find all output files recursively and group by category
    out = Path(output_dir)
    all_output = [f for f in out.rglob("*") if f.is_file()]

    # Map subdirectory names to friendly labels
    DIR_LABELS = {
        'ecsa': 'ECSA',
        'eis': 'EIS',
        'crossover': 'H₂ Crossover',
        'polcurve': 'Polarization Curve',
    }

    grouped = {}
    flat_list = []
    for f in sorted(all_output):
        rel = f.relative_to(out)
        flat_list.append(str(rel))
        # Group by first directory component, or script name for root files
        parts = rel.parts
        if len(parts) > 1:
            dir_key = parts[0]
            label = DIR_LABELS.get(dir_key, dir_key.replace('_', ' ').title())
        else:
            label = script_name
        grouped.setdefault(label, []).append(str(rel))

    return {
        "output_files": flat_list,
        "output_groups": grouped,
        "script_result": result or {},
    }


def _on_job_done(job_id: str, future):
    """Callback when a job finishes (success or failure)."""
    with jobs_lock:
        if job_id not in jobs:
            return
        try:
            result = future.result()
            jobs[job_id].update({
                "status": "complete",
                "message": "Analysis complete",
                "output_files": result["output_files"],
                "output_groups": result.get("output_groups", {}),
                "script_result": result["script_result"],
                "completed_at": datetime.now().isoformat(),
            })
        except Exception as e:
            jobs[job_id].update({
                "status": "failed",
                "message": str(e),
                "error": traceback.format_exc(),
                "completed_at": datetime.now().isoformat(),
            })


# ═══════════════════════════════════════════════════════════════════
# Cleanup: remove expired jobs
# ═══════════════════════════════════════════════════════════════════

def _cleanup_old_jobs():
    """Remove jobs and their files after JOB_TTL_HOURS."""
    ttl_seconds = JOB_TTL_HOURS * 3600
    now = time.time()

    with jobs_lock:
        expired = []
        for jid, meta in jobs.items():
            job_dir = JOBS_DIR / jid
            if job_dir.exists():
                age = now - job_dir.stat().st_mtime
                if age > ttl_seconds and meta["status"] in ("complete", "failed"):
                    expired.append(jid)

        for jid in expired:
            shutil.rmtree(JOBS_DIR / jid, ignore_errors=True)
            del jobs[jid]


def _start_cleanup_timer():
    _cleanup_old_jobs()
    timer = threading.Timer(3600, _start_cleanup_timer)  # Run every hour
    timer.daemon = True
    timer.start()


@app.on_event("startup")
def startup():
    _start_cleanup_timer()


# ═══════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse((TEMPLATE_DIR / "index.html").read_text())


@app.get("/api/scripts")
async def list_scripts():
    from scripts import SCRIPT_REGISTRY, SCRIPT_PARAMS
    return {
        "scripts": [
            {
                "name": name,
                "description": (fn.__doc__ or "").strip().split("\n")[0],
                "params": SCRIPT_PARAMS.get(name, []),
            }
            for name, fn in SCRIPT_REGISTRY.items()
        ]
    }


@app.post("/api/upload")
async def upload_and_run(
    script: str = Form(...),
    params: str = Form("{}"),
    files: list[UploadFile] = File(...),
):
    import json
    from scripts import SCRIPT_REGISTRY
    if script not in SCRIPT_REGISTRY:
        raise HTTPException(400, f"Unknown script: {script}")
    if not files:
        raise HTTPException(400, "No files uploaded")

    # Parse params JSON
    try:
        user_params = json.loads(params)
    except json.JSONDecodeError:
        user_params = {}

    # Validate: at least one recognized data file
    allowed_ext = ('.csv', '.txt', '.tsv', '.fcd')
    data_files = [f for f in files if f.filename.lower().endswith(allowed_ext)
                  or '/' in f.filename]  # folder uploads pass through
    if not data_files and not any('/' in f.filename for f in files):
        has_any_data = any(f.filename.lower().endswith(allowed_ext) for f in files)
        if not has_any_data:
            raise HTTPException(400, "No recognized data files (CSV, TXT, TSV, FCD)")

    # Create job directories
    job_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    input_dir = JOBS_DIR / job_id / "input"
    output_dir = JOBS_DIR / job_id / "output"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    # Save uploaded files, preserving folder structure from paths
    filenames = []
    for f in files:
        content = await f.read()
        if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
            shutil.rmtree(JOBS_DIR / job_id)
            raise HTTPException(413, f"{f.filename} exceeds {MAX_UPLOAD_MB}MB limit")

        # filename may contain path separators (e.g. "folder/sub/file.csv")
        safe_path = Path(f.filename)
        # Security: prevent path traversal
        if '..' in safe_path.parts:
            continue
        dest = input_dir / safe_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        filenames.append(f.filename)

    # Register job
    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "message": f"Running {script}...",
            "script": script,
            "input_files": filenames,
            "submitted_at": datetime.now().isoformat(),
        }

    # Submit to process pool
    future = executor.submit(
        _run_job, job_id, script, str(input_dir), str(output_dir), user_params
    )
    future.add_done_callback(lambda f: _on_job_done(job_id, f))

    return {"job_id": job_id, "status": "running", "files_received": filenames}


@app.get("/api/jobs")
async def all_jobs():
    with jobs_lock:
        return {"jobs": list(jobs.values())}


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(404, "Job not found")
        return jobs[job_id]


@app.get("/api/download/{job_id}/{filepath:path}")
async def download_result(job_id: str, filepath: str):
    # Resolve and validate path to prevent traversal
    base = JOBS_DIR / job_id / "output"
    file_path = (base / filepath).resolve()

    # Ensure the resolved path is still inside the output directory
    if not str(file_path).startswith(str(base.resolve())):
        raise HTTPException(400, "Invalid path")

    if not file_path.exists():
        raise HTTPException(404, "File not found")

    media_types = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "csv": "text/csv",
        "pdf": "application/pdf",
    }
    ext = file_path.suffix.lstrip(".").lower()
    media_type = media_types.get(ext, "application/octet-stream")

    return FileResponse(
        file_path,
        media_type=media_type,
        filename=file_path.name,
    )
