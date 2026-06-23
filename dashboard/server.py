"""
VectorBridge Dashboard — FastAPI backend
Serves the web UI and handles agent reporting, license validation, usage tracking.
"""

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Optional
import uuid
import json
import time
from pathlib import Path
from collections import defaultdict

app = FastAPI(title="VectorBridge Dashboard", version="0.1.0")

# ── In-memory store (replace with PostgreSQL in production) ──────────────────
LICENSES: dict[str, dict] = {}
JOBS: dict[str, dict] = {}
REPORTS: list[dict] = []
USAGE: dict[str, int] = defaultdict(int)  # license_key → total DWVs used

PLANS = {
    "free":       {"dwv_limit": 200_000_000,    "price": 0},
    "starter":    {"dwv_limit": 2_000_000_000,  "price": 49},
    "pro":        {"dwv_limit": 25_000_000_000, "price": 199},
    "enterprise": {"dwv_limit": -1,             "price": 999},
}


# ── Models ───────────────────────────────────────────────────────────────────

class LicenseValidateRequest(BaseModel):
    license_key: str

class ReportRequest(BaseModel):
    license_key: str
    job_id: str
    transferred: int = 0
    verified: int = 0
    failed_watermark: int = 0
    wire_bytes: int = 0
    raw_bytes: int = 0
    verification_rate_pct: float = 0
    bandwidth_savings_x: float = 0
    source: str = ""
    target: str = ""
    mode: str = ""
    completed_at: str = ""

class CreateLicenseRequest(BaseModel):
    org: str
    plan: str = "free"
    email: str

class CreateJobRequest(BaseModel):
    license_key: str
    source: dict
    target: dict
    mode: str = "full"
    batch_size: int = 256


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/v1/health")
def health():
    return {"status": "ok", "service": "vectorbridge-api"}


# ── License endpoints ────────────────────────────────────────────────────────

@app.post("/v1/license/validate")
def validate_license(req: LicenseValidateRequest):
    lic = LICENSES.get(req.license_key)
    if not lic:
        raise HTTPException(status_code=403, detail="Invalid license key")
    dwv_used = USAGE[req.license_key]
    dwv_limit = PLANS[lic["plan"]]["dwv_limit"]
    return {
        "valid": True,
        "org": lic["org"],
        "plan": lic["plan"],
        "dwv_limit": dwv_limit,
        "dwv_used": dwv_used,
    }

@app.post("/v1/license/create")
def create_license(req: CreateLicenseRequest):
    key = "vb_live_" + uuid.uuid4().hex[:24]
    LICENSES[key] = {"org": req.org, "plan": req.plan, "email": req.email,
                     "created_at": time.time()}
    return {"license_key": key, "plan": req.plan, "org": req.org}


# ── Agent endpoints ──────────────────────────────────────────────────────────

@app.get("/v1/agent/config")
def agent_config(x_license: str = Header(...), job_id: Optional[str] = None):
    lic = LICENSES.get(x_license)
    if not lic:
        raise HTTPException(status_code=403, detail="Invalid license key")
    if job_id:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        return job
    # Return most recent pending job for this license
    pending = [j for j in JOBS.values()
               if j.get("license_key") == x_license and not j.get("completed")]
    if not pending:
        raise HTTPException(status_code=404, detail="No pending jobs for this license")
    return pending[-1]

@app.post("/v1/agent/report")
def agent_report(req: ReportRequest):
    lic = LICENSES.get(req.license_key)
    if not lic:
        raise HTTPException(status_code=403, detail="Invalid license key")

    # Calculate DWVs from this run (need dimension from job config)
    job = JOBS.get(req.job_id, {})
    dim = job.get("source", {}).get("dimension", 1536)   # default 1536
    dwvs = req.transferred * dim
    USAGE[req.license_key] += dwvs

    report = req.dict()
    report["dwv_consumed"] = dwvs
    report["org"] = lic["org"]
    report["received_at"] = time.time()
    REPORTS.append(report)

    # Mark job complete
    if req.job_id in JOBS:
        JOBS[req.job_id]["completed"] = True
        JOBS[req.job_id]["last_report"] = report

    return {"received": True, "dwv_consumed": dwvs, "total_dwv_used": USAGE[req.license_key]}


# ── Dashboard API ─────────────────────────────────────────────────────────────

@app.post("/v1/jobs")
def create_job(req: CreateJobRequest):
    job_id = "job_" + uuid.uuid4().hex[:8]
    job = {
        "job_id": job_id,
        "license_key": req.license_key,
        "source": req.source,
        "target": req.target,
        "mode": req.mode,
        "batch_size": req.batch_size,
        "resume": True,
        "completed": False,
        "created_at": time.time(),
    }
    JOBS[job_id] = job
    return job

@app.get("/v1/jobs")
def list_jobs(x_license: str = Header(...)):
    return [j for j in JOBS.values() if j.get("license_key") == x_license]

@app.get("/v1/reports")
def list_reports(x_license: str = Header(...)):
    lic = LICENSES.get(x_license)
    if not lic:
        raise HTTPException(status_code=403)
    return [r for r in REPORTS if r.get("org") == lic["org"]]

@app.get("/v1/usage")
def get_usage(x_license: str = Header(...)):
    lic = LICENSES.get(x_license)
    if not lic:
        raise HTTPException(status_code=403)
    dwv_used = USAGE[x_license]
    dwv_limit = PLANS[lic["plan"]]["dwv_limit"]
    pct = (dwv_used / dwv_limit * 100) if dwv_limit > 0 else 0
    return {
        "plan": lic["plan"],
        "dwv_used": dwv_used,
        "dwv_limit": dwv_limit,
        "usage_pct": round(pct, 2),
        "reports_count": len([r for r in REPORTS if r.get("org") == lic["org"]]),
        "jobs_count": len([j for j in JOBS.values() if j.get("license_key") == x_license]),
    }


# ── Serve dashboard UI ────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return FileResponse(Path(__file__).parent / "static" / "index.html")

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)
