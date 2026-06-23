"""
License validation and remote config fetch from vectorbridge.io API.
"""

import os
import json
import logging
import urllib.request
import urllib.error
from dataclasses import dataclass

log = logging.getLogger("vectorbridge")

API_BASE = os.getenv("VECTORBRIDGE_API", "https://insightits.co/api/vectorbridge/v1")


@dataclass
class LicenseInfo:
    license_key: str
    plan: str
    org: str
    dwv_limit: int
    dwv_used: int
    valid: bool


def validate_license(license_key: str) -> LicenseInfo:
    """Call vectorbridge.io to validate a license key."""
    url = f"{API_BASE}/license/validate"
    payload = json.dumps({"license_key": license_key}).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json", "X-License": license_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return LicenseInfo(
                license_key=license_key,
                plan=data.get("plan", "unknown"),
                org=data.get("org", "unknown"),
                dwv_limit=data.get("dwv_limit", 0),
                dwv_used=data.get("dwv_used", 0),
                valid=data.get("valid", False),
            )
    except urllib.error.HTTPError as e:
        log.error(f"License validation failed: HTTP {e.code}")
        return LicenseInfo(license_key=license_key, plan="", org="",
                           dwv_limit=0, dwv_used=0, valid=False)
    except Exception as e:
        log.warning(f"License server unreachable: {e}. Running in offline mode.")
        # Offline grace — allow run but log warning
        return LicenseInfo(license_key=license_key, plan="offline", org="offline",
                           dwv_limit=0, dwv_used=0, valid=True)


def fetch_job_config(license_key: str, job_id: str = None) -> dict:
    """Fetch job configuration from the dashboard."""
    url = f"{API_BASE}/agent/config"
    params = {"job_id": job_id} if job_id else {}
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    full_url = f"{url}?{qs}" if qs else url
    req = urllib.request.Request(
        full_url,
        headers={"X-License": license_key},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        raise RuntimeError(f"Failed to fetch job config from dashboard: {e}")


def report_usage(license_key: str, job_id: str, report: dict) -> None:
    """Post DWV usage and integrity report back to dashboard."""
    url = f"{API_BASE}/agent/report"
    payload = json.dumps({"license_key": license_key, "job_id": job_id, **report}).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json", "X-License": license_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            log.info(f"Usage reported to dashboard: {resp.status}")
    except Exception as e:
        log.warning(f"Could not report usage to dashboard: {e}")
