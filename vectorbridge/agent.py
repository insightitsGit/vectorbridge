"""
Docker agent entrypoint.
Reads LICENSE_KEY + optional JOB_ID from environment,
fetches config from dashboard, runs migration, reports back.
"""

import os
import sys
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vectorbridge.agent")


def main():
    license_key = os.getenv("VB_LICENSE_KEY") or os.getenv("LICENSE_KEY")
    job_id      = os.getenv("VB_JOB_ID")
    poll_sec    = int(os.getenv("VB_POLL_SECONDS", "0"))   # 0 = run once

    if not license_key:
        log.error("VB_LICENSE_KEY environment variable is required")
        sys.exit(1)

    log.info("VectorBridge Agent starting...")
    log.info(f"License key: {license_key[:8]}****")
    log.info(f"Job ID: {job_id or 'fetch from dashboard'}")

    from .bridge import Bridge
    from .license import report_usage

    while True:
        try:
            bridge = Bridge.from_license(license_key, job_id)
            log.info(f"Running job {bridge.job_id} — {bridge.source_type} → {bridge.target_type}")
            report = bridge.run(verbose=True)
            log.info(f"Job complete — {report.transferred:,} vectors, "
                     f"{report.verification_rate:.2f}% verified")
        except PermissionError as e:
            log.error(f"License error: {e}")
            sys.exit(1)
        except Exception as e:
            log.error(f"Job failed: {e}", exc_info=True)
            if poll_sec == 0:
                sys.exit(1)

        if poll_sec == 0:
            break

        log.info(f"Next run in {poll_sec}s...")
        time.sleep(poll_sec)


if __name__ == "__main__":
    main()
