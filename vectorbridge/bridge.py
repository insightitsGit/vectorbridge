"""
Main public API — Bridge is the single entry point for all migrations.
"""

import json
import uuid
import logging
from pathlib import Path

from .connectors import get_connector, ConnectorConfig
from .orchestrator import MigrationJob
from .integrity import IntegrityReport
from .license import validate_license, fetch_job_config, report_usage

log = logging.getLogger("vectorbridge")


class Bridge:
    """
    Universal vector database migration engine.

    Usage — from config file:
        bridge = Bridge.from_config("vb_config.json")
        report = bridge.run()

    Usage — from license key (Docker agent mode):
        bridge = Bridge.from_license("vb_live_xxxx")
        report = bridge.run()

    Usage — programmatic:
        bridge = Bridge(
            source_type="pgvector", source_config={...},
            target_type="pinecone", target_config={...},
        )
        report = bridge.run()
    """

    def __init__(
        self,
        source_type: str,
        source_config: dict,
        target_type: str,
        target_config: dict,
        mode: str = "full",
        batch_size: int = 256,
        job_id: str = None,
        license_key: str = None,
        resume: bool = True,
        metric_override: bool = False,
        semantic_verify: bool = True,
        semantic_probes: int = 100,
        semantic_top_k: int = 5,
    ):
        self.source_type = source_type
        self.source_config = source_config
        self.target_type = target_type
        self.target_config = target_config
        self.mode = mode
        self.batch_size = batch_size
        self.job_id = job_id or str(uuid.uuid4())[:8]
        self.license_key = license_key
        self.resume = resume
        self.metric_override  = metric_override
        self.semantic_verify  = semantic_verify
        self.semantic_probes  = semantic_probes
        self.semantic_top_k   = semantic_top_k

    @classmethod
    def from_config(cls, path: str) -> "Bridge":
        """Load from a JSON config file downloaded from the dashboard."""
        config = json.loads(Path(path).read_text())
        return cls(
            source_type=config["source"]["type"],
            source_config={k: v for k, v in config["source"].items() if k != "type"},
            target_type=config["target"]["type"],
            target_config={k: v for k, v in config["target"].items() if k != "type"},
            mode=config.get("mode", "full"),
            batch_size=config.get("batch_size", 256),
            job_id=config.get("job_id", str(uuid.uuid4())[:8]),
            license_key=config.get("license_key"),
            resume=config.get("resume", True),
            metric_override=config.get("metric_override", False),
            semantic_verify=config.get("semantic_verify", True),
            semantic_probes=config.get("semantic_probes", 100),
            semantic_top_k=config.get("semantic_top_k", 5),
        )

    @classmethod
    def from_license(cls, license_key: str, job_id: str = None) -> "Bridge":
        """
        Docker agent mode — fetch config from insightits.co/vectorbridge dashboard.
        The agent only needs a license key to start.
        """
        log.info("Fetching job config from insightits.co/vectorbridge dashboard...")
        info = validate_license(license_key)
        if not info.valid:
            raise PermissionError(f"Invalid license key: {license_key}")
        log.info(f"License valid — org={info.org} plan={info.plan} "
                 f"DWV used={info.dwv_used:,}/{info.dwv_limit:,}")
        config = fetch_job_config(license_key, job_id)
        return cls(
            source_type=config["source"]["type"],
            source_config={k: v for k, v in config["source"].items() if k != "type"},
            target_type=config["target"]["type"],
            target_config={k: v for k, v in config["target"].items() if k != "type"},
            mode=config.get("mode", "full"),
            batch_size=config.get("batch_size", 256),
            job_id=config.get("job_id", str(uuid.uuid4())[:8]),
            license_key=license_key,
            resume=config.get("resume", True),
        )

    def run(self, verbose: bool = True) -> IntegrityReport:
        if verbose:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s  %(levelname)s  %(message)s",
                datefmt="%H:%M:%S",
            )

        # Validate license if provided
        if self.license_key:
            info = validate_license(self.license_key)
            if not info.valid:
                raise PermissionError("Invalid or expired license key.")

        src_cfg = ConnectorConfig(**{
            k: v for k, v in self.source_config.items()
            if k in ConnectorConfig.__dataclass_fields__
        })
        tgt_cfg = ConnectorConfig(**{
            k: v for k, v in self.target_config.items()
            if k in ConnectorConfig.__dataclass_fields__
        })

        source = get_connector(self.source_type, src_cfg)
        target = get_connector(self.target_type, tgt_cfg)

        job = MigrationJob(
            job_id=self.job_id,
            source=source,
            target=target,
            mode=self.mode,
            batch_size=self.batch_size,
            resume=self.resume,
            metric_override=self.metric_override,
            semantic_verify=self.semantic_verify,
            semantic_probes=self.semantic_probes,
            semantic_top_k=self.semantic_top_k,
        )

        log.info(f"Starting VectorBridge job {self.job_id}")
        log.info(f"  {self.source_type} → {self.target_type}  mode={self.mode}")

        report = job.run()

        if verbose:
            print(report.summary())

        # Report usage back to dashboard
        if self.license_key:
            report_usage(self.license_key, self.job_id, report.to_dict())

        return report
