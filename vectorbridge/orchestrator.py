"""
Core migration engine — source → CHORUS transport → target.
"""

import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Callable

from .connectors.base import VectorConnector
from .transport import DirectTransport
from .checkpoint import Checkpoint, checkpoint_path
from .integrity import IntegrityReport
from .verify import validate_metrics, SemanticValidator, SemanticVerifyReport

log = logging.getLogger("vectorbridge")


class MigrationJob:
    def __init__(
        self,
        job_id: str,
        source: VectorConnector,
        target: VectorConnector,
        mode: str = "full",          # full | incremental | live
        batch_size: int = 256,
        on_progress: Callable = None,
        on_batch: Callable = None,
        resume: bool = True,
        metric_override: bool = False,   # allow mismatched distance metrics
        semantic_verify: bool = True,    # run post-migration semantic probes
        semantic_probes: int = 100,      # how many probe vectors to use
        semantic_top_k: int = 5,         # neighbours compared per probe
    ):
        self.job_id = job_id
        self.source = source
        self.target = target
        self.mode = mode
        self.batch_size = batch_size
        self.on_progress = on_progress
        self.on_batch = on_batch
        self.resume = resume
        self.metric_override   = metric_override
        self.semantic_verify   = semantic_verify
        self.semantic_probes   = semantic_probes
        self.semantic_top_k    = semantic_top_k

    def _preflight(self) -> None:
        """
        Run pre-migration checks before touching any data.

        1. Distance metric alignment  — block if source != target unless
           metric_override=True, because a metric mismatch lets vectors
           import cleanly while queries silently return wrong results.
        """
        src_metric = getattr(self.source.config, "distance_metric", "cosine")
        tgt_metric = getattr(self.target.config, "distance_metric", "cosine")
        validate_metrics(src_metric, tgt_metric, override=self.metric_override)

        src_dim = getattr(self.source.config, "dimension", 0)
        tgt_dim = getattr(self.target.config, "dimension", 0)
        if src_dim and tgt_dim and src_dim != tgt_dim:
            raise ValueError(
                f"Dimension mismatch: source={src_dim}, target={tgt_dim}. "
                "Both connectors must be configured for the same embedding dimension."
            )

    def run(self) -> IntegrityReport:
        # Pre-flight: catch silent data-corruption issues before migration starts
        self._preflight()

        cp_path = checkpoint_path(self.job_id)
        source_name = type(self.source).__name__.replace("Connector", "").lower()
        target_name = type(self.target).__name__.replace("Connector", "").lower()

        # Resume or start fresh
        if self.resume:
            try:
                cp = Checkpoint.load(cp_path)
                log.info(f"Resuming job {self.job_id} from offset {cp.last_offset}")
            except FileNotFoundError:
                cp = Checkpoint.new(self.job_id, source_name, target_name, self.mode)
        else:
            cp = Checkpoint.new(self.job_id, source_name, target_name, self.mode)

        report = IntegrityReport(
            job_id=self.job_id,
            source=source_name,
            target=target_name,
            mode=self.mode,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        with self.source, self.target:
            total = self.source.count()
            report.total_vectors = total
            log.info(f"Source has {total:,} vectors")

            # Auto-detect dimension from first batch
            transport = None
            offset = cp.last_offset

            if self.mode == "full":
                batches = self.source.read_batches(self.batch_size, offset)
            elif self.mode == "incremental":
                if not self.source.supports_incremental():
                    raise ValueError(f"{source_name} does not support incremental sync")
                batches = self.source.read_changes(cp.last_timestamp, self.batch_size)
            else:
                raise ValueError(f"Mode '{self.mode}' not yet supported in this build")

            for batch in batches:
                if not batch:
                    continue

                # Init transport once we know dimension
                if transport is None:
                    dim = batch[0].vector.shape[0]
                    transport = DirectTransport(dim)
                    self.target.create_index(dim)
                    log.info(f"Dimension: {dim}  Transport session: {transport.session.session_seed.hex()[:16]}...")

                # CHORUS transport: encrypt → wire → decrypt → verify
                received, stats = transport.transfer(batch)
                report.record_batch(stats)

                if stats["failed"] > 0:
                    msg = f"Batch at offset {offset}: {stats['failed']} watermark failures"
                    log.warning(msg)
                    report.errors.append(msg)

                # Write to target
                written = self.target.write_batch(received)
                offset += written

                last_id = batch[-1].id
                cp.advance(written, offset, last_id)
                cp.save(cp_path)

                if self.on_progress:
                    self.on_progress(offset, total, stats)
                if self.on_batch:
                    self.on_batch(report)

                log.info(
                    f"  [{offset:,}/{total:,}] "
                    f"verified={stats['verified']}/{stats['sent']} "
                    f"wire={stats['wire_bytes']:,}b"
                )

        cp.mark_complete()
        cp.save(cp_path)
        report.finalize()

        # Post-migration semantic validation
        if self.semantic_verify:
            log.info(
                f"Running semantic validation "
                f"({self.semantic_probes} probes x top-{self.semantic_top_k}) ..."
            )
            validator = SemanticValidator(
                source   = self.source,
                target   = self.target,
                n_probes = self.semantic_probes,
                top_k    = self.semantic_top_k,
            )
            with self.source, self.target:
                sv_report = validator.run(job_id=self.job_id)

            report.semantic_verify = sv_report.to_dict()

            if not sv_report.passed:
                msg = (
                    f"Semantic validation FAILED: avg neighbour overlap "
                    f"{sv_report.avg_overlap*100:.1f}% (threshold 95%). "
                    "Check distance_metric and normalisation settings."
                )
                log.error(msg)
                report.errors.append(msg)
            else:
                log.info(
                    f"Semantic validation PASSED: "
                    f"{sv_report.avg_overlap*100:.2f}% avg neighbour overlap"
                )

        return report
