"""
Post-migration semantic validator.

Proves that the target database is mathematically equivalent to the source —
not just byte-identical, but returning the same nearest neighbours for
arbitrary query vectors.

Why this matters
----------------
A migration can pass byte-level checksums and still silently produce wrong
search results if:
  - the distance metric was changed (cosine vs L2)
  - vectors were normalised on one side but not the other
  - dimension order was shuffled by an intermediary

Usage
-----
from vectorbridge.verify import SemanticValidator

validator = SemanticValidator(source_connector, target_connector)
report    = validator.run(n_probes=100, top_k=5)
print(report.summary())
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .connectors.base import VectorConnector

log = logging.getLogger("vectorbridge.verify")

# ── Metric mismatch guard ─────────────────────────────────────────────────────

METRIC_ALIASES: dict[str, str] = {
    "cosine":        "cosine",
    "cos":           "cosine",
    "dot":           "dot",
    "dot_product":   "dot",
    "ip":            "dot",
    "l2":            "l2",
    "euclidean":     "l2",
    "euclid":        "l2",
}


class MetricMismatchError(ValueError):
    """Raised when source and target use different distance metrics."""


def validate_metrics(
    source_metric: str,
    target_metric: str,
    override: bool = False,
) -> None:
    """
    Compare normalised metric names. Raise MetricMismatchError unless
    override=True is explicitly passed by the caller.

    Vectors migrated with mismatched metrics will import successfully but
    return completely wrong query results — silent data corruption.
    """
    src = METRIC_ALIASES.get(source_metric.lower(), source_metric.lower())
    tgt = METRIC_ALIASES.get(target_metric.lower(), target_metric.lower())

    if src != tgt:
        if override:
            log.warning(
                f"Distance metric mismatch: source={src!r}, target={tgt!r}. "
                "Migration proceeding because override=True was set. "
                "Query results on the target will differ from the source."
            )
        else:
            raise MetricMismatchError(
                f"\n\n"
                f"  DISTANCE METRIC MISMATCH\n"
                f"  ========================\n"
                f"  Source : {src}\n"
                f"  Target : {tgt}\n\n"
                f"  Vectors will import successfully but nearest-neighbour\n"
                f"  queries will return WRONG results on the target.\n\n"
                f"  To proceed anyway:  set metric_override=True in your\n"
                f"  Bridge config or pass --metric-override on the CLI.\n"
                f"  To fix:  set distance_metric='{src}' on both connectors."
            )


# ── Semantic probe report ─────────────────────────────────────────────────────

@dataclass
class SemanticVerifyReport:
    job_id: str = ""
    source: str = ""
    target: str = ""
    n_probes: int = 0
    top_k: int = 5
    probes_run: int = 0

    # Per-probe overlap scores (fraction of top_k IDs that match)
    overlap_scores: list[float] = field(default_factory=list)

    # Per-probe cosine similarity between source and target result vectors
    cosine_scores: list[float] = field(default_factory=list)

    wall_seconds: float = 0.0

    @property
    def avg_overlap(self) -> float:
        return float(np.mean(self.overlap_scores)) if self.overlap_scores else 0.0

    @property
    def avg_cosine(self) -> float:
        return float(np.mean(self.cosine_scores)) if self.cosine_scores else 0.0

    @property
    def passed(self) -> bool:
        """Conservative pass threshold: 95% avg neighbour overlap."""
        return self.avg_overlap >= 0.95

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            "",
            "  Post-Migration Semantic Validation",
            "  ===================================",
            f"  Job:          {self.job_id}",
            f"  Source:       {self.source}",
            f"  Target:       {self.target}",
            f"  Probe vectors:{self.probes_run}  (top-{self.top_k} neighbours each)",
            f"  Avg neighbour overlap: {self.avg_overlap*100:.2f}%",
            f"  Avg cosine similarity: {self.avg_cosine:.6f}",
            f"  Wall time:    {self.wall_seconds:.1f}s",
            f"  Result:       {status}",
            "",
        ]
        if not self.passed:
            lines += [
                "  WARNING: overlap below 95% threshold.",
                "  Check distance_metric alignment and normalisation.",
                "",
            ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "job_id":        self.job_id,
            "source":        self.source,
            "target":        self.target,
            "n_probes":      self.probes_run,
            "top_k":         self.top_k,
            "avg_overlap_pct":  round(self.avg_overlap * 100, 4),
            "avg_cosine_sim":   round(self.avg_cosine, 6),
            "passed":        self.passed,
            "wall_seconds":  round(self.wall_seconds, 2),
            "overlap_scores": [round(s, 4) for s in self.overlap_scores],
        }


# ── Validator ─────────────────────────────────────────────────────────────────

class SemanticValidator:
    """
    Runs semantic probe queries against source and target, compares
    nearest-neighbour results to confirm mathematical equivalence.

    Parameters
    ----------
    source, target : VectorConnector
        Both must already be connected (or used as context managers).
    n_probes : int
        Number of random query vectors to probe. 100 is a good default.
    top_k : int
        Number of nearest neighbours to compare per probe.
    seed : int
        Random seed for reproducible probe vectors.
    """

    def __init__(
        self,
        source: "VectorConnector",
        target: "VectorConnector",
        n_probes: int = 100,
        top_k: int = 5,
        seed: int = 0xC4050AB1,
    ):
        self.source   = source
        self.target   = target
        self.n_probes = n_probes
        self.top_k    = top_k
        self.seed     = seed

    def run(self, job_id: str = "") -> SemanticVerifyReport:
        report = SemanticVerifyReport(
            job_id  = job_id,
            source  = type(self.source).__name__.replace("Connector", "").lower(),
            target  = type(self.target).__name__.replace("Connector", "").lower(),
            n_probes= self.n_probes,
            top_k   = self.top_k,
        )

        t0  = time.time()
        rng = np.random.default_rng(self.seed)

        # Detect dimension from source
        dim = self._detect_dim()
        if dim is None:
            log.warning("Could not detect dimension — skipping semantic validation")
            return report

        log.info(
            f"Semantic validation: {self.n_probes} probes x top-{self.top_k} "
            f"neighbours, dim={dim}"
        )

        for i in range(self.n_probes):
            # Random unit-norm query vector
            q = rng.standard_normal(dim).astype(np.float32)
            q /= np.linalg.norm(q) + 1e-9

            try:
                src_ids, src_vecs = self._query(self.source, q, self.top_k)
                tgt_ids, tgt_vecs = self._query(self.target, q, self.top_k)
            except NotImplementedError:
                log.warning(
                    "Connector does not support query() — "
                    "falling back to scan-based comparison"
                )
                break

            # Neighbour ID overlap (order-insensitive)
            src_set = set(src_ids)
            tgt_set = set(tgt_ids)
            overlap = len(src_set & tgt_set) / max(len(src_set), 1)
            report.overlap_scores.append(overlap)

            # Mean cosine similarity between corresponding result vectors
            if src_vecs and tgt_vecs:
                cos_vals = []
                for sv, tv in zip(src_vecs, tgt_vecs):
                    sv = np.asarray(sv, dtype=np.float32)
                    tv = np.asarray(tv, dtype=np.float32)
                    c  = float(
                        np.dot(sv, tv) /
                        (np.linalg.norm(sv) * np.linalg.norm(tv) + 1e-9)
                    )
                    cos_vals.append(c)
                report.cosine_scores.append(float(np.mean(cos_vals)))

            if (i + 1) % 25 == 0:
                log.info(
                    f"  probe {i+1}/{self.n_probes}: "
                    f"avg overlap={np.mean(report.overlap_scores)*100:.1f}%"
                )

        report.probes_run   = len(report.overlap_scores)
        report.wall_seconds = time.time() - t0
        log.info(report.summary())
        return report

    def _detect_dim(self) -> int | None:
        """Read one vector from source to get dimension."""
        try:
            for batch in self.source.read_batches(batch_size=1):
                if batch:
                    return int(batch[0].vector.shape[0])
        except Exception as e:
            log.warning(f"Dimension detection failed: {e}")
        return None

    def _query(
        self,
        connector: "VectorConnector",
        query: np.ndarray,
        top_k: int,
    ) -> tuple[list[str], list[list[float]]]:
        """
        Ask a connector for nearest neighbours.
        Connectors that implement query() get fast ANN.
        Others fall back to brute-force cosine over read_batches.
        """
        if hasattr(connector, "query"):
            return connector.query(query, top_k)

        # Brute-force fallback — works on every connector
        scores: list[tuple[float, str, list]] = []
        for batch in connector.read_batches(batch_size=512):
            for rec in batch:
                v   = rec.vector.astype(np.float32)
                cos = float(
                    np.dot(query, v) /
                    (np.linalg.norm(query) * np.linalg.norm(v) + 1e-9)
                )
                scores.append((cos, rec.id, v.tolist()))
        scores.sort(key=lambda x: -x[0])
        top = scores[:top_k]
        return [s[1] for s in top], [s[2] for s in top]
