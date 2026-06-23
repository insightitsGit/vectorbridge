"""
Tests for metric mismatch guard and semantic validator.
"""

import numpy as np
import pytest

from vectorbridge.verify import validate_metrics, MetricMismatchError, SemanticValidator
from tests.conftest import MemoryConnector


# ── Metric guard ──────────────────────────────────────────────────────────────

def test_matching_metrics_pass():
    validate_metrics("cosine", "cosine")   # must not raise


def test_alias_normalisation_pass():
    validate_metrics("cos", "cosine")
    validate_metrics("l2", "euclidean")
    validate_metrics("dot", "ip")
    validate_metrics("dot_product", "dot")


def test_mismatch_raises():
    with pytest.raises(MetricMismatchError):
        validate_metrics("cosine", "l2")


def test_mismatch_with_override_passes(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="vectorbridge.verify"):
        validate_metrics("cosine", "l2", override=True)
    assert "mismatch" in caplog.text.lower()


def test_dot_vs_cosine_raises():
    with pytest.raises(MetricMismatchError):
        validate_metrics("dot", "cosine")


def test_error_message_contains_metrics():
    try:
        validate_metrics("cosine", "l2")
    except MetricMismatchError as e:
        assert "cosine" in str(e)
        assert "l2" in str(e)


# ── Semantic validator ────────────────────────────────────────────────────────

def make_store(n=100, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    from vectorbridge.connectors.base import VectorRecord
    records = [VectorRecord(id=f"v{i}", vector=vecs[i], metadata={}) for i in range(n)]
    return MemoryConnector(records)


def test_semantic_validator_identical_stores_passes():
    """Exact copy of source should pass 100% overlap."""
    src = make_store(100, dim=32)
    # Build target as exact copy
    tgt = make_store(100, dim=32)   # same seed → identical vectors

    validator = SemanticValidator(src, tgt, n_probes=20, top_k=5, seed=1)
    with src, tgt:
        report = validator.run(job_id="test")

    assert report.passed, f"Identical stores should pass (got {report.avg_overlap*100:.1f}%)"
    assert report.avg_overlap > 0.95


def test_semantic_validator_random_stores_fails():
    """Random store that differs from source should fail."""
    src = make_store(100, dim=32, seed=0)
    tgt = make_store(100, dim=32, seed=99)   # completely different vectors

    validator = SemanticValidator(src, tgt, n_probes=20, top_k=5, seed=1)
    with src, tgt:
        report = validator.run(job_id="test")

    assert not report.passed, "Completely different stores should fail"


def test_semantic_validator_report_fields():
    src = make_store(50, dim=16)
    tgt = make_store(50, dim=16)

    validator = SemanticValidator(src, tgt, n_probes=10, top_k=3, seed=2)
    with src, tgt:
        report = validator.run(job_id="unit_test")

    assert report.probes_run == 10
    assert report.top_k == 3
    assert report.job_id == "unit_test"
    assert report.wall_seconds > 0
    assert len(report.overlap_scores) == 10


def test_semantic_report_to_dict():
    src = make_store(50, dim=16)
    tgt = make_store(50, dim=16)
    validator = SemanticValidator(src, tgt, n_probes=5, top_k=3)
    with src, tgt:
        report = validator.run()
    d = report.to_dict()
    assert "avg_overlap_pct" in d
    assert "passed" in d
    assert "n_probes" in d
    assert isinstance(d["overlap_scores"], list)
