"""
Shared fixtures for VectorBridge tests.

All tests use in-memory connectors — no external services required.
"""

import numpy as np
import pytest

from vectorbridge.connectors.base import VectorRecord, ConnectorConfig, VectorConnector


# ── In-memory connector ───────────────────────────────────────────────────────

class MemoryConnector(VectorConnector):
    """Minimal in-memory vector store for tests."""

    def __init__(self, records=None, metric="cosine"):
        cfg = ConnectorConfig(distance_metric=metric)
        super().__init__(cfg)
        self._records: list[VectorRecord] = list(records or [])

    def connect(self): pass
    def disconnect(self): pass
    def count(self): return len(self._records)
    def create_index(self, dimension, **kwargs): pass

    def read_batches(self, batch_size=256, offset=0):
        data = self._records[offset:]
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    def write_batch(self, records):
        self._records.extend(records)
        return len(records)

    def query(self, query_vec, top_k=5):
        q = np.asarray(query_vec, dtype=np.float32)
        scores = []
        for r in self._records:
            v = r.vector.astype(np.float32)
            cos = float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-9))
            scores.append((cos, r.id, v.tolist()))
        scores.sort(key=lambda x: -x[0])
        top = scores[:top_k]
        return [s[1] for s in top], [s[2] for s in top]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def dim():
    return 64


@pytest.fixture
def records(dim):
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((200, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    return [
        VectorRecord(id=f"vec_{i:04d}", vector=vecs[i], metadata={"idx": i})
        for i in range(200)
    ]


@pytest.fixture
def source(records):
    return MemoryConnector(records, metric="cosine")


@pytest.fixture
def empty_target():
    return MemoryConnector(metric="cosine")
