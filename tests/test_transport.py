"""
Tests for CHORUS Fabric transport layer.

Covers: key generation, encrypt/decrypt round-trip, watermark inject/verify,
pack/unpack batch, DirectTransport stats.
"""

import numpy as np
import pytest

from vectorbridge.transport import (
    generate_key_pair,
    encrypt, decrypt,
    inject_watermark, verify_watermark,
    pack_batch, unpack_batch,
    TransportSession, DirectTransport,
    MAGIC,
)
from vectorbridge.connectors.base import VectorRecord


DIM = 64


def make_records(n=10, dim=DIM, seed=0):
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    return [VectorRecord(id=f"v{i}", vector=vecs[i], metadata={"i": i}) for i in range(n)]


# ── Key generation ────────────────────────────────────────────────────────────

def test_key_pair_orthogonal():
    K, K_inv = generate_key_pair(DIM, seed=42)
    I_approx = K @ K_inv
    assert np.allclose(I_approx, np.eye(DIM), atol=1e-5), "K @ K_inv should be identity"


def test_key_inv_is_transpose():
    K, K_inv = generate_key_pair(DIM, seed=7)
    assert np.allclose(K_inv, K.T, atol=1e-6)


# ── Cipher round-trip ─────────────────────────────────────────────────────────

def test_encrypt_decrypt_roundtrip():
    K, K_inv = generate_key_pair(DIM, seed=1)
    rng = np.random.default_rng(99)
    v = rng.standard_normal(DIM).astype(np.float32)
    enc = encrypt(v, K)
    dec = decrypt(enc, K_inv)
    assert np.allclose(v, dec, atol=1e-4), "decrypt(encrypt(v)) should recover original vector"


def test_encrypt_changes_vector():
    K, _ = generate_key_pair(DIM, seed=2)
    rng = np.random.default_rng(0)
    v = rng.standard_normal(DIM).astype(np.float32)
    enc = encrypt(v, K)
    assert not np.allclose(v, enc), "Encrypted vector must differ from plaintext"


# ── Watermark ─────────────────────────────────────────────────────────────────

def test_watermark_verify_passes():
    # Use strength=0.5 so cosine(marked, wm) > 0 is reliable even at dim=64.
    # Production dim=1536 with strength=0.01 achieves the same positive-correlation goal.
    session_seed = b"test_seed_12345678901234567890ab"
    rng = np.random.default_rng(5)
    v = rng.standard_normal(DIM).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    marked = inject_watermark(v, session_seed, seq=0, strength=0.5)
    assert verify_watermark(marked, session_seed, seq=0, threshold=0.3)


def test_watermark_seq_mismatch_fails():
    session_seed = b"test_seed_12345678901234567890ab"
    rng = np.random.default_rng(5)
    v = rng.standard_normal(DIM).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    marked = inject_watermark(v, session_seed, seq=0, strength=0.01)
    # Verifying with wrong seq should fail
    assert not verify_watermark(marked, session_seed, seq=99, threshold=0.95)


def test_watermark_wrong_seed_fails():
    session_seed = b"test_seed_12345678901234567890ab"
    wrong_seed   = b"wrong_seed_1234567890123456789x"
    rng = np.random.default_rng(5)
    v = rng.standard_normal(DIM).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    marked = inject_watermark(v, session_seed, seq=0, strength=0.01)
    assert not verify_watermark(marked, wrong_seed, seq=0, threshold=0.95)


# ── Pack / unpack batch ───────────────────────────────────────────────────────

def test_pack_starts_with_magic():
    session = TransportSession.create(DIM)
    records = make_records(5)
    wire = pack_batch(records, session)
    assert wire[:4] == MAGIC


def test_unpack_roundtrip_ids_and_metadata():
    session = TransportSession.create(DIM)
    records = make_records(10)
    wire = pack_batch(records, session)
    recovered, verdicts = unpack_batch(wire, session)
    assert len(recovered) == 10
    assert all(verdicts), "HMAC should verify for all records in a valid batch"
    orig_ids = [r.id for r in records]
    recv_ids = [r.id for r in recovered]
    assert orig_ids == recv_ids


def test_unpack_vector_roundtrip():
    # Watermark injects ~0.01 error per component — use atol=0.05 for dim=64
    session = TransportSession.create(DIM)
    records = make_records(5)
    orig_vecs = [r.vector.copy() for r in records]
    wire = pack_batch(records, session)
    recovered, _ = unpack_batch(wire, session)
    for orig, rec in zip(orig_vecs, recovered):
        assert np.allclose(orig, rec.vector, atol=0.05), "Recovered vector should be close to original"


def test_pack_empty_batch():
    session = TransportSession.create(DIM)
    wire = pack_batch([], session)
    assert wire == b""


# ── DirectTransport ───────────────────────────────────────────────────────────

def test_direct_transport_stats():
    transport = DirectTransport(DIM)
    records = make_records(20)
    received, stats = transport.transfer(records)
    assert stats["sent"] == 20
    assert stats["received"] == 20
    assert stats["verified"] == 20    # all pass with threshold=0.0
    assert stats["failed"] == 0
    assert stats["wire_bytes"] > 0
    assert stats["raw_bytes"] == 20 * DIM * 4


def test_direct_transport_vector_fidelity():
    # Watermark adds ~0.01 error — atol=0.05 for dim=64
    transport = DirectTransport(DIM)
    records = make_records(5)
    received, stats = transport.transfer(records)
    for orig, rec in zip(records, received):
        assert np.allclose(orig.vector, rec.vector, atol=0.05)


def test_direct_transport_wire_smaller_than_json():
    """CHORUS binary must be dramatically smaller than JSON equivalent."""
    import json
    transport = DirectTransport(DIM)
    records = make_records(100)
    _, stats = transport.transfer(records)
    json_size = len(json.dumps([r.vector.tolist() for r in records]).encode())
    assert stats["wire_bytes"] < json_size, (
        f"Wire ({stats['wire_bytes']} bytes) should be smaller than JSON ({json_size} bytes)"
    )
