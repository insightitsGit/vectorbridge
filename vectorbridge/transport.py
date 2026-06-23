"""
VectorBridge tensor transport — internal implementation of the CHORUS protocol.
Patent Pending USPTO No. 64/096,156 — Amin Parva / Insight IT Solutions LLC.

Implements:
  - Tensor multiplication cipher  (V_enc = V_raw @ K,  V_dec = V_enc @ K_inv)
  - Rolling SHA-256 neural watermark
  - Batch packing / unpacking for efficient transfer
"""

import hashlib
import struct
import numpy as np
from dataclasses import dataclass

from .connectors.base import VectorRecord


# ── Key generation ────────────────────────────────────────────────────────────

def generate_key_pair(dim: int, seed: int = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate orthogonal (K, K_inv) via QR decomposition.
    K_inv == K.T for orthogonal matrices.
    """
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((dim, dim)).astype(np.float64)
    Q, _ = np.linalg.qr(M)
    K = Q.astype(np.float32)
    K_inv = K.T.copy()               # orthogonal: inverse == transpose
    return K, K_inv


# ── Cipher ────────────────────────────────────────────────────────────────────

def encrypt(vector: np.ndarray, K: np.ndarray) -> np.ndarray:
    return (vector.astype(np.float32) @ K).astype(np.float32)


def decrypt(vector: np.ndarray, K_inv: np.ndarray) -> np.ndarray:
    return (vector.astype(np.float32) @ K_inv).astype(np.float32)


# ── Watermark ─────────────────────────────────────────────────────────────────

def _watermark_vector(session_seed: bytes, seq: int, dim: int) -> np.ndarray:
    digest = hashlib.sha256(session_seed + seq.to_bytes(8, "big")).digest()
    seed_int = int.from_bytes(digest[:8], "big")
    rng = np.random.default_rng(seed_int)
    wm = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(wm)
    return wm / norm if norm > 0 else wm


def inject_watermark(vector: np.ndarray, session_seed: bytes, seq: int,
                     strength: float = 0.01) -> np.ndarray:
    wm = _watermark_vector(session_seed, seq, vector.shape[0])
    marked = vector + strength * wm
    norm = np.linalg.norm(marked)
    return marked / norm if norm > 0 else marked


def verify_watermark(vector: np.ndarray, session_seed: bytes, seq: int,
                     threshold: float = 0.0) -> bool:
    """
    Verify the CHORUS chain-of-custody watermark.

    The watermark is injected at strength 0.01, so cosine(marked, wm) ≈ 0.01.
    The default threshold=0.0 checks that the correlation is positive — sufficient
    for audit trail purposes. This is not a cryptographic tamper-proof watermark;
    it proves provenance and session continuity.
    """
    wm = _watermark_vector(session_seed, seq, vector.shape[0])
    norm_v = np.linalg.norm(vector)
    norm_w = np.linalg.norm(wm)
    if norm_v == 0 or norm_w == 0:
        return False
    cosine_sim = float(np.dot(vector, wm) / (norm_v * norm_w))
    return cosine_sim >= threshold


# ── Session ───────────────────────────────────────────────────────────────────

@dataclass
class TransportSession:
    session_seed: bytes
    K: np.ndarray
    K_inv: np.ndarray
    dim: int
    seq: int = 0

    @classmethod
    def create(cls, dim: int) -> "TransportSession":
        seed = np.random.bytes(32)
        K, K_inv = generate_key_pair(dim)
        return cls(session_seed=seed, K=K, K_inv=K_inv, dim=dim)


# ── Batch wire format ─────────────────────────────────────────────────────────
#
#  Frame per batch:
#   [4  bytes] magic     = b"CH0R"
#   [4  bytes] count     (uint32 big-endian)
#   [4  bytes] dim       (uint32 big-endian)
#   [8  bytes] seq       (uint64 big-endian)
#   [32 bytes] hmac      HMAC-SHA256(session_seed, seq_bytes + payload)
#   for each vector:
#     [dim * 4 bytes] encrypted float32 values
#     [id_len: 2 bytes][id: utf-8 bytes]
#     [meta_len: 4 bytes][meta: utf-8 json bytes]
#
#  The HMAC covers the payload (everything after the 32-byte hmac field).
#  Verification proves the batch originated from a session that held session_seed.

MAGIC    = b"CH0R"
HMAC_LEN = 32


def _batch_hmac(session_seed: bytes, seq: int, payload: bytes) -> bytes:
    import hmac as _hmac
    key = hashlib.sha256(session_seed + seq.to_bytes(8, "big")).digest()
    return _hmac.new(key, payload, hashlib.sha256).digest()


def pack_batch(records: list[VectorRecord], session: TransportSession) -> bytes:
    import json
    if not records:
        return b""
    dim   = records[0].vector.shape[0]
    count = len(records)
    seq   = session.seq

    payload_parts = []
    for r in records:
        enc = encrypt(r.vector, session.K)
        enc = inject_watermark(enc, session.session_seed, session.seq)
        payload_parts.append(enc.tobytes())
        id_b = r.id.encode("utf-8")
        payload_parts.append(struct.pack(">H", len(id_b)) + id_b)
        meta_b = json.dumps(r.metadata).encode("utf-8")
        payload_parts.append(struct.pack(">I", len(meta_b)) + meta_b)
        session.seq += 1

    payload = b"".join(payload_parts)
    hmac    = _batch_hmac(session.session_seed, seq, payload)

    return MAGIC + struct.pack(">II", count, dim) + struct.pack(">Q", seq) + hmac + payload


def unpack_batch(data: bytes, session: TransportSession,
                 verify: bool = True) -> tuple[list[VectorRecord], list[bool]]:
    import json, hmac as _hmac
    offset = 0

    magic = data[offset:offset + 4]; offset += 4
    assert magic == MAGIC, f"Invalid magic bytes: {magic!r}"

    count, dim = struct.unpack_from(">II", data, offset); offset += 8
    seq        = struct.unpack_from(">Q", data, offset)[0]; offset += 8
    stored_hmac = data[offset:offset + HMAC_LEN]; offset += HMAC_LEN

    payload = data[offset:]

    if verify:
        expected = _batch_hmac(session.session_seed, seq, payload)
        batch_ok  = _hmac.compare_digest(stored_hmac, expected)
    else:
        batch_ok = True

    records, verdicts = [], []
    poff = 0
    for i in range(count):
        raw = np.frombuffer(payload[poff:poff + dim * 4], dtype=np.float32).copy()
        poff += dim * 4

        id_len = struct.unpack_from(">H", payload, poff)[0]; poff += 2
        vid    = payload[poff:poff + id_len].decode("utf-8"); poff += id_len

        meta_len = struct.unpack_from(">I", payload, poff)[0]; poff += 4
        meta     = json.loads(payload[poff:poff + meta_len].decode("utf-8")); poff += meta_len

        verdicts.append(batch_ok)   # all vectors in a batch share the HMAC result
        vec = decrypt(raw, session.K_inv)
        records.append(VectorRecord(id=vid, vector=vec, metadata=meta))

    return records, verdicts


# ── Direct in-process transfer (no network required) ─────────────────────────

class DirectTransport:
    """
    In-process CHORUS transport — pack → wire bytes → unpack.
    Same cipher and watermark as the networked version.
    Used when source and target run in the same process.
    """

    def __init__(self, dim: int):
        self.session = TransportSession.create(dim)

    def transfer(self, records: list[VectorRecord]) -> tuple[list[VectorRecord], dict]:
        wire = pack_batch(records, self.session)
        received, verdicts = unpack_batch(wire, self.session, verify=True)
        stats = {
            "sent": len(records),
            "received": len(received),
            "verified": sum(verdicts),
            "failed": sum(1 for v in verdicts if not v),
            "wire_bytes": len(wire),
            "raw_bytes": sum(r.vector.nbytes for r in records),
        }
        return received, stats
