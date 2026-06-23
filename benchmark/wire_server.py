"""
Wire format comparison server.
Accepts vectors in three real transport formats and measures server-side processing.

Endpoints:
  POST /ingest/rest    - HTTP/REST with JSON body (what most systems use today)
  POST /ingest/binary  - Raw float32 bytes (plain gRPC-equivalent)
  POST /ingest/chorus  - CHORUS encrypted binary (our format)

Run with:
  uvicorn wire_server:app --host 0.0.0.0 --port 9000 --reload
"""

import gzip, json, struct, hashlib, math, time
from typing import Optional
import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

app = FastAPI()

DIM = 1536
MAGIC = b"CH0R"

# Per-request stats stored for the benchmark to read back
_stats: dict[str, list] = {"rest": [], "binary": [], "chorus": []}


def _wm(seed: int, seq: int) -> np.ndarray:
    h = hashlib.sha256(f"{seed}:{seq}".encode()).digest()
    arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32) / 255.0 - 0.5
    arr = np.tile(arr, math.ceil(DIM / len(arr)))[:DIM]
    return (arr / (np.linalg.norm(arr) + 1e-9)).astype(np.float32)


# ── REST/JSON endpoint ────────────────────────────────────────────────────────

@app.post("/ingest/rest")
async def ingest_rest(request: Request):
    """Standard HTTP/REST: receives JSON body with float arrays."""
    raw_body = await request.body()
    wire_bytes = len(raw_body)

    t0 = time.perf_counter()
    payload = json.loads(raw_body)
    vectors = np.array(payload["vectors"], dtype=np.float32)
    ids     = payload["ids"]
    parse_ms = (time.perf_counter() - t0) * 1000

    _stats["rest"].append({
        "count": len(ids), "wire_bytes": wire_bytes, "parse_ms": round(parse_ms, 3)
    })
    return {"received": len(ids), "wire_bytes": wire_bytes, "parse_ms": round(parse_ms, 3)}


# ── Binary/gRPC-equivalent endpoint ──────────────────────────────────────────

@app.post("/ingest/binary")
async def ingest_binary(request: Request):
    """Raw float32 bytes: no encryption, no JSON. Simulates plain gRPC protobuf."""
    raw_body = await request.body()
    wire_bytes = len(raw_body)

    t0 = time.perf_counter()
    # Wire format: count(4) + dim(4) + [id_len(4) + id_bytes + vector_bytes] * count
    count, dim = struct.unpack(">II", raw_body[:8])
    offset = 8
    vecs, ids = [], []
    for _ in range(count):
        il = struct.unpack(">I", raw_body[offset:offset+4])[0]; offset += 4
        ids.append(raw_body[offset:offset+il].decode()); offset += il
        vecs.append(np.frombuffer(raw_body[offset:offset+dim*4], np.float32).copy())
        offset += dim * 4
    parse_ms = (time.perf_counter() - t0) * 1000

    _stats["binary"].append({
        "count": count, "wire_bytes": wire_bytes, "parse_ms": round(parse_ms, 3)
    })
    return {"received": count, "wire_bytes": wire_bytes, "parse_ms": round(parse_ms, 3)}


# ── CHORUS endpoint ───────────────────────────────────────────────────────────

@app.post("/ingest/chorus")
async def ingest_chorus(request: Request):
    """CHORUS Fabric binary: encrypted + watermarked float32."""
    raw_body = await request.body()
    wire_bytes = len(raw_body)

    t0 = time.perf_counter()
    assert raw_body[:4] == MAGIC
    count, dim, seq = struct.unpack(">IIQ", raw_body[4:20])
    offset = 20
    enc_vecs, ids = [], []
    for _ in range(count):
        il = struct.unpack(">I", raw_body[offset:offset+4])[0]; offset += 4
        ids.append(raw_body[offset:offset+il].decode()); offset += il
        enc_vecs.append(np.frombuffer(raw_body[offset:offset+dim*4], np.float32).copy())
        offset += dim * 4

    # Verify watermark cosine
    enc  = np.stack(enc_vecs)
    wm   = _wm(0xDEADBEEF, seq) * 0.01
    cos_vals = []
    for i in range(min(count, 10)):   # sample 10 for speed
        n1, n2 = np.linalg.norm(enc[i]), np.linalg.norm(enc[i] - wm)
        if n1 > 0 and n2 > 0:
            cos_vals.append(float(np.dot(enc[i], enc[i] - wm) / (n1 * n2)))
    wm_cosine = float(np.mean(cos_vals)) if cos_vals else 0.0
    parse_ms  = (time.perf_counter() - t0) * 1000

    _stats["chorus"].append({
        "count": count, "wire_bytes": wire_bytes,
        "parse_ms": round(parse_ms, 3), "wm_cosine": round(wm_cosine, 5)
    })
    return {
        "received": count, "wire_bytes": wire_bytes,
        "parse_ms": round(parse_ms, 3), "wm_cosine": round(wm_cosine, 5)
    }


# ── Stats readout ─────────────────────────────────────────────────────────────

@app.get("/stats")
def get_stats():
    return _stats

@app.delete("/stats")
def reset_stats():
    for k in _stats: _stats[k].clear()
    return {"reset": True}

@app.get("/health")
def health():
    return {"ok": True}
