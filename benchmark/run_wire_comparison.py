"""
CHORUS vs REST Wire Format Benchmark
=====================================
Measures actual bytes on the wire and round-trip time for each transport format.

Formats tested:
  1. HTTP/REST JSON          - what most systems use today
  2. HTTP/REST JSON+gzip     - compressed REST (best-case HTTP)
  3. HTTP raw binary (float32) - plain binary over HTTP (gRPC-equivalent)
  4. HTTP CHORUS binary      - our encrypted + watermarked format

All formats use the same server on localhost:9000.
No Qdrant, no transatlantic. Pure wire comparison.
"""

import gzip, hashlib, json, math, os, struct, subprocess, sys, time
from pathlib import Path
import numpy as np
import requests

# ── Config ────────────────────────────────────────────────────────────────────

SERVER_URL  = "http://localhost:9000"
BATCH_SIZE  = 1000
NUM_BATCHES = 20         # 20K vectors total, ~2 minutes
DIM         = 1536
MAGIC       = b"CH0R"
WM_SEED     = 0xDEADBEEF

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── CHORUS primitives (inline — no VectorBridge dependency needed for bench) ──

def _make_key(dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A   = rng.standard_normal((dim, dim)).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    return Q.astype(np.float32)


def _wm_vec(seed: int, seq: int, dim: int) -> np.ndarray:
    h   = hashlib.sha256(f"{seed}:{seq}".encode()).digest()
    arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32) / 255.0 - 0.5
    arr = np.tile(arr, math.ceil(dim / len(arr)))[:dim]
    return (arr / (np.linalg.norm(arr) + 1e-9)).astype(np.float32)


K = _make_key(DIM)    # shared key, generated once


def chorus_pack(vecs: np.ndarray, ids: list[str], seq: int) -> bytes:
    """Encrypt + watermark + pack to CHORUS binary wire format."""
    count, dim = vecs.shape
    wm   = _wm_vec(WM_SEED, seq, dim) * 0.01
    enc  = (vecs @ K) + wm          # cipher + watermark

    buf = bytearray()
    buf += MAGIC
    buf += struct.pack(">IIQ", count, dim, seq)
    for i in range(count):
        ib = ids[i].encode()
        buf += struct.pack(">I", len(ib)) + ib
        buf += enc[i].astype(np.float32).tobytes()
    return bytes(buf)


def binary_pack(vecs: np.ndarray, ids: list[str]) -> bytes:
    """Raw float32 binary wire format — gRPC-equivalent."""
    count, dim = vecs.shape
    buf = bytearray()
    buf += struct.pack(">II", count, dim)
    for i in range(count):
        ib = ids[i].encode()
        buf += struct.pack(">I", len(ib)) + ib
        buf += vecs[i].astype(np.float32).tobytes()
    return bytes(buf)


# ── Data generation (in-memory, reproducible) ─────────────────────────────────

def make_batch(batch_idx: int) -> tuple[np.ndarray, list[str]]:
    rng  = np.random.default_rng(batch_idx)
    vecs = rng.standard_normal((BATCH_SIZE, DIM)).astype(np.float32)
    # L2-normalize so cosine similarity makes sense
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs  = vecs / (norms + 1e-9)
    ids   = [f"vec_{batch_idx:04d}_{i:06d}" for i in range(BATCH_SIZE)]
    return vecs, ids


# ── Server helpers ─────────────────────────────────────────────────────────────

def wait_for_server(timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


# ── Per-format senders ────────────────────────────────────────────────────────

def send_rest(vecs, ids, session):
    payload = json.dumps({"ids": ids, "vectors": vecs.tolist()}).encode("utf-8")
    t0 = time.perf_counter()
    r  = session.post(f"{SERVER_URL}/ingest/rest", data=payload,
                      headers={"Content-Type": "application/json"})
    rtt = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    return {"wire_bytes": len(payload), "rtt_ms": round(rtt, 2), **r.json()}


def send_rest_gz(vecs, ids, session):
    raw     = json.dumps({"ids": ids, "vectors": vecs.tolist()}).encode("utf-8")
    payload = gzip.compress(raw, compresslevel=6)
    t0  = time.perf_counter()
    r   = session.post(f"{SERVER_URL}/ingest/rest", data=payload,
                       headers={"Content-Type": "application/json",
                                "Content-Encoding": "gzip"})
    rtt = (time.perf_counter() - t0) * 1000
    # We measure the JSON payload before compression for wire_bytes
    # but actual bytes on wire is the gzip blob
    return {"wire_bytes_json": len(raw), "wire_bytes_gz": len(payload),
            "compression_ratio": round(len(raw) / len(payload), 2),
            "rtt_ms": round(rtt, 2)}


def send_binary(vecs, ids, session):
    payload = binary_pack(vecs, ids)
    t0 = time.perf_counter()
    r  = session.post(f"{SERVER_URL}/ingest/binary", data=payload,
                      headers={"Content-Type": "application/octet-stream"})
    rtt = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    return {"wire_bytes": len(payload), "rtt_ms": round(rtt, 2), **r.json()}


def send_chorus(vecs, ids, seq, session):
    payload = chorus_pack(vecs, ids, seq)
    t0 = time.perf_counter()
    r  = session.post(f"{SERVER_URL}/ingest/chorus", data=payload,
                      headers={"Content-Type": "application/octet-stream"})
    rtt = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    return {"wire_bytes": len(payload), "rtt_ms": round(rtt, 2), **r.json()}


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print()
    print("=" * 65)
    print("  VectorBridge Wire Format Benchmark")
    print("  CHORUS Fabric vs HTTP/REST JSON vs Binary vs gzip-REST")
    print("=" * 65)
    print(f"  Batches : {NUM_BATCHES} x {BATCH_SIZE:,} vectors = {NUM_BATCHES*BATCH_SIZE:,} total")
    print(f"  Dims    : {DIM}")
    print(f"  Server  : {SERVER_URL}")
    print()

    if not wait_for_server():
        print("[ERROR] Server not responding at", SERVER_URL)
        print("  Start it first:")
        print("  cd benchmark && uvicorn wire_server:app --host 0.0.0.0 --port 9000")
        sys.exit(1)
    print("  Server OK")

    # Reset server-side stats
    requests.delete(f"{SERVER_URL}/stats")

    results = {"rest": [], "rest_gz": [], "binary": [], "chorus": []}

    session = requests.Session()

    for b in range(NUM_BATCHES):
        vecs, ids = make_batch(b)

        r_rest   = send_rest(vecs, ids, session)
        r_gz     = send_rest_gz(vecs, ids, session)
        r_bin    = send_binary(vecs, ids, session)
        r_chorus = send_chorus(vecs, ids, b, session)

        results["rest"].append(r_rest)
        results["rest_gz"].append(r_gz)
        results["binary"].append(r_bin)
        results["chorus"].append(r_chorus)

        if b % 5 == 0 or b == NUM_BATCHES - 1:
            print(f"  [{b+1:3d}/{NUM_BATCHES}] "
                  f"REST {r_rest['wire_bytes']//1024:,}KB  "
                  f"gzip {r_gz['wire_bytes_gz']//1024:,}KB({r_gz['compression_ratio']}x)  "
                  f"binary {r_bin['wire_bytes']//1024:,}KB  "
                  f"CHORUS {r_chorus['wire_bytes']//1024:,}KB  "
                  f"| rtt REST:{r_rest['rtt_ms']:.0f}ms CHORUS:{r_chorus['rtt_ms']:.0f}ms")

    # ── Aggregate ──────────────────────────────────────────────────────────────

    def avg(rows, key): return sum(r[key] for r in rows) / len(rows)

    rest_bytes   = avg(results["rest"],    "wire_bytes")
    gz_bytes     = avg(results["rest_gz"], "wire_bytes_gz")
    binary_bytes = avg(results["binary"],  "wire_bytes")
    chorus_bytes = avg(results["chorus"],  "wire_bytes")

    rest_rtt   = avg(results["rest"],   "rtt_ms")
    binary_rtt = avg(results["binary"], "rtt_ms")
    chorus_rtt = avg(results["chorus"], "rtt_ms")

    rest_parse   = avg(results["rest"],   "parse_ms")
    binary_parse = avg(results["binary"], "parse_ms")
    chorus_parse = avg(results["chorus"], "parse_ms")

    # Bandwidth ratios vs REST/JSON (vs chorus)
    vs_rest   = rest_bytes   / chorus_bytes
    vs_gz     = gz_bytes     / chorus_bytes
    vs_binary = binary_bytes / chorus_bytes
    vs_rest_rtt   = rest_rtt   / chorus_rtt
    vs_binary_rtt = binary_rtt / chorus_rtt

    # Serialize timing
    sample_vecs, sample_ids = make_batch(99)
    t0 = time.perf_counter()
    for _ in range(10): json.dumps({"ids": sample_ids, "vectors": sample_vecs.tolist()}).encode()
    json_ser_ms = (time.perf_counter() - t0) / 10 * 1000

    t0 = time.perf_counter()
    for _ in range(10): chorus_pack(sample_vecs, sample_ids, 999)
    chorus_ser_ms = (time.perf_counter() - t0) / 10 * 1000

    t0 = time.perf_counter()
    for _ in range(10): binary_pack(sample_vecs, sample_ids)
    binary_ser_ms = (time.perf_counter() - t0) / 10 * 1000

    # ── Print table ────────────────────────────────────────────────────────────

    print()
    print("=" * 65)
    print("  WIRE FORMAT COMPARISON  (per batch of 1,000 x 1,536-dim)")
    print("=" * 65)
    print(f"  {'Format':<28} {'Wire KB':>8} {'vs REST':>8} {'RTT ms':>8} {'Parse ms':>10}")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    print(f"  {'HTTP/REST JSON':<28} {rest_bytes/1024:>8.0f} {'baseline':>8} {rest_rtt:>8.1f} {rest_parse:>10.1f}")
    print(f"  {'HTTP/REST JSON+gzip':<28} {gz_bytes/1024:>8.0f} {f'{vs_gz:.2f}x':>8} {'N/A':>8} {'N/A':>10}")
    print(f"  {'HTTP raw binary (float32)':<28} {binary_bytes/1024:>8.0f} {f'{vs_binary:.2f}x':>8} {binary_rtt:>8.1f} {binary_parse:>10.1f}")
    print(f"  {'CHORUS Fabric (enc+wm)':<28} {chorus_bytes/1024:>8.0f} {f'{vs_rest:.2f}x':>8} {chorus_rtt:>8.1f} {chorus_parse:>10.1f}")
    print()
    print("  Serialization time (1,000 x 1,536-dim, 10 reps):")
    print(f"    JSON serialize        : {json_ser_ms:.0f} ms")
    print(f"    Binary pack           : {binary_ser_ms:.0f} ms")
    print(f"    CHORUS pack (enc+wm)  : {chorus_ser_ms:.0f} ms")
    print()
    print("  CHORUS vs REST:")
    print(f"    Bandwidth reduction   : {vs_rest:.2f}x less data")
    print(f"    vs gzip REST          : {vs_gz:.2f}x less data")
    print(f"    vs raw binary         : {vs_binary:.2f}x less data")
    print(f"    RTT vs REST           : {vs_rest_rtt:.2f}x faster")
    print(f"    RTT vs binary         : {vs_binary_rtt:.2f}x faster")

    total_gb_rest   = rest_bytes   * NUM_BATCHES / 1e9
    total_gb_chorus = chorus_bytes * NUM_BATCHES / 1e9
    savings_gb = total_gb_rest - total_gb_chorus
    print()
    print(f"  Over {NUM_BATCHES*BATCH_SIZE:,} vectors:")
    print(f"    REST total data       : {total_gb_rest*1024:.1f} MB")
    print(f"    CHORUS total data     : {total_gb_chorus*1024:.1f} MB")
    print(f"    Saved                 : {savings_gb*1024:.1f} MB ({savings_gb/total_gb_rest*100:.0f}%)")
    print("=" * 65)

    # ── Save results ───────────────────────────────────────────────────────────

    ts  = int(time.time())
    out = {
        "ts": ts,
        "config": {"batch_size": BATCH_SIZE, "num_batches": NUM_BATCHES, "dim": DIM},
        "per_batch": results,
        "summary": {
            "avg_wire_bytes": {
                "rest_json": round(rest_bytes), "rest_gz": round(gz_bytes),
                "binary": round(binary_bytes), "chorus": round(chorus_bytes),
            },
            "vs_rest_ratio": {
                "gz": round(vs_gz, 3), "binary": round(vs_binary, 3), "chorus": round(vs_rest, 3)
            },
            "avg_rtt_ms": {
                "rest": round(rest_rtt, 2), "binary": round(binary_rtt, 2), "chorus": round(chorus_rtt, 2)
            },
            "avg_parse_ms": {
                "rest": round(rest_parse, 2), "binary": round(binary_parse, 2), "chorus": round(chorus_parse, 2)
            },
            "serialize_ms": {
                "json": round(json_ser_ms, 2), "binary": round(binary_ser_ms, 2), "chorus": round(chorus_ser_ms, 2)
            },
            "total_vectors": NUM_BATCHES * BATCH_SIZE,
            "bandwidth_saved_pct": round(savings_gb / total_gb_rest * 100, 1),
        }
    }

    out_path = RESULTS_DIR / f"wire_comparison_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # Landing page stats
    lp = {
        "bandwidth_reduction_vs_rest": f"{vs_rest:.2f}x",
        "bandwidth_reduction_vs_gzip": f"{vs_gz:.2f}x",
        "chorus_wire_kb_per_1k_vecs": round(chorus_bytes / 1024),
        "rest_wire_kb_per_1k_vecs": round(rest_bytes / 1024),
        "rtt_speedup_vs_rest": f"{vs_rest_rtt:.2f}x",
        "serialize_overhead_ms": round(chorus_ser_ms - binary_ser_ms, 1),
        "total_vectors_tested": NUM_BATCHES * BATCH_SIZE,
        "bandwidth_saved_pct": round(savings_gb / total_gb_rest * 100, 1),
        "generated_at": ts,
    }
    with open(RESULTS_DIR / "landing_page_wire_stats.json", "w") as f:
        json.dump(lp, f, indent=2)

    print(f"\n  Results saved to {out_path}")
    print(f"  Landing page stats: {RESULTS_DIR}/landing_page_wire_stats.json")
    print()


if __name__ == "__main__":
    run()
