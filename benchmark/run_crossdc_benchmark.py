"""
VectorBridge Cross-Datacenter Wire Format Benchmark
====================================================
US East (Virginia) --> US West (Washington) real network RTT.

Tests four formats against each remote server:
  1. HTTP/REST JSON
  2. HTTP/REST JSON + gzip
  3. HTTP raw binary (float32)
  4. HTTP CHORUS binary (encrypted + watermarked)

Saves full per-batch logs for landing page and README.

Usage:
    python run_crossdc_benchmark.py \
        --east <IP>:9000 \
        --west <IP>:9000 \
        [--batches 30] [--batch-size 1000] [--dim 1536]
"""

import argparse
import gzip
import hashlib
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np
import requests

# ── Config ─────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MAGIC    = b"CH0R"
WM_SEED  = 0xDEADBEEF

# ── CHORUS primitives ──────────────────────────────────────────────────────────

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


def chorus_pack(vecs: np.ndarray, ids: list, seq: int, K: np.ndarray) -> bytes:
    count, dim = vecs.shape
    wm  = _wm_vec(WM_SEED, seq, dim) * 0.01
    enc = (vecs @ K) + wm
    buf = bytearray()
    buf += MAGIC
    buf += struct.pack(">IIQ", count, dim, seq)
    for i in range(count):
        ib = ids[i].encode()
        buf += struct.pack(">I", len(ib)) + ib
        buf += enc[i].astype(np.float32).tobytes()
    return bytes(buf)


def binary_pack(vecs: np.ndarray, ids: list) -> bytes:
    count, dim = vecs.shape
    buf = bytearray()
    buf += struct.pack(">II", count, dim)
    for i in range(count):
        ib = ids[i].encode()
        buf += struct.pack(">I", len(ib)) + ib
        buf += vecs[i].astype(np.float32).tobytes()
    return bytes(buf)


# ── Batch generator ────────────────────────────────────────────────────────────

def make_batch(idx: int, batch_size: int, dim: int):
    rng  = np.random.default_rng(idx)
    vecs = rng.standard_normal((batch_size, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs  = vecs / (norms + 1e-9)
    ids   = [f"vec_{idx:04d}_{i:06d}" for i in range(batch_size)]
    return vecs, ids


# ── Server health check ────────────────────────────────────────────────────────

def wait_for_server(url: str, label: str, timeout: int = 60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=3)
            if r.status_code == 200:
                print(f"  [{label}] OK  ({url})")
                return True
        except Exception:
            pass
        time.sleep(1)
    print(f"  [{label}] TIMEOUT — not reachable at {url}")
    return False


# ── Format senders ─────────────────────────────────────────────────────────────

def send_rest(url, vecs, ids, session):
    payload = json.dumps({"ids": ids, "vectors": vecs.tolist()}).encode("utf-8")
    t0 = time.perf_counter()
    r  = session.post(f"{url}/ingest/rest", data=payload,
                      headers={"Content-Type": "application/json"}, timeout=120)
    rtt = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    return {"wire_bytes": len(payload), "rtt_ms": round(rtt, 2), **r.json()}


def send_rest_gz(url, vecs, ids, session):
    raw     = json.dumps({"ids": ids, "vectors": vecs.tolist()}).encode("utf-8")
    payload = gzip.compress(raw, compresslevel=6)
    t0  = time.perf_counter()
    r   = session.post(f"{url}/ingest/rest", data=payload,
                       headers={"Content-Type": "application/json",
                                "Content-Encoding": "gzip"}, timeout=120)
    rtt = (time.perf_counter() - t0) * 1000
    return {
        "wire_bytes_json": len(raw), "wire_bytes_gz": len(payload),
        "compression_ratio": round(len(raw) / len(payload), 2),
        "rtt_ms": round(rtt, 2),
    }


def send_binary(url, vecs, ids, session):
    payload = binary_pack(vecs, ids)
    t0 = time.perf_counter()
    r  = session.post(f"{url}/ingest/binary", data=payload,
                      headers={"Content-Type": "application/octet-stream"}, timeout=120)
    rtt = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    return {"wire_bytes": len(payload), "rtt_ms": round(rtt, 2), **r.json()}


def send_chorus(url, vecs, ids, seq, K, session):
    payload = chorus_pack(vecs, ids, seq, K)
    t0 = time.perf_counter()
    r  = session.post(f"{url}/ingest/chorus", data=payload,
                      headers={"Content-Type": "application/octet-stream"}, timeout=120)
    rtt = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    return {"wire_bytes": len(payload), "rtt_ms": round(rtt, 2), **r.json()}


# ── Per-region benchmark ───────────────────────────────────────────────────────

def bench_region(label: str, url: str, num_batches: int, batch_size: int,
                 dim: int, K: np.ndarray) -> dict:
    print(f"\n{'='*65}")
    print(f"  {label}  -->  {url}")
    print(f"  {num_batches} batches x {batch_size:,} vectors x {dim} dims")
    print(f"{'='*65}")

    session = requests.Session()
    requests.delete(f"{url}/stats", timeout=5)

    rows: dict = {"rest": [], "rest_gz": [], "binary": [], "chorus": []}
    batch_log = []

    for b in range(num_batches):
        vecs, ids = make_batch(b, batch_size, dim)
        ts_batch  = time.time()

        r_rest   = send_rest(url, vecs, ids, session)
        r_gz     = send_rest_gz(url, vecs, ids, session)
        r_bin    = send_binary(url, vecs, ids, session)
        r_chorus = send_chorus(url, vecs, ids, b, K, session)

        rows["rest"].append(r_rest)
        rows["rest_gz"].append(r_gz)
        rows["binary"].append(r_bin)
        rows["chorus"].append(r_chorus)

        log_entry = {
            "batch": b,
            "ts": round(ts_batch, 3),
            "rest":   {"wire_kb": round(r_rest["wire_bytes"]/1024),   "rtt_ms": r_rest["rtt_ms"],   "parse_ms": r_rest.get("parse_ms")},
            "rest_gz":{"wire_kb": round(r_gz["wire_bytes_gz"]/1024),  "rtt_ms": r_gz["rtt_ms"],     "ratio": r_gz["compression_ratio"]},
            "binary": {"wire_kb": round(r_bin["wire_bytes"]/1024),    "rtt_ms": r_bin["rtt_ms"],    "parse_ms": r_bin.get("parse_ms")},
            "chorus": {"wire_kb": round(r_chorus["wire_bytes"]/1024), "rtt_ms": r_chorus["rtt_ms"], "parse_ms": r_chorus.get("parse_ms"),
                       "wm_cosine": r_chorus.get("wm_cosine")},
        }
        batch_log.append(log_entry)

        if b % 5 == 0 or b == num_batches - 1:
            print(f"  [{b+1:3d}/{num_batches}] "
                  f"REST {r_rest['wire_bytes']//1024:,}KB/{r_rest['rtt_ms']:.0f}ms  "
                  f"GZ {r_gz['wire_bytes_gz']//1024:,}KB/{r_gz['rtt_ms']:.0f}ms  "
                  f"BIN {r_bin['wire_bytes']//1024:,}KB/{r_bin['rtt_ms']:.0f}ms  "
                  f"CHORUS {r_chorus['wire_bytes']//1024:,}KB/{r_chorus['rtt_ms']:.0f}ms")

    def avg(lst, key): return sum(r[key] for r in lst) / len(lst)

    rest_bytes   = avg(rows["rest"],    "wire_bytes")
    gz_bytes     = avg(rows["rest_gz"], "wire_bytes_gz")
    binary_bytes = avg(rows["binary"],  "wire_bytes")
    chorus_bytes = avg(rows["chorus"],  "wire_bytes")

    rest_rtt   = avg(rows["rest"],   "rtt_ms")
    gz_rtt     = avg(rows["rest_gz"],"rtt_ms")
    binary_rtt = avg(rows["binary"], "rtt_ms")
    chorus_rtt = avg(rows["chorus"], "rtt_ms")

    rest_parse   = avg(rows["rest"],   "parse_ms")
    binary_parse = avg(rows["binary"], "parse_ms")
    chorus_parse = avg(rows["chorus"], "parse_ms")

    vs_rest   = rest_bytes   / chorus_bytes
    vs_gz     = gz_bytes     / chorus_bytes
    vs_binary = binary_bytes / chorus_bytes

    rtt_vs_rest   = rest_rtt   / chorus_rtt
    rtt_vs_binary = binary_rtt / chorus_rtt

    total_mb_rest   = rest_bytes   * num_batches / 1e6
    total_mb_chorus = chorus_bytes * num_batches / 1e6
    saved_mb        = total_mb_rest - total_mb_chorus
    saved_pct       = saved_mb / total_mb_rest * 100

    print(f"\n  {'Format':<28} {'Wire KB':>8} {'vs REST':>8} {'RTT ms':>9} {'Parse ms':>10}")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*9} {'-'*10}")
    print(f"  {'HTTP/REST JSON':<28} {rest_bytes/1024:>8.0f} {'baseline':>8} {rest_rtt:>9.1f} {rest_parse:>10.1f}")
    print(f"  {'HTTP/REST JSON+gzip':<28} {gz_bytes/1024:>8.0f} {f'{vs_gz:.2f}x':>8} {gz_rtt:>9.1f} {'N/A':>10}")
    print(f"  {'HTTP raw binary (float32)':<28} {binary_bytes/1024:>8.0f} {f'{vs_binary:.2f}x':>8} {binary_rtt:>9.1f} {binary_parse:>10.1f}")
    print(f"  {'CHORUS Fabric (enc+wm)':<28} {chorus_bytes/1024:>8.0f} {f'{vs_rest:.2f}x':>8} {chorus_rtt:>9.1f} {chorus_parse:>10.1f}")
    print()
    print(f"  Bandwidth: CHORUS saves {saved_pct:.0f}% vs REST  ({saved_mb:.0f} MB over {num_batches*batch_size:,} vectors)")
    print(f"  RTT:       CHORUS {rtt_vs_rest:.1f}x faster than REST, {rtt_vs_binary:.1f}x faster than raw binary")
    print(f"  Watermark: {sum(1 for r in rows['chorus'] if r.get('wm_cosine', 0) > 0.99)}/{num_batches} batches verified (cosine > 0.99)")

    return {
        "label": label,
        "url": url,
        "config": {"batches": num_batches, "batch_size": batch_size, "dim": dim},
        "batch_log": batch_log,
        "summary": {
            "avg_wire_bytes": {
                "rest_json": round(rest_bytes),
                "rest_gz":   round(gz_bytes),
                "binary":    round(binary_bytes),
                "chorus":    round(chorus_bytes),
            },
            "avg_rtt_ms": {
                "rest":   round(rest_rtt,   2),
                "rest_gz":round(gz_rtt,     2),
                "binary": round(binary_rtt, 2),
                "chorus": round(chorus_rtt, 2),
            },
            "avg_parse_ms": {
                "rest":   round(rest_parse,   2),
                "binary": round(binary_parse, 2),
                "chorus": round(chorus_parse, 2),
            },
            "bandwidth_ratio": {
                "chorus_vs_rest":   round(vs_rest,   3),
                "chorus_vs_gz":     round(vs_gz,     3),
                "chorus_vs_binary": round(vs_binary, 3),
            },
            "rtt_ratio": {
                "chorus_vs_rest":   round(rtt_vs_rest,   2),
                "chorus_vs_binary": round(rtt_vs_binary, 2),
            },
            "total_vectors": num_batches * batch_size,
            "bandwidth_saved_pct": round(saved_pct, 1),
            "watermark_verified": f"{sum(1 for r in rows['chorus'] if r.get('wm_cosine',0) > 0.99)}/{num_batches}",
        }
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--east",       required=True, help="US East server URL, e.g. http://1.2.3.4:9000")
    ap.add_argument("--west",       required=True, help="US West server URL, e.g. http://5.6.7.8:9000")
    ap.add_argument("--batches",    type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument("--dim",        type=int, default=1536)
    args = ap.parse_args()

    dim = args.dim
    K   = _make_key(dim)

    print()
    print("=" * 65)
    print("  VectorBridge Cross-DC Wire Format Benchmark")
    print("  US East (Virginia) + US West (Washington)")
    print("  CHORUS Fabric vs HTTP/REST JSON vs Binary vs gzip-REST")
    print("=" * 65)
    print(f"  Dim: {dim}  |  Batches: {args.batches}  |  Batch size: {args.batch_size:,}")
    print(f"  Total vectors per region: {args.batches * args.batch_size:,}")

    print("\n  Checking server connectivity...")
    east_ok = wait_for_server(args.east, "US-East")
    west_ok = wait_for_server(args.west, "US-West")
    if not (east_ok and west_ok):
        print("\n  One or both servers unreachable. Aborting.")
        sys.exit(1)

    ts = int(time.time())

    # Run both regions
    east_result = bench_region("US East (Virginia)", args.east,
                               args.batches, args.batch_size, dim, K)
    west_result = bench_region("US West (Washington)", args.west,
                               args.batches, args.batch_size, dim, K)

    # ── Combined summary ───────────────────────────────────────────────────────

    print(f"\n{'='*65}")
    print("  COMBINED RESULTS — US East + US West")
    print(f"{'='*65}")

    for r in [east_result, west_result]:
        s = r["summary"]
        print(f"\n  {r['label']}")
        print(f"    Bandwidth  : {s['bandwidth_ratio']['chorus_vs_rest']:.2f}x less than REST "
              f"/ {s['bandwidth_ratio']['chorus_vs_gz']:.2f}x less than gzip")
        print(f"    RTT CHORUS : {s['avg_rtt_ms']['chorus']:.0f} ms  "
              f"(REST {s['avg_rtt_ms']['rest']:.0f} ms,  "
              f"binary {s['avg_rtt_ms']['binary']:.0f} ms)")
        print(f"    RTT ratio  : {s['rtt_ratio']['chorus_vs_rest']:.1f}x faster than REST")
        print(f"    Bandwidth saved : {s['bandwidth_saved_pct']}% vs REST")
        print(f"    Watermark  : {s['watermark_verified']} batches verified")

    # ── Save full log ──────────────────────────────────────────────────────────

    full_log = {
        "benchmark": "VectorBridge Cross-DC Wire Format Comparison",
        "description": "US East (Virginia) and US West (Washington) Azure Container Instances",
        "formats_tested": ["HTTP/REST JSON", "HTTP/REST JSON+gzip", "HTTP raw binary float32", "CHORUS Fabric (encrypted+watermarked)"],
        "generated_at_unix": ts,
        "regions": [east_result, west_result],
    }

    full_path = RESULTS_DIR / f"crossdc_full_{ts}.json"
    with open(full_path, "w") as f:
        json.dump(full_log, f, indent=2)

    # Landing page stats (simple, human-readable)
    es = east_result["summary"]
    ws = west_result["summary"]
    lp = {
        "generated_at": ts,
        "methodology": "Real HTTP requests, Azure Container Instances, 1536-dim float32 vectors",
        "us_east": {
            "location": "Virginia (Azure eastus)",
            "vectors_tested": es["total_vectors"],
            "chorus_wire_kb_per_1k": round(es["avg_wire_bytes"]["chorus"] / 1024),
            "rest_wire_kb_per_1k":   round(es["avg_wire_bytes"]["rest_json"] / 1024),
            "gz_wire_kb_per_1k":     round(es["avg_wire_bytes"]["rest_gz"] / 1024),
            "bandwidth_vs_rest":     f"{es['bandwidth_ratio']['chorus_vs_rest']:.2f}x",
            "bandwidth_vs_gzip":     f"{es['bandwidth_ratio']['chorus_vs_gz']:.2f}x",
            "chorus_rtt_ms":         es["avg_rtt_ms"]["chorus"],
            "rest_rtt_ms":           es["avg_rtt_ms"]["rest"],
            "rtt_speedup":           f"{es['rtt_ratio']['chorus_vs_rest']:.1f}x",
            "bandwidth_saved_pct":   es["bandwidth_saved_pct"],
            "watermark_verified":    es["watermark_verified"],
        },
        "us_west": {
            "location": "Washington (Azure westus2)",
            "vectors_tested": ws["total_vectors"],
            "chorus_wire_kb_per_1k": round(ws["avg_wire_bytes"]["chorus"] / 1024),
            "rest_wire_kb_per_1k":   round(ws["avg_wire_bytes"]["rest_json"] / 1024),
            "gz_wire_kb_per_1k":     round(ws["avg_wire_bytes"]["rest_gz"] / 1024),
            "bandwidth_vs_rest":     f"{ws['bandwidth_ratio']['chorus_vs_rest']:.2f}x",
            "bandwidth_vs_gzip":     f"{ws['bandwidth_ratio']['chorus_vs_gz']:.2f}x",
            "chorus_rtt_ms":         ws["avg_rtt_ms"]["chorus"],
            "rest_rtt_ms":           ws["avg_rtt_ms"]["rest"],
            "rtt_speedup":           f"{ws['rtt_ratio']['chorus_vs_rest']:.1f}x",
            "bandwidth_saved_pct":   ws["bandwidth_saved_pct"],
            "watermark_verified":    ws["watermark_verified"],
        },
    }
    lp_path = RESULTS_DIR / "landing_page_crossdc_stats.json"
    with open(lp_path, "w") as f:
        json.dump(lp, f, indent=2)

    print(f"\n  Full log  : {full_path}")
    print(f"  LP stats  : {lp_path}")
    print()


if __name__ == "__main__":
    main()
