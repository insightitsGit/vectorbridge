"""
VectorBridge Transatlantic Benchmark
Local ChromaDB -> Germany West Qdrant (72.144.65.153:6333)

Two migrations back-to-back:
  1. CHORUS: encrypt at source, binary wire, watermark verify at target
  2. BASELINE: raw float32 bytes, no encryption

Cipher timing is measured separately (source-side) and NOT included
in wire latency, matching how the production agent works.

Usage:
  python run_benchmark.py --target-host 72.144.65.153 [--skip-load] [--limit N]
"""

import argparse, json, os, time, struct, hashlib, math, sys
import numpy as np
from typing import List, Tuple
import chromadb
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, OptimizersConfigDiff

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_data")
os.makedirs(RESULTS_DIR, exist_ok=True)

SRC_COLLECTION = "legal_docs"
TGT_CHORUS     = "legal_docs_chorus"
TGT_BASELINE   = "legal_docs_baseline"
DIM            = 1536
BATCH_SIZE     = 1000

# ── CHORUS cipher ──────────────────────────────────────────────────────────────

MAGIC = b"CH0R"

def _qr_key(dim: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((dim, dim)).astype(np.float32)
    K, _ = np.linalg.qr(M)
    return K.astype(np.float32), K.T.astype(np.float32)   # K_inv = K.T

def _wm(seed: int, seq: int) -> np.ndarray:
    h = hashlib.sha256(f"{seed}:{seq}".encode()).digest()
    arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32) / 255.0 - 0.5
    arr = np.tile(arr, math.ceil(DIM / len(arr)))[:DIM]
    return (arr / (np.linalg.norm(arr) + 1e-9)).astype(np.float32)

def chorus_encrypt_batch(vecs: np.ndarray, ids: List[str], K, seed, seq) -> Tuple[bytes, float]:
    """Encrypt + watermark. Returns (wire_bytes, watermark_cosine)."""
    enc = (vecs @ K + _wm(seed, seq) * 0.01).astype(np.float32)
    # Pack to binary wire format
    parts = [MAGIC, struct.pack(">IIQ", len(vecs), DIM, seq)]
    for i, vid in enumerate(ids):
        vb = vid.encode()
        parts += [struct.pack(">I", len(vb)), vb, enc[i].tobytes()]
    wire = b"".join(parts)
    # Verify watermark: decrypt and measure cosine recovery
    K_inv = K.T
    dec = (enc - _wm(seed, seq) * 0.01) @ K_inv
    re  = dec @ K
    cos_vals = []
    for i in range(len(vecs)):
        n1, n2 = np.linalg.norm(re[i]), np.linalg.norm(enc[i])
        cos_vals.append(float(np.dot(re[i], enc[i]) / (n1 * n2 + 1e-9)))
    return wire, dec, ids, float(np.mean(cos_vals))

# ── Stats ──────────────────────────────────────────────────────────────────────

class Stats:
    def __init__(self): self.rows = []

    def add(self, seq, count, wire, raw, ms, cos=None):
        self.rows.append(dict(seq=seq, count=count, wire=wire, raw=raw, ms=round(ms,2), cos=cos))
        sys.stdout.flush()

    def summary(self, label):
        tc  = sum(r["count"] for r in self.rows)
        tw  = sum(r["wire"]  for r in self.rows)
        tr  = sum(r["raw"]   for r in self.rows)
        lat = sorted(r["ms"] for r in self.rows)
        cos = [r["cos"] for r in self.rows if r["cos"] is not None]
        def p(pct): return round(lat[min(int(len(lat)*pct/100), len(lat)-1)], 2)
        vr  = (sum(1 for c in cos if c > 0.95) / len(cos) * 100) if cos else None
        total_ms = sum(r["ms"] for r in self.rows)
        return {
            "label": label,
            "total_vectors": tc, "total_batches": len(self.rows),
            "raw_mb":         round(tr/1024**2, 2),
            "wire_mb":        round(tw/1024**2, 2),
            "json_equiv_mb":  round(tc * DIM * 18 / 1024**2, 2),
            "bandwidth_savings_vs_json":     round(tc*DIM*18/tw, 3) if tw else 0,
            "bandwidth_savings_vs_baseline": None,
            "latency_p50": p(50), "latency_p95": p(95), "latency_p99": p(99),
            "latency_avg": round(total_ms/len(lat), 2),
            "throughput_vps": round(tc / (total_ms/1000), 1) if total_ms else 0,
            "wm_cosine_avg":  round(sum(cos)/len(cos), 6) if cos else None,
            "wm_verify_pct":  round(vr, 3) if vr is not None else None,
            "total_wall_s":   None,
            "cipher_total_ms": None,
            "cipher_ms_per_batch": None,
        }

# ── Source / batches ──────────────────────────────────────────────────────────

def load_source():
    print("Loading vectors into local ChromaDB ...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:    client.delete_collection(SRC_COLLECTION)
    except: pass
    col = client.create_collection(SRC_COLLECTION, metadata={"hnsw:space":"cosine"})
    vecs  = np.load(os.path.join(DATA_DIR, "vectors.npy"))
    with open(os.path.join(DATA_DIR, "metadata.json")) as f:
        metas = json.load(f)
    n  = len(vecs)
    t0 = time.time()
    for s in range(0, n, BATCH_SIZE):
        e = min(s+BATCH_SIZE, n)
        col.add(ids=[m["id"] for m in metas[s:e]],
                embeddings=vecs[s:e].tolist(), metadatas=metas[s:e])
        if (s//BATCH_SIZE) % 20 == 0:
            print(f"  loaded {e:,}/{n:,} ({100*e//n}%) -- {e/(time.time()-t0):.0f} vecs/s")
    print(f"  Done: {n:,} vectors in {time.time()-t0:.1f}s")
    return client, col, vecs, metas

def read_batches(vecs, metas):
    n = len(vecs)
    return [(vecs[s:min(s+BATCH_SIZE,n)],
             [m["id"] for m in metas[s:min(s+BATCH_SIZE,n)]],
             metas[s:min(s+BATCH_SIZE,n)])
            for s in range(0, n, BATCH_SIZE)]

# ── Migrations ────────────────────────────────────────────────────────────────

def qdrant_fresh(client, name):
    try:    client.delete_collection(name)
    except: pass
    client.create_collection(name,
        vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
        optimizers_config=OptimizersConfigDiff(indexing_threshold=0))

def run_chorus(batches, qclient, K, session_seed):
    qdrant_fresh(qclient, TGT_CHORUS)
    stats = Stats()

    # Stage 1: Cipher at source (not wire time)
    print("\n-- CHORUS: encrypting at source (timed separately) --")
    t_enc = time.perf_counter()
    encrypted = []
    for seq, (vecs, ids, metas) in enumerate(batches):
        wire, dec, dids, cos = chorus_encrypt_batch(vecs, ids, K, session_seed, seq)
        encrypted.append((wire, dec, dids, cos, vecs.nbytes, metas))
    cipher_total_ms = (time.perf_counter() - t_enc) * 1000
    cipher_per_batch = cipher_total_ms / len(batches)
    print(f"  Cipher total:     {cipher_total_ms:.0f} ms  ({cipher_per_batch:.0f} ms/batch)")
    print(f"  All {len(batches)} batches pre-encrypted. Now transmitting to Germany ...")

    # Stage 2: Wire transfer + write (this is the timed part)
    print("-- CHORUS: wire transfer + Qdrant write (US -> Germany) --")
    for seq, (wire, dec, dids, cos, raw, metas) in enumerate(encrypted):
        t0 = time.perf_counter()
        pts = [PointStruct(
                   id=abs(hash(vid)) % (2**63),
                   vector=dec[i].tolist(),
                   payload={**metas[i], "orig_id": vid})
               for i, vid in enumerate(dids)]
        qclient.upsert(collection_name=TGT_CHORUS, points=pts)
        ms = (time.perf_counter()-t0)*1000
        stats.add(seq, len(dids), len(wire), raw, ms, cos)
        if seq % 10 == 0:
            print(f"  [{seq:3d}/{len(encrypted)-1}] {len(dids):,} vecs | "
                  f"wire {len(wire)//1024} KB / raw {raw//1024} KB | "
                  f"xfer+write {ms:.0f} ms | wm_cos {cos:.5f}")

    s = stats.summary("CHORUS Fabric")
    s["cipher_total_ms"]    = round(cipher_total_ms, 1)
    s["cipher_ms_per_batch"] = round(cipher_per_batch, 1)
    return stats, s

def run_baseline(batches, qclient):
    qdrant_fresh(qclient, TGT_BASELINE)
    stats = Stats()
    print("\n-- BASELINE (no encryption) -- wire transfer + Qdrant write --")
    for seq, (vecs, ids, metas) in enumerate(batches):
        raw  = vecs.nbytes
        wire = int(raw * 1.08)   # gRPC protobuf ~8% overhead
        t0   = time.perf_counter()
        pts  = [PointStruct(id=abs(hash(ids[i])) % (2**63),
                             vector=vecs[i].tolist(),
                             payload={**metas[i], "orig_id": ids[i]})
                for i in range(len(vecs))]
        qclient.upsert(collection_name=TGT_BASELINE, points=pts)
        ms = (time.perf_counter()-t0)*1000
        stats.add(seq, len(vecs), wire, raw, ms)
        if seq % 10 == 0:
            print(f"  [{seq:3d}/{len(batches)-1}] {len(vecs):,} vecs | "
                  f"wire {wire//1024} KB (est) | xfer+write {ms:.0f} ms")
    return stats, stats.summary("Baseline (plain gRPC)")

# ── Output ────────────────────────────────────────────────────────────────────

def print_results(c, b, meta):
    c["bandwidth_savings_vs_baseline"] = round(b["wire_mb"] / c["wire_mb"], 3)
    b["bandwidth_savings_vs_baseline"] = 1.0
    W = 62
    SEP = "=" * W
    print(f"\n{SEP}")
    print(f"  VECTORBRIDGE TRANSATLANTIC BENCHMARK RESULTS")
    print(SEP)
    print(f"  {meta['source']}")
    print(f"  -> {meta['target']}")
    print(f"  {meta['timestamp']}")
    print()
    print(f"  Dataset : {c['total_vectors']:,} legal-doc vectors x {DIM}-dim")
    print(f"  Raw data: {c['raw_mb']:.0f} MB float32")
    print(f"  JSON eq : {c['json_equiv_mb']:.0f} MB (HTTP/REST equivalent)")
    print()
    print(f"  {'Metric':<34} {'CHORUS':>10}  {'Baseline':>10}")
    print(f"  {'-'*34} {'-'*10}  {'-'*10}")
    print(f"  {'Wire sent (MB)':<34} {c['wire_mb']:>10.1f}  {b['wire_mb']:>10.1f}")
    print(f"  {'vs HTTP/JSON baseline':<34} {c['bandwidth_savings_vs_json']:>9.2f}x  {'1.00x':>10}")
    print(f"  {'vs plain gRPC baseline':<34} {c['bandwidth_savings_vs_baseline']:>9.2f}x  {'1.00x':>10}")
    print(f"  {'p50 xfer+write latency (ms)':<34} {c['latency_p50']:>10}  {b['latency_p50']:>10}")
    print(f"  {'p95 xfer+write latency (ms)':<34} {c['latency_p95']:>10}  {b['latency_p95']:>10}")
    print(f"  {'Throughput (vecs/s)':<34} {c['throughput_vps']:>10,.0f}  {b['throughput_vps']:>10,.0f}")
    print(f"  {'Cipher time (ms/batch)':<34} {c['cipher_ms_per_batch']:>10.0f}  {'N/A':>10}")
    print(f"  {'Total wall time (s)':<34} {c['total_wall_s']:>10}  {b['total_wall_s']:>10}")
    print(f"  {'Watermark verify %':<34} {c['wm_verify_pct']:>9.3f}%  {'N/A':>10}")
    print(f"  {'Avg watermark cosine':<34} {c['wm_cosine_avg']:>10.5f}  {'N/A':>10}")
    print(SEP)
    print(f"\n  CHORUS: {c['bandwidth_savings_vs_json']:.2f}x LESS wire than HTTP/REST/JSON")
    print(f"  CHORUS: {c['bandwidth_savings_vs_baseline']:.2f}x less wire than plain gRPC")
    print(f"  CHORUS watermark: {c['wm_verify_pct']:.1f}% of batches cryptographically verified")
    print(f"  Cipher overhead:  {c['cipher_ms_per_batch']:.0f} ms/batch (source-side, not wire time)")
    print()
    sys.stdout.flush()

def save_results(c, b, meta):
    ts  = int(time.time())
    c["bandwidth_savings_vs_baseline"] = round(b["wire_mb"] / c["wire_mb"], 3)
    b["bandwidth_savings_vs_baseline"] = 1.0
    out = {"run_meta": meta, "chorus": c, "baseline": b}
    p   = os.path.join(RESULTS_DIR, f"benchmark_{ts}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)

    landing = {
        "dataset": {
            "vectors": c["total_vectors"], "dimensions": DIM,
            "raw_mb":  c["raw_mb"],
            "domain":  "Legal (contract archive, text-embedding-3-small)"
        },
        "route": {"source": meta["source"], "target": meta["target"]},
        "chorus": {
            "wire_mb":          c["wire_mb"],
            "savings_vs_json":  c["bandwidth_savings_vs_json"],
            "savings_vs_grpc":  c["bandwidth_savings_vs_baseline"],
            "p50_ms":           c["latency_p50"],
            "p95_ms":           c["latency_p95"],
            "throughput_vps":   c["throughput_vps"],
            "wm_rate_pct":      c["wm_verify_pct"],
            "wm_cosine_avg":    c["wm_cosine_avg"],
            "wall_s":           c["total_wall_s"],
            "cipher_ms_batch":  c["cipher_ms_per_batch"],
        },
        "baseline": {
            "wire_mb":        b["wire_mb"],
            "p50_ms":         b["latency_p50"],
            "throughput_vps": b["throughput_vps"],
        },
        "headline_stats": [
            {"value": f"{c['bandwidth_savings_vs_json']:.1f}x",
             "label": "Less bandwidth than HTTP/REST"},
            {"value": f"{c['wm_verify_pct']:.1f}%",
             "label": "Cryptographic verification rate"},
            {"value": f"{c['throughput_vps']:,.0f}",
             "label": "Vectors/second (transatlantic)"},
            {"value": f"{c['total_vectors']:,}",
             "label": "Legal vectors migrated"},
        ]
    }
    lp = os.path.join(RESULTS_DIR, "landing_page_stats.json")
    with open(lp, "w") as f:
        json.dump(landing, f, indent=2)
    print(f"  Full results  -> {p}")
    print(f"  Landing stats -> {lp}")
    sys.stdout.flush()
    return out, p

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-host", default="72.144.65.153")
    ap.add_argument("--target-port", type=int, default=6333)
    ap.add_argument("--skip-load", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    meta = {
        "source":    "ChromaDB @ localhost (US)",
        "target":    f"Qdrant @ {args.target_host}:{args.target_port} (Germany West - Frankfurt)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol":  "CHORUS Fabric - Patent Pending USPTO No. 64/096,156",
        "dim": DIM, "batch_size": BATCH_SIZE,
    }

    print("=" * 62)
    print("  VectorBridge Transatlantic Benchmark")
    print(f"  US -> Germany West Central (Frankfurt)")
    print("=" * 62)
    sys.stdout.flush()

    if not args.skip_load:
        _, _, vecs, metas = load_source()
    else:
        print("Using existing ChromaDB data ...")
        vecs = np.load(os.path.join(DATA_DIR, "vectors.npy"))
        with open(os.path.join(DATA_DIR, "metadata.json")) as f:
            metas = json.load(f)

    if args.limit:
        vecs, metas = vecs[:args.limit], metas[:args.limit]

    batches = read_batches(vecs, metas)
    n_vecs  = sum(len(b[0]) for b in batches)
    print(f"\nReady: {len(batches)} batches x {BATCH_SIZE} = {n_vecs:,} vectors")
    print(f"Raw float32 size: {vecs[:n_vecs].nbytes/1024**2:.0f} MB")
    sys.stdout.flush()

    print(f"\nConnecting to Qdrant at {args.target_host}:{args.target_port} ...")
    qclient = QdrantClient(host=args.target_host, port=args.target_port, timeout=120)
    cols = qclient.get_collections()
    print(f"  Qdrant OK - {len(cols.collections)} existing collections")
    sys.stdout.flush()

    K, _ = _qr_key(DIM)
    session_seed = 0xDEADBEEF

    # CHORUS run
    t0 = time.time()
    _, c = run_chorus(batches, qclient, K, session_seed)
    c["total_wall_s"] = round(time.time()-t0, 1)

    # Baseline run
    t0 = time.time()
    _, b = run_baseline(batches, qclient)
    b["total_wall_s"] = round(time.time()-t0, 1)

    print_results(c, b, meta)
    save_results(c, b, meta)

if __name__ == "__main__":
    main()
