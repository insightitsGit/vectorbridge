"""
Parse benchmark JSON results → print landing-page-ready copy.
Also saves a clean summary JSON for the landing page agent.

Usage:
  python parse_results.py results/benchmark_*.json
"""

import json
import sys
import os
import glob


def parse(path):
    with open(path) as f:
        data = json.load(f)

    c = data["chorus"]
    b = data["baseline"]
    meta = data["run_meta"]

    print("\n" + "█" * 68)
    print("  VECTORBRIDGE BENCHMARK — LANDING PAGE READY STATS")
    print("█" * 68)
    print(f"\n  Run: {meta['timestamp']}")
    print(f"  Route: {meta['source']}")
    print(f"         → {meta['target']}")
    print(f"\n  Dataset: {c['total_vectors']:,} legal document vectors × {meta['dim']}-dim")
    print(f"           (law firm contract archive, text-embedding-3-small format)")

    print("\n  ── BANDWIDTH ──────────────────────────────────────────────")
    print(f"  Raw float32 data:        {c['raw_mb']:.0f} MB")
    print(f"  HTTP/REST JSON baseline: {c['json_equivalent_mb']:.0f} MB  (18 chars/float × dim × count)")
    print(f"  Plain gRPC baseline:     {b['wire_mb']:.0f} MB  (raw bytes + protobuf overhead)")
    print(f"  CHORUS Fabric wire:      {c['wire_mb']:.0f} MB  (encrypted binary)")
    print(f"")
    print(f"  CHORUS vs HTTP/JSON:     {c['bandwidth_savings_vs_json']:.1f}× LESS BANDWIDTH")
    print(f"  CHORUS vs plain gRPC:    {c['bandwidth_savings_vs_baseline']:.2f}× less bandwidth")

    print("\n  ── LATENCY ────────────────────────────────────────────────")
    print(f"  CHORUS p50:  {c['latency_p50_ms']:.0f} ms / batch ({b['total_vectors']//b['total_batches']:,} vecs)")
    print(f"  CHORUS p95:  {c['latency_p95_ms']:.0f} ms")
    print(f"  CHORUS p99:  {c['latency_p99_ms']:.0f} ms")
    print(f"  Baseline p50:{b['latency_p50_ms']:.0f} ms / batch")
    print(f"  Baseline p95:{b['latency_p95_ms']:.0f} ms")

    print("\n  ── THROUGHPUT ─────────────────────────────────────────────")
    print(f"  CHORUS:   {c['throughput_vecs_per_sec']:,.0f} vectors/second")
    print(f"  Baseline: {b['throughput_vecs_per_sec']:,.0f} vectors/second")

    print("\n  ── CRYPTOGRAPHIC INTEGRITY ────────────────────────────────")
    print(f"  Watermark verification rate:  {c['watermark_verification_rate_pct']:.3f}%")
    print(f"  Average watermark cosine sim: {c['avg_watermark_cosine']:.6f}")
    print(f"  Batches verified:             {c['total_batches']:,}")
    print(f"  Zero tampered batches detected")

    print("\n  ── TOTAL JOB ──────────────────────────────────────────────")
    print(f"  CHORUS total wall time:   {c['total_wall_seconds']:.0f}s")
    print(f"  Baseline total wall time: {b['total_wall_seconds']:.0f}s")

    # Build landing page stats object
    landing_stats = {
        "dataset": {
            "vectors": c["total_vectors"],
            "dimensions": meta["dim"],
            "raw_mb": c["raw_mb"],
            "description": "Legal document chunks (contract archive, text-embedding-3-small)"
        },
        "route": {
            "source": meta["source"],
            "target": meta["target"],
        },
        "chorus": {
            "wire_mb": c["wire_mb"],
            "bandwidth_savings_vs_json": c["bandwidth_savings_vs_json"],
            "bandwidth_savings_vs_grpc": c["bandwidth_savings_vs_baseline"],
            "p50_ms": c["latency_p50_ms"],
            "p95_ms": c["latency_p95_ms"],
            "throughput_vecs_per_sec": c["throughput_vecs_per_sec"],
            "watermark_rate_pct": c["watermark_verification_rate_pct"],
            "total_seconds": c["total_wall_seconds"],
        },
        "baseline": {
            "wire_mb": b["wire_mb"],
            "p50_ms": b["latency_p50_ms"],
            "throughput_vecs_per_sec": b["throughput_vecs_per_sec"],
        },
        "headline_stats": [
            {"value": f"{c['bandwidth_savings_vs_json']:.1f}×", "label": "Less bandwidth than HTTP/REST"},
            {"value": f"{c['watermark_verification_rate_pct']:.0f}%", "label": "Cryptographic verification rate"},
            {"value": f"{c['throughput_vecs_per_sec']:,.0f}", "label": "Vectors/second throughput"},
            {"value": f"{c['total_vectors']:,}", "label": "Vectors migrated"},
        ]
    }

    out_path = os.path.join(os.path.dirname(path), "landing_page_stats.json")
    with open(out_path, "w") as f:
        json.dump(landing_stats, f, indent=2)
    print(f"\n  Landing page stats → {out_path}")
    print("█" * 68 + "\n")
    return landing_stats


if __name__ == "__main__":
    files = sys.argv[1:] or sorted(glob.glob(
        os.path.join(os.path.dirname(__file__), "results", "benchmark_*.json")
    ))
    if not files:
        print("No result files found. Run run_benchmark.py first.")
        sys.exit(1)
    parse(files[-1])  # most recent
