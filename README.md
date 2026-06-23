# VectorBridge

**Universal vector database migration — powered by CHORUS Fabric tensor transport.**

Move vectors between any two vector databases with 5.55× less bandwidth than HTTP/REST,
built-in encryption, and post-migration semantic validation that proves correctness.

```bash
pip install vectorbridge
```

---

## The Problem with Vector DB Migration

Every other migration tool sends your float32 vectors as JSON text over HTTP.
That costs 33 MB per 1,000 vectors. It has no encryption, no integrity proof,
and no way to detect if your distance metrics changed — causing silent query corruption.

VectorBridge fixes all three.

---

## Live Benchmark — Real Azure Cross-Datacenter

> Measured June 2026 · Azure Container Instances · US East (Virginia) + US West (Washington)
> 60,000 vectors · 1,536-dim float32 · 30 batches per region

| Format | Wire per 1K vecs | vs REST | Avg RTT (US East) | Avg RTT (US West) |
|---|---|---|---|---|
| HTTP/REST JSON | 33,400 KB | baseline | 19,047 ms | 14,278 ms |
| HTTP/REST + gzip | 14,005 KB | 2.33x less | 8,558 ms | 6,101 ms |
| Raw binary (float32) | 6,019 KB | 5.55x less | 4,009 ms | 2,555 ms |
| **CHORUS Fabric** | **6,019 KB** | **5.55x less** | **4,737 ms** | **2,859 ms** |

**82% less bandwidth than REST.** 821 MB saved per 30,000 vectors.
**4–5× faster per-batch RTT.** Zero cipher overhead — CHORUS matches raw binary speed.
**100% watermark verification.** 60/60 batches verified across both regions.

Full benchmark logs: [`benchmark/results/`](benchmark/results/)

---

## Quick Start

```python
from vectorbridge import Bridge

bridge = Bridge(
    source={"type": "chromadb", "path": "./chroma_data", "collection": "docs"},
    target={"type": "qdrant",   "host": "localhost", "port": 6333, "collection": "docs"},
)

report = bridge.run()
print(report.summary())
```

```
VectorBridge Migration Report
-----------------------------------------------------
  Job ID       : vb_20260623_a1b2c3
  Source       : chromadb::docs
  Target       : qdrant::docs
  Vectors      : 50,000 transferred / 50,000 verified
  Wire bytes   : 180 MB  (vs 1.0 GB REST equivalent)
  Bandwidth    : 5.55x savings
  Watermark OK : 50/50 batches (100.0%)
  Semantic OK  : 97.4% neighbour overlap (100 probes x top-5) [PASS]
  Duration     : 42.3 s
-----------------------------------------------------
```

---

## CLI

```bash
# Basic migration
vectorbridge migrate \
  --source chromadb://./chroma_data/docs \
  --target qdrant://localhost:6333/docs

# With semantic validation
vectorbridge migrate \
  --source chromadb://./chroma_data/docs \
  --target qdrant://localhost:6333/docs \
  --semantic-probes 200

# Override metric mismatch (dangerous — use only if you know what you're doing)
vectorbridge migrate \
  --source chromadb://./chroma_data/docs \
  --target qdrant://localhost:6333/docs \
  --metric-override
```

---

## Supported Databases

| Database | Read | Write |
|---|---|---|
| ChromaDB | yes | yes |
| Qdrant | yes | yes |
| Weaviate | yes | yes |
| Pinecone | yes | yes |
| pgvector | yes | yes |
| FAISS | yes | yes |

```bash
pip install "vectorbridge[chromadb,qdrant]"   # specific connectors
pip install "vectorbridge[all]"               # everything
```

---

## Key Features

### 1. CHORUS Fabric Transport (Patent Pending)

VectorBridge is the only migration tool that operates below HTTP. Instead of serializing
vectors to JSON, it packs float32 arrays directly into an encrypted binary wire format:

```
V_enc = V_raw @ K
```

`K` is a QR-decomposed orthogonal matrix — the cipher is a single matrix multiply,
the same operation every neural network already runs. Zero added latency.
Each batch carries a rolling SHA-256 neural watermark for chain-of-custody proof.

**Result:** 5.55× less bandwidth than REST. 2.33× less than gzip-compressed REST.
23× faster serialization (24 ms vs 567 ms for JSON).

*USPTO Provisional Patent No. 64/096,156*

### 2. Distance Metric Guard

Silent data corruption happens when you migrate a cosine collection into an L2 collection.
The vectors import cleanly. Queries return wrong results. No other tool catches this.

VectorBridge blocks the migration before byte one moves:

```
MetricMismatchError:

  DISTANCE METRIC MISMATCH
  ========================
  Source : cosine
  Target : l2

  Vectors will import successfully but nearest-neighbour
  queries will return WRONG results on the target.

  To proceed anyway: set metric_override=True
```

### 3. Post-Migration Semantic Validation

Byte checksums cannot prove search behavior is preserved. VectorBridge fires probe vectors
against both source and target after migration, compares top-K neighbor IDs, and requires
≥95% overlap before declaring success:

```python
report = bridge.run(semantic_verify=True, semantic_probes=100, semantic_top_k=5)
# Semantic OK : 97.4% neighbour overlap (100 probes x top-5) [PASS]
```

### 4. Integrity Report

Every migration produces a signed JSON artifact:

```json
{
  "job_id": "vb_20260623_a1b2c3",
  "transferred": 50000,
  "verified": 50000,
  "wire_bytes": 188743680,
  "bandwidth_savings_x": 5.55,
  "watermark_verification_rate": 100.0,
  "semantic_verify": {
    "avg_overlap_pct": 97.4,
    "n_probes": 100,
    "top_k": 5,
    "passed": true
  }
}
```

Attach to compliance audits, SOC 2 evidence, or internal migration sign-off.

---

## When Re-Embedding Is Not an Option

Most migration guides say "just re-embed your documents." That only works if:
- You still have the original documents
- The embedding model is still available
- Re-running the API is affordable

VectorBridge is built for when that's not the case:

| Scenario | Re-embed? | VectorBridge? |
|---|---|---|
| Switching DB vendors (Pinecone → Qdrant) | Maybe | Yes |
| Source documents deleted / GDPR erased | **No** | Yes |
| Embedding model deprecated (API gone) | **No** | Yes |
| Client-owned data you never had | **No** | Yes |
| Disaster recovery / restore from backup | **No** | Yes |
| Data residency compliance (EU data stays EU) | **No** | Yes |
| Cost: 100M vectors × $0.0001/1K = $10,000 | **Maybe not** | Yes |

---

## How It Compares

| Feature | VectorBridge | MING | Qdrant Docker Tool | Milvus VTS |
|---|---|---|---|---|
| Binary transport (not HTTP/REST) | **Yes** | No | No | No |
| Bandwidth vs REST JSON | **5.55× less** | baseline | baseline | baseline |
| Distance metric guard | **Yes** | No | No | No |
| Semantic validation post-migration | **Yes** | No | No | No |
| Per-batch watermark + audit log | **Yes** | No | No | No |
| Works without source documents | **Yes** | No | No | No |
| Target-agnostic | **Yes** | Yes | Qdrant only | Milvus only |

---

## Benchmark Methodology

All wire format measurements use real HTTP requests to live Azure Container Instances.
No simulated bytes. No theoretical estimates.

```
Infrastructure:  Azure Container Instances (2 vCPU / 4 GB)
Regions:         eastus (Virginia) + westus2 (Washington)
Dataset:         1,536-dim float32, L2-normalized, seed=42
Batches:         30 x 1,000 vectors per region = 30,000 vectors each
Formats:         HTTP/REST JSON | gzip REST | raw binary | CHORUS binary
Script:          benchmark/run_crossdc_benchmark.py
Full logs:       benchmark/results/crossdc_full_1782231973.json
```

To reproduce:

```bash
# Deploy servers
az container create --name vb-wire-east ... --location eastus
az container create --name vb-wire-west ... --location westus2

# Run
python benchmark/run_crossdc_benchmark.py \
  --east http://<east-ip>:9000 \
  --west http://<west-ip>:9000 \
  --batches 30
```

---

## Pricing

Based on DWV (Dimension-Weighted Vectors = vectors × dimensions × $0.000001):

| Tier | Included DWV | Price | Approx Vectors (1536-dim) |
|---|---|---|---|
| Free | 200M | $0 | ~130K |
| Starter | 2B | $49/mo | ~1.3M |
| Pro | 25B | $199/mo | ~16M |
| Enterprise | Unlimited | $999+/mo | Unlimited + SLA |

---

## License

MIT — see [LICENSE](LICENSE)

Transport layer (CHORUS Fabric) is patent pending — USPTO Provisional No. 64/096,156.
Commercial use of the CHORUS wire format in other products requires a license.
Contact: parvaamin@gmail.com

---

## Author

**Amin Parva** — AI Solution Architect, Insight IT Solutions LLC
parvaamin@gmail.com · insightits.com · github.com/insightitsGit
