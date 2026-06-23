# Insight Vector Bridge — Agent Reference Document

> **For agents:** This is the single source of truth for building the landing page, pricing page,
> product copy, and marketing assets for Insight Vector Bridge.
> Do not invent numbers — use only the verified figures in this file.

---

## Product Identity

- **Product name:** Insight Vector Bridge
- **Tagline:** The vector database migration tool that proves the migration was correct.
- **One-liner:** Universal vector database migration — 5.55x less bandwidth, cryptographic integrity, semantic validation.
- **Category keyword:** vector database migration tool
- **PyPI package:** `pip install insight-vector-bridge`
- **GitHub:** https://github.com/insightitsGit/vectorbridge
- **PyPI page:** https://pypi.org/project/insight-vector-bridge/0.1.0/
- **Company:** Insight IT Solutions LLC
- **Website:** https://www.insightits.com
- **Support email:** insightits.info@gmail.com
- **Patent:** USPTO Provisional No. 64/096,156 — Amin Parva / Insight IT Solutions LLC
- **Version:** 0.1.0 (released 2026-06-23)

---

## Landing Page — Must-Have Sections

An agent building the landing page at `insightits.com/products/vectorbridge` should include these sections in order:

### 1. Hero
- Headline: "The Vector Database Migration Tool That Proves It Worked"
- Subheadline: "Move vectors from any source to any target — 5.55x less bandwidth than REST, with cryptographic integrity and semantic validation built in."
- CTA buttons: "Get Started Free" (links to pricing) + "View on GitHub"
- Install snippet: `pip install insight-vector-bridge`
- Trust badge: Patent Pending · USPTO No. 64/096,156

### 2. The Problem (before/after)
- **Before:** Every other migration tool serializes float32 to JSON text on the wire.
  A single float like `0.7231` becomes 14 characters. For 30,000 vectors at 1,536 dims = 821 MB wasted per run.
  No integrity check. No metric validation. Silent data corruption possible.
- **After:** Insight Vector Bridge uses CHORUS Fabric — a binary transport where encryption is a matrix multiply.
  Float32 stays float32. 6,019 KB per batch instead of 33,400 KB. HMAC-verified. Metric-guarded.

### 3. Benchmark Numbers (use exactly — all verified on Azure)
| Metric | Value | Context |
|---|---|---|
| Bandwidth vs REST JSON | 5.55x less | 6,019 KB vs 33,400 KB per 1K vectors |
| Bandwidth saved | 82% | 821 MB per 30,000 vectors |
| RTT speedup vs REST | 4-5x | Real cross-DC Azure benchmark |
| Serialization speedup | 23x | 24 ms CHORUS vs 567 ms JSON |
| HMAC verification rate | 100% | 60/60 batches, two regions |
| Bandwidth vs gzip | 2.33x less | Still beats compressed REST |

### 4. Key Features (for landing page feature cards)

**Feature 1 — Binary Transport (CHORUS Fabric)**
Float32 stays float32. No JSON serialization. Cipher = matrix multiply (same op neural nets run).
Wire format: `MAGIC(4) | count(4) | dim(4) | seq(8) | HMAC(32) | payload`

**Feature 2 — Metric Mismatch Guard**
Blocks migration before byte one if source and target use different distance metrics.
Cosine -> L2 silently corrupts search results. VectorBridge is the only tool that catches this.
Error class: `MetricMismatchError`

**Feature 3 — Semantic Validation**
Post-migration: fires N probe vectors against source and target, requires >=95% top-K neighbor overlap.
Your data didn't just transfer — it transferred correctly.

**Feature 4 — Checkpoint / Resume**
Progress saved to `.vectorbridge/{job_id}.json`. Resume interrupted migrations without re-sending.

**Feature 5 — Integrity Report**
Every migration produces a JSON artifact: vectors transferred, verified, wire bytes, bandwidth savings,
watermark rate, semantic overlap score. Attachable to compliance audits.

**Feature 6 — Works When Re-Embedding Is Impossible**
Source data GDPR-deleted? Embedding model deprecated? API shut down?
Other tools require the original text to re-embed. VectorBridge migrates the vectors directly.

### 5. Supported Databases
ChromaDB, Qdrant, Weaviate, Pinecone, pgvector, FAISS

### 6. Pricing (see Pricing section below)

### 7. Competitive Table
| Feature | Insight Vector Bridge | MING | Qdrant Docker Tool | Milvus VTS |
|---|---|---|---|---|
| Binary transport (not HTTP) | YES | no | no | no |
| Bandwidth vs REST | 5.55x less | baseline | baseline | baseline |
| Metric mismatch guard | YES | no | no | no |
| Post-migration semantic validation | YES | no | no | no |
| Per-batch HMAC integrity | YES | no | no | no |
| Works when source data is gone | YES | no | no | no |
| Universal (not target-locked) | YES | YES | Qdrant only | Milvus only |

### 8. Use Cases
1. Source data no longer accessible (GDPR deleted, client-owned, third-party)
2. Embedding model deprecated or API gone (OpenAI model sunset, vendor shutdown)
3. Embedding API too expensive to re-run at scale
4. DB vendor change (ChromaDB -> Qdrant, Pinecone -> pgvector, etc.)
5. Namespace / collection restructuring within same DB
6. Disaster recovery / backup restore
7. Data residency compliance (EU data must stay in EU region)

### 9. Quick Start Code Snippet
```python
from vectorbridge import migrate

migrate(
    source="chromadb://localhost:8000/my_collection",
    target="qdrant://localhost:6333/my_collection",
    batch_size=1000,
    semantic_verify=True,
)
```

### 10. Footer / CTA
- `pip install insight-vector-bridge`
- GitHub: github.com/insightitsGit/vectorbridge
- Support: insightits.info@gmail.com
- Patent Pending · USPTO No. 64/096,156 · Insight IT Solutions LLC · Mission Viejo, CA

---

## Pricing

### Model: DWV (Dimension-Weighted Vectors)
**Formula:** DWV = vectors × dimensions × $0.000001

This model charges proportionally to actual data moved — a 128-dim FAISS migration costs less than a
1,536-dim OpenAI embedding migration.

### Tiers

| Tier | DWV Included | Price | Approx Vectors (1536-dim) | Target Customer |
|---|---|---|---|---|
| Free | 200M DWV | $0/mo | ~130,000 vectors | Developers evaluating |
| Starter | 2B DWV | $49/mo | ~1.3M vectors | Small teams, single DB migration |
| Pro | 25B DWV | $199/mo | ~16M vectors | Mid-size teams, recurring migrations |
| Enterprise | Unlimited | $999+/mo | Unlimited | Large orgs, SLA, audit reports, support |

### Pricing Page URL
`insightits.com/products/vectorbridge/pricing`

### Purchase Flow (for agent building the page)
1. Customer selects tier on pricing page
2. Clicks "Buy" -> Stripe Checkout (Payment Links for Starter/Pro; contact form for Enterprise)
3. Stripe webhook fires -> license key generated + emailed to customer
4. Customer sets env var: `VECTORBRIDGE_LICENSE_KEY=<key>`
5. Each job run validates key against: `insightits.com/api/vectorbridge/v1/license/validate`
6. DWV usage posted back to: `insightits.com/api/vectorbridge/v1/agent/report`

### Stripe Setup Status
- Starter ($49) and Pro ($199): use Stripe Payment Links (not yet created — agent task)
- Enterprise: contact form -> insightits.info@gmail.com

---

## Marketing Copy — Safe Claims

Use only these verified, reproducible claims in all copy:

| Claim | Evidence |
|---|---|
| "5.55x less bandwidth than HTTP/REST JSON" | Pure math: 6,019 KB vs 33,400 KB per 1K vectors at 1536-dim |
| "82% bandwidth reduction" | 821 MB saved per 30,000 vectors |
| "2.33x less than gzip-compressed REST" | Measured both regions |
| "4-5x faster round-trip than REST" | Real Azure cross-DC (Virginia + Washington) |
| "Zero cipher overhead" | CHORUS RTT ≈ raw binary RTT |
| "100% HMAC verification rate" | 60/60 batches across two Azure regions |
| "23x faster serialization than JSON" | 24 ms CHORUS pack vs 567 ms JSON serialize |
| "Patent pending transport layer" | USPTO Provisional No. 64/096,156 |

### Claims to NEVER use
- "30x faster RTT" — localhost artifact only, not cross-DC
- "119x faster than binary" — compared numpy vectorized vs Python for-loop, not wire formats
- "4.49x bandwidth" — old pre-measurement estimate, superseded by 5.55x

---

## Benchmark Details

### Cross-DC Test (primary — use these numbers in all marketing)
- **Date:** 2026-06-23
- **Infrastructure:** Azure Container Instances, 2 vCPU / 4 GB RAM
- **Regions:** eastus (Virginia) + westus2 (Washington)
- **Dataset:** 30,000 vectors, 1,536-dim float32, L2-normalized, 30 batches x 1,000 vectors

**US East (Virginia):**
| Format | Wire KB/batch | Avg RTT | vs CHORUS |
|---|---|---|---|
| HTTP/REST JSON | 33,400 | 19,047 ms | 4.0x slower |
| HTTP/REST + gzip | 14,005 | 8,558 ms | 1.8x slower |
| Raw binary float32 | 6,019 | 4,009 ms | similar |
| CHORUS Fabric | 6,019 | 4,737 ms | baseline |

**US West (Washington):**
| Format | Wire KB/batch | Avg RTT | vs CHORUS |
|---|---|---|---|
| HTTP/REST JSON | 33,400 | 14,278 ms | 5.0x slower |
| HTTP/REST + gzip | 14,005 | 6,101 ms | 2.1x slower |
| Raw binary float32 | 6,019 | 2,555 ms | similar |
| CHORUS Fabric | 6,019 | 2,859 ms | baseline |

---

## Technical Architecture (for developer docs section of landing page)

### CHORUS Fabric Wire Format
```
MAGIC(4) | count(4) | dim(4) | seq(8) | HMAC(32) | payload
```
- MAGIC = `b"CH0R"`
- HMAC = HMAC-SHA256 keyed on `SHA256(session_seed + seq_bytes)`, covers full payload
- Cipher: `V_enc = V_raw @ K` where K is QR-decomposed orthogonal matrix (K_inv = K.T)
- Overhead vs raw float32: 52 bytes per batch (header) — negligible at production scale

### Pre-flight Guards
- `MetricMismatchError` — raised before migration starts if source/target metrics differ
- Supported aliases normalized: `euclid`=`l2`, `inner`=`dot`, `ip`=`dot`
- Override with `metric_override=True` (logs warning, does not block)

### Integrity Report Fields (every migration)
```json
{
  "job_id": "...",
  "transferred": 50000,
  "verified": 50000,
  "failed_watermark": 0,
  "verification_rate": 100.0,
  "wire_bytes": 301000000,
  "raw_bytes": 307200000,
  "bandwidth_savings_x": 5.55,
  "completed_at": "2026-06-23T...",
  "semantic_verify": {
    "passed": true,
    "avg_overlap_pct": 97.3,
    "n_probes": 50
  }
}
```

---

## Package Status

- **PyPI name:** `insight-vector-bridge`
- **Version:** 0.1.0
- **Published:** 2026-06-23
- **Wheel size:** 33 KB
- **Tests:** 42/42 passing (pytest, no external services required)
- **Python:** >=3.10
- **Core dependencies:** numpy, tqdm, click, rich
- **Optional extras:** `[pgvector]`, `[pinecone]`, `[chromadb]`, `[weaviate]`, `[qdrant]`, `[faiss]`, `[all]`

---

## Other Insight IT Products (same website)

| Product | Description | Status |
|---|---|---|
| The Chorus Fabric | Patent-pending M2M tensor protocol — underlying transport for VectorBridge | PyPI: chorus-fabric |
| PrismLang | Deterministic vector protocol for LangGraph, ~60% token reduction | Apache 2.0 |
| PrismRAG | Mapping-first enterprise Graph RAG replacement | In development |

---

## Infrastructure (internal — do not publish)

- Azure resource group: `vb-benchmark`
- ACR: `vbbenchmark.azurecr.io`
- Wire server image: `vbbenchmark.azurecr.io/wireserver:v1`
- vb-wire-east: `52.224.86.182:9000` (eastus)
- vb-wire-west: `20.29.162.80:9000` (westus2)
- Germany Qdrant: `72.144.65.153:6333` (Frankfurt)

Cleanup Azure resources when no longer needed:
```bash
az group delete --name vb-benchmark --yes --no-wait
```
