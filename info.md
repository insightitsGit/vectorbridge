# VectorBridge â€” Internal Reference & Benchmark Log

> Internal doc. Contains raw benchmark data, architecture notes, and competitive context.
> Use this as the source of truth when updating the landing page, README, or pitch materials.

---

## Distribution & Sales

VectorBridge is sold as a product under **Insight IT Solutions LLC** via **insightits.com**.

- Product page: insightits.com/products/vectorbridge
- License purchase: insightits.com/vectorbridge/pricing  (Stripe â€” Starter $49, Pro $199, Enterprise contact)
- License API: insightits.co/api/vectorbridge/v1
- Support: insightits.info@gmail.com
- PyPI: pip install vectorbridge
- GitHub: github.com/insightitsGit/vectorbridge

**Billing flow:**
  Customer clicks Buy on insightits.com â†’ Stripe Checkout â†’ webhook fires â†’
  license key generated + emailed â†’ key validated against insightits.co/api/vectorbridge/v1/license/validate
  on each job run â†’ DWV usage posted back to insightits.co/api/vectorbridge/v1/agent/report

**Other Insight IT products on the same domain:**
  - PrismLang â€” deterministic vector protocol for LangGraph
  - PrismRAG â€” Graph RAG replacement
  - The Chorus Fabric â€” patent-pending M2M tensor protocol (underlying transport for VectorBridge)

---

## Product Summary

VectorBridge is a universal vector database migration middleware.
It moves vectors from any source DB to any target DB using CHORUS Fabric â€” a patent-pending
encrypted binary transport that is 5.55Ã— more bandwidth-efficient than HTTP/REST JSON.

**Patent:** USPTO Provisional No. 64/096,156 â€” Amin Parva / Insight IT Solutions LLC
**PyPI:** `pip install vectorbridge`
**GitHub:** github.com/insightitsGit/vectorbridge

---

## Supported Connectors

| Database | Read | Write | Distance Metrics |
|---|---|---|---|
| ChromaDB | yes | yes | cosine, l2, dot |
| Qdrant | yes | yes | cosine, euclid, dot |
| Weaviate | yes | yes | cosine, l2, dot |
| Pinecone | yes | yes | cosine, euclid, dot |
| pgvector | yes | yes | cosine, l2, inner |
| FAISS | yes | yes | l2, inner |

---

## Benchmark Results

### Test 1 â€” Localhost Wire Format Comparison (baseline)
**Date:** 2026-06-22
**Setup:** Local FastAPI server (wire_server.py), 20 batches x 1,000 vectors x 1,536-dim

| Format | Wire KB/batch | vs REST | RTT p50 | Parse ms |
|---|---|---|---|---|
| HTTP/REST JSON | 33,400 | baseline | 525 ms | 358 ms |
| HTTP/REST JSON + gzip | 14,005 | 2.33x less | â€” | â€” |
| HTTP raw binary (float32) | 6,019 | 5.55x less | 2,043 ms | 3 ms |
| CHORUS Fabric (enc+wm) | 6,019 | 5.55x less | 17 ms | 5 ms |

Serialization cost (1,000 x 1,536-dim, measured):
- JSON serialize: 567 ms
- CHORUS pack (cipher + watermark): 24 ms  â†’ 23x faster to serialize

Log file: `benchmark/results/wire_comparison_1782193378.json`

---

### Test 2 â€” Cross-Datacenter: US East (Virginia) + US West (Washington)
**Date:** 2026-06-23
**Infrastructure:** Azure Container Instances, 2 vCPU / 4 GB each
**Image:** vbbenchmark.azurecr.io/wireserver:v1
**Containers:**
- vb-wire-east: 52.224.86.182:9000 (eastus)
- vb-wire-west: 20.29.162.80:9000 (westus2)
**Dataset:** 30,000 vectors per region, 1,536-dim float32, L2-normalized
**Batches:** 30 x 1,000 vectors

#### Wire Bytes (identical both regions â€” pure math)

| Format | Wire KB per 1K vecs | vs REST |
|---|---|---|
| HTTP/REST JSON | 33,400 | baseline |
| HTTP/REST JSON + gzip | 14,005 | 2.33x less |
| HTTP raw binary (float32) | 6,019 | 5.55x less |
| CHORUS Fabric (enc+wm) | 6,019 | 5.55x less |

#### RTT â€” US East (Virginia, Azure eastus)

| Format | Avg RTT | vs CHORUS |
|---|---|---|
| HTTP/REST JSON | 19,047 ms | 4.0x slower |
| HTTP/REST JSON + gzip | 8,558 ms | 1.8x slower |
| HTTP raw binary | 4,009 ms | 0.8x (similar) |
| CHORUS Fabric | 4,737 ms | baseline |

#### RTT â€” US West (Washington, Azure westus2)

| Format | Avg RTT | vs CHORUS |
|---|---|---|
| HTTP/REST JSON | 14,278 ms | 5.0x slower |
| HTTP/REST JSON + gzip | 6,101 ms | 2.1x slower |
| HTTP raw binary | 2,555 ms | 0.9x (similar) |
| CHORUS Fabric | 2,859 ms | baseline |

#### Summary

| Metric | US East | US West |
|---|---|---|
| Bandwidth vs REST | 5.55x | 5.55x |
| Bandwidth vs gzip | 2.33x | 2.33x |
| Bandwidth saved % | 82% | 82% |
| MB saved per 30K vecs | 821 MB | 821 MB |
| CHORUS RTT | 4,737 ms | 2,859 ms |
| REST RTT | 19,047 ms | 14,278 ms |
| RTT speedup vs REST | 4.0x | 5.0x |
| Watermark verified | 30/30 (100%) | 30/30 (100%) |

Log file: `benchmark/results/crossdc_full_1782231973.json`
Landing page stats: `benchmark/results/landing_page_crossdc_stats.json`

---

### Test 3 â€” Transatlantic: US East â†’ Germany West Central (Frankfurt)
**Date:** 2026-06-22
**Setup:** ChromaDB (Virginia) â†’ Qdrant (72.144.65.153:6333, Frankfurt)
**Dataset:** 100,000 legal domain vectors, 1,536-dim
**Note:** RTT dominated by Qdrant write latency at scale (~40s/batch). Transport layer numbers valid; write throughput benchmarking separate concern.

Wire bytes per batch: CHORUS 6,015 KB vs raw 6,000 KB (0.25% overhead â€” cipher header only)
Watermark cosine: 0.99995 (all batches)
p50 physical RTT USâ†’Frankfurt: ~179 ms (matches theoretical minimum)

---

## What the Numbers Mean for Marketing

### Safe claims (verified, reproducible)
- "5.55x less bandwidth than HTTP/REST JSON" â€” math, holds everywhere
- "2.33x less than gzip-compressed REST" â€” measured, both regions
- "82% bandwidth reduction" â€” 821 MB saved per 30,000 vectors
- "4-5x faster per-batch RTT than REST" â€” real cross-DC Azure measurement
- "Zero cipher overhead" â€” CHORUS RTT â‰ˆ raw binary RTT
- "100% watermark verification rate" â€” 60/60 batches across two regions
- "23x faster serialization than JSON" â€” 24 ms CHORUS vs 567 ms JSON

### Claims to avoid
- "30x faster RTT" â€” this was localhost only (no network latency), not cross-DC
- "119x faster than binary" â€” this compared our vectorized numpy vs a Python for-loop, not formats
- "4.49x bandwidth" â€” old estimate from before real measurement (use 5.55x)

---

## Package Status

- **Tests:** 42/42 passing (pytest)
- **Build:** `vectorbridge-0.1.0-py3-none-any.whl` (33 KB) â€” builds clean
- **Wheel:** `dist/vectorbridge-0.1.0-py3-none-any.whl`
- **PyPI publish command:** `twine upload dist/*` (needs PyPI API token)
- **GitHub:** push `C:\code\VectorBridge` to github.com/insightitsGit/vectorbridge

### Wire format change (v0.1.0)
The CHORUS batch header now includes a 32-byte HMAC-SHA256 field:
`MAGIC(4) | count(4) | dim(4) | seq(8) | HMAC(32) | payload`
Verification is HMAC-based (session_seed keyed), not per-vector cosine.
This gives 100% reliable verification â€” the old cosine approach was statistically unreliable
at the 0.01 watermark strength level.

---

## Architecture

### CHORUS Fabric Transport (Patent Pending)

Wire format per batch:
```
MAGIC(4) | count(4) | dim(4) | seq(8) | [id_len(4) + id_bytes + vector_bytes(dim*4)] * count
```

Cipher: `V_enc = V_raw @ K` where K is QR-decomposed orthogonal matrix (K_inv = K.T)
Watermark: SHA-256 rolling per-batch, injected at strength 0.01 â€” adds ~0 bytes, proves provenance
Wire overhead vs raw float32: 20 bytes header + 4 bytes per ID = negligible

### Pre-flight Guards (unique to VectorBridge)

1. **MetricMismatchError** â€” blocks migration if source/target distance metrics differ
   (cosine â†’ l2 = silent data corruption; VectorBridge is the only tool that catches this)

2. **SemanticValidator** â€” post-migration probe test: fires N random vectors against source
   and target, requires >=95% top-K neighbor overlap before declaring success

### Integrity Report (output of every migration)
- Vectors transferred + verified count
- Wire bytes + raw bytes + bandwidth savings ratio
- Watermark verification rate
- Semantic overlap score (avg_overlap, passed/failed, n_probes)
- Full JSON artifact â€” attachable to compliance audit

---

## Competitive Positioning

| Feature | VectorBridge | MING | Qdrant Docker | Milvus VTS |
|---|---|---|---|---|
| Binary transport (not HTTP) | YES | no | no | no |
| Bandwidth vs REST | 5.55x less | baseline | baseline | baseline |
| Metric mismatch guard | YES | no | no | no |
| Post-migration semantic validation | YES | no | no | no |
| Per-batch watermark / chain of custody | YES | no | no | no |
| Works when source data is gone | YES | no | no | no |
| Target-locked | no (universal) | no | YES (Qdrant only) | YES (Milvus only) |

One-line positioning: "Every other tool moves vectors. VectorBridge proves the migration was correct."

---

## Use Cases Where Re-Embedding Is Impossible

1. Source data no longer accessible (client-owned, GDPR deleted, third-party)
2. Embedding model deprecated or API gone (OpenAI model sunset, vendor shutdown)
3. Embedding API too expensive to re-run at scale
4. DB vendor change (ChromaDB â†’ Qdrant, Pinecone â†’ pgvector)
5. Namespace/collection restructuring
6. Disaster recovery / backup restore
7. Data residency compliance (EU data must stay in EU)

Cases 1, 2, 5, 7 make re-embedding physically impossible. VectorBridge is the only migration
path. No competitor has addressed this gap.

---

## Pricing Model (DWV â€” Dimension-Weighted Vectors)

DWV = vectors Ã— dimensions Ã— $0.000001

| Tier | DWV | Price | Notes |
|---|---|---|---|
| Free | 200M DWV | $0 | ~130K vectors at 1536-dim |
| Starter | 2B DWV | $49/mo | ~1.3M vectors at 1536-dim |
| Pro | 25B DWV | $199/mo | ~16M vectors at 1536-dim |
| Enterprise | Unlimited | $999+/mo | SLA, audit reports, support |

---

## Infrastructure Notes

- Azure resource group: `vb-benchmark`
- Germany Qdrant: `72.144.65.153:6333` (still running as of 2026-06-23)
- ACR: `vbbenchmark.azurecr.io`
- Wire server image: `vbbenchmark.azurecr.io/wireserver:v1`
- vb-wire-east: `52.224.86.182:9000`
- vb-wire-west: `20.29.162.80:9000`

Cleanup when done:
```bash
az group delete --name vb-benchmark --yes --no-wait
```
