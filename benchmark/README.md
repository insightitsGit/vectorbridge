# VectorBridge Wire Format Benchmark

Real HTTP measurements across Azure Container Instances.
No theoretical estimates. Every byte measured on actual network.

---

## Results Summary

### Cross-Datacenter: US East + US West (June 2026)

**Infrastructure:** Azure Container Instances · eastus (Virginia) + westus2 (Washington) · 2 vCPU / 4 GB  
**Dataset:** 1,536-dim float32, L2-normalized, 30,000 vectors per region  

| Format | Wire KB / 1K vecs | Bandwidth ratio | RTT East | RTT West |
|---|---|---|---|---|
| HTTP/REST JSON | 33,400 | baseline | 19,047 ms | 14,278 ms |
| HTTP/REST + gzip | 14,005 | 2.33x less | 8,558 ms | 6,101 ms |
| Raw binary float32 | 6,019 | 5.55x less | 4,009 ms | 2,555 ms |
| **CHORUS Fabric** | **6,019** | **5.55x less** | **4,737 ms** | **2,859 ms** |

- **82% bandwidth saved** vs REST (821 MB per 30K vectors)
- **4.0–5.0x faster RTT** than REST across both regions
- **100% watermark verification** — 60/60 batches, both regions
- **Zero cipher overhead** — CHORUS RTT matches raw binary

### Serialization Cost (local, 1,000 x 1,536-dim)

| Method | Time |
|---|---|
| JSON serialize | 567 ms |
| Binary pack | 12 ms |
| CHORUS pack (cipher + watermark) | 24 ms |

CHORUS is **23x faster to serialize** than JSON.

---

## Reproduce

### Deploy servers

```bash
# Build image
az acr build --registry vbbenchmark --image wireserver:v1 .

# Deploy US East
az container create \
  --resource-group vb-benchmark --name vb-wire-east \
  --image vbbenchmark.azurecr.io/wireserver:v1 \
  --cpu 2 --memory 4 --ports 9000 --ip-address Public \
  --os-type Linux --location eastus

# Deploy US West
az container create \
  --resource-group vb-benchmark --name vb-wire-west \
  --image vbbenchmark.azurecr.io/wireserver:v1 \
  --cpu 2 --memory 4 --ports 9000 --ip-address Public \
  --os-type Linux --location westus2
```

### Run benchmark

```bash
pip install fastapi uvicorn numpy requests

python run_crossdc_benchmark.py \
  --east http://<east-ip>:9000 \
  --west http://<west-ip>:9000 \
  --batches 30 \
  --batch-size 1000 \
  --dim 1536
```

### Output files

| File | Contents |
|---|---|
| `results/crossdc_full_<ts>.json` | Full per-batch log, all formats, both regions |
| `results/landing_page_crossdc_stats.json` | Clean summary for landing page |
| `results/wire_comparison_<ts>.json` | Localhost baseline comparison |

---

## Files

| File | Purpose |
|---|---|
| `wire_server.py` | FastAPI server with `/ingest/rest`, `/ingest/binary`, `/ingest/chorus` |
| `run_crossdc_benchmark.py` | Cross-DC client — sends all formats, records all metrics |
| `run_wire_comparison.py` | Localhost baseline benchmark |
| `run_benchmark.py` | Original transatlantic ChromaDB→Qdrant migration benchmark |
| `generate_vectors.py` | Generate 100K legal domain vectors (seed=42, reproducible) |

---

## Cleanup

```bash
az group delete --name vb-benchmark --yes --no-wait
```
