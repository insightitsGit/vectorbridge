"""Quick 5-batch smoke test to catch errors before full run."""
import sys, traceback, json
import numpy as np
import chromadb
from qdrant_client import QdrantClient

sys.path.insert(0, ".")
import run_benchmark as rb

CHROMA_PATH = "./chroma_data"
DATA_DIR    = "./data"

print("Loading ChromaDB ...")
client = chromadb.PersistentClient(path=CHROMA_PATH)
col    = client.get_collection("legal_docs")
vecs   = np.load(DATA_DIR + "/vectors.npy")
with open(DATA_DIR + "/metadata.json") as f:
    metas = json.load(f)

# Limit to 5 batches (5000 vectors)
vecs  = vecs[:5000]
metas = metas[:5000]
batches = rb.read_batches(col, vecs, metas)
print(f"Batches: {len(batches)}  (batch_size={rb.BATCH_SIZE})")

print("Connecting to Qdrant (Germany) ...")
qclient = QdrantClient(host="72.144.65.153", port=6333, timeout=60)

print("Generating key pair ...")
K, K_inv = rb._qr_key(rb.DIM)
print(f"  K shape: {K.shape}  K_inv shape: {K_inv.shape}")

print("\nRunning CHORUS (5 batches) ...")
try:
    stats = rb.run_chorus(batches, qclient, K, K_inv, 0xDEADBEEF)
    s = stats.summary("test")
    print(f"\nCHORUS OK")
    print(f"  wire_mb          = {s['wire_mb']}")
    print(f"  wm_verify_pct    = {s['wm_verify_pct']}")
    print(f"  wm_cosine_avg    = {s['wm_cosine_avg']}")
    print(f"  throughput_vps   = {s['throughput_vps']}")
except Exception:
    print("\nCHORUS FAILED:")
    traceback.print_exc()

print("\nRunning BASELINE (5 batches) ...")
try:
    stats2 = rb.run_baseline(batches, qclient)
    s2 = stats2.summary("test_base")
    print(f"\nBASELINE OK")
    print(f"  wire_mb        = {s2['wire_mb']}")
    print(f"  throughput_vps = {s2['throughput_vps']}")
except Exception:
    print("\nBASELINE FAILED:")
    traceback.print_exc()
