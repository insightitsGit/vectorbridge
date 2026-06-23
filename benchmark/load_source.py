"""
Load generated vectors into ChromaDB (source container — US East).
Run this on the US side BEFORE starting the migration benchmark.

Usage:
  python load_source.py [--host localhost] [--port 8000] [--data-dir ./data]
"""

import argparse
import json
import time
import numpy as np
import chromadb
from chromadb.config import Settings

COLLECTION = "legal_docs"
BATCH = 500


def load(host: str, port: int, data_dir: str):
    print(f"Connecting to ChromaDB at {host}:{port} ...")
    client = chromadb.HttpClient(host=host, port=port,
                                  settings=Settings(anonymized_telemetry=False))

    # Drop + recreate for clean benchmark
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    print("Loading vectors.npy ...")
    vectors = np.load(f"{data_dir}/vectors.npy")
    with open(f"{data_dir}/metadata.json") as f:
        metadata = json.load(f)

    n = len(vectors)
    raw_mb = vectors.nbytes / 1024**2
    print(f"  {n:,} vectors × {vectors.shape[1]}-dim = {raw_mb:.0f} MB raw")

    t0 = time.time()
    inserted = 0
    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        batch_vecs = vectors[start:end].tolist()
        batch_ids = [m["id"] for m in metadata[start:end]]
        batch_meta = metadata[start:end]
        col.add(ids=batch_ids, embeddings=batch_vecs, metadatas=batch_meta)
        inserted += end - start
        if inserted % 10_000 == 0:
            elapsed = time.time() - t0
            rate = inserted / elapsed
            eta = (n - inserted) / rate
            print(f"  {inserted:,}/{n:,} ({100*inserted/n:.0f}%) "
                  f"— {rate:.0f} vecs/s — ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nDone. {n:,} vectors loaded in {elapsed:.1f}s "
          f"({n/elapsed:.0f} vecs/s)")
    print(f"Collection: {COLLECTION}  |  Count: {col.count()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--data-dir", default="./data")
    args = p.parse_args()
    load(args.host, args.port, args.data_dir)
