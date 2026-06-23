"""
Generate 100K synthetic legal document chunk embeddings (1536-dim, float32).
Produces reproducible data: same seed = same vectors every run.

Dataset represents:
  - 100,000 contract/case document chunks
  - 1536-dim embeddings (text-embedding-3-small format)
  - Realistic cluster structure: 20 practice areas × 5,000 docs each
  - Total raw size: ~614 MB float32
"""

import numpy as np
import json
import os
import time

SEED = 42
NUM_VECTORS = 100_000
DIM = 1536
NUM_CLUSTERS = 20  # practice areas: contracts, litigation, IP, M&A, etc.

PRACTICE_AREAS = [
    "contracts", "litigation", "intellectual_property", "mergers_acquisitions",
    "employment", "real_estate", "tax_law", "bankruptcy", "securities",
    "antitrust", "environmental", "healthcare_law", "immigration",
    "criminal_defense", "family_law", "insurance", "privacy_data",
    "regulatory_compliance", "arbitration", "international_trade"
]

DOC_TYPES = ["contract", "motion", "brief", "deposition", "exhibit", "opinion", "memo"]


def generate(out_dir: str = "./data"):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print(f"Generating {NUM_VECTORS:,} vectors × {DIM}-dim ...")
    t0 = time.time()

    # Cluster centers — one per practice area
    centers = rng.standard_normal((NUM_CLUSTERS, DIM)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    # Assign vectors to clusters
    cluster_ids = rng.integers(0, NUM_CLUSTERS, size=NUM_VECTORS)
    noise = rng.standard_normal((NUM_VECTORS, DIM)).astype(np.float32) * 0.3
    vectors = centers[cluster_ids] + noise

    # L2-normalize (typical for cosine-similarity search)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = (vectors / norms).astype(np.float32)

    raw_bytes = vectors.nbytes
    print(f"  Raw float32 size: {raw_bytes / 1024**2:.1f} MB")

    # Build metadata
    metadata = []
    for i in range(NUM_VECTORS):
        area = PRACTICE_AREAS[cluster_ids[i]]
        doc_type = DOC_TYPES[i % len(DOC_TYPES)]
        metadata.append({
            "id": f"doc_{i:07d}",
            "practice_area": area,
            "doc_type": doc_type,
            "chunk_index": i % 20,
            "firm_id": f"firm_{(i // 1000) % 50:03d}",
            "year": 2018 + (i % 7),
        })

    # Save numpy array + metadata separately for fast loading
    np.save(os.path.join(out_dir, "vectors.npy"), vectors)
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    elapsed = time.time() - t0
    print(f"  Generated in {elapsed:.1f}s")
    print(f"  Saved to {out_dir}/vectors.npy ({raw_bytes/1024**2:.0f} MB)")
    print(f"  Metadata: {out_dir}/metadata.json")
    return vectors, metadata


if __name__ == "__main__":
    generate()
