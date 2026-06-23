from typing import Iterator
import numpy as np

from .base import VectorConnector, VectorRecord, ConnectorConfig


class PineconeConnector(VectorConnector):
    """Read/write vectors from Pinecone."""

    def connect(self) -> None:
        from pinecone import Pinecone
        pc = Pinecone(api_key=self.config.api_key)
        self._index = pc.Index(self.config.index_name)
        self._namespace = self.config.namespace or ""

    def disconnect(self) -> None:
        self._index = None

    def count(self) -> int:
        stats = self._index.describe_index_stats()
        ns = stats.namespaces.get(self._namespace)
        return ns.vector_count if ns else 0

    def read_batches(self, batch_size: int = 256, offset: int = 0) -> Iterator[list[VectorRecord]]:
        # Pinecone does not support offset-based listing natively.
        # We use list() + fetch() pattern (Pinecone >= 3.x)
        ids_iter = self._index.list(namespace=self._namespace)
        batch = []
        for id_chunk in ids_iter:
            for vid in id_chunk:
                batch.append(vid)
                if len(batch) >= batch_size:
                    yield self._fetch_batch(batch)
                    batch = []
        if batch:
            yield self._fetch_batch(batch)

    def _fetch_batch(self, ids: list[str]) -> list[VectorRecord]:
        response = self._index.fetch(ids=ids, namespace=self._namespace)
        records = []
        for vid, item in response.vectors.items():
            vec = np.array(item.values, dtype=np.float32)
            meta = dict(item.metadata) if item.metadata else {}
            records.append(VectorRecord(id=vid, vector=vec, metadata=meta, namespace=self._namespace))
        return records

    def write_batch(self, records: list[VectorRecord]) -> int:
        vectors = [
            {
                "id": r.id,
                "values": r.vector.tolist(),
                "metadata": r.metadata,
            }
            for r in records
        ]
        self._index.upsert(vectors=vectors, namespace=self._namespace)
        return len(records)

    def create_index(self, dimension: int, **kwargs) -> None:
        from pinecone import Pinecone, ServerlessSpec
        pc = Pinecone(api_key=self.config.api_key)
        existing = [i.name for i in pc.list_indexes()]
        if self.config.index_name not in existing:
            pc.create_index(
                name=self.config.index_name,
                dimension=dimension,
                metric=kwargs.get("metric", "cosine"),
                spec=ServerlessSpec(
                    cloud=kwargs.get("cloud", "aws"),
                    region=kwargs.get("region", "us-east-1"),
                ),
            )
