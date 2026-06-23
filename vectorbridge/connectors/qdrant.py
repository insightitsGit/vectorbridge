from typing import Iterator
import numpy as np

from .base import VectorConnector, VectorRecord, ConnectorConfig


class QdrantConnector(VectorConnector):
    """Read/write vectors from Qdrant."""

    def connect(self) -> None:
        from qdrant_client import QdrantClient
        self._client = QdrantClient(
            url=self.config.qdrant_url,
            api_key=self.config.qdrant_api_key or None,
        )

    def disconnect(self) -> None:
        self._client = None

    def count(self) -> int:
        info = self._client.get_collection(self.config.collection)
        return info.vectors_count

    def read_batches(self, batch_size: int = 256, offset: int = 0) -> Iterator[list[VectorRecord]]:
        next_offset = None
        first = True
        while first or next_offset is not None:
            first = False
            result, next_offset = self._client.scroll(
                collection_name=self.config.collection,
                limit=batch_size,
                offset=next_offset,
                with_vectors=True,
                with_payload=True,
            )
            if not result:
                break
            batch = []
            for point in result:
                vec = np.array(point.vector, dtype=np.float32)
                meta = dict(point.payload) if point.payload else {}
                batch.append(VectorRecord(id=str(point.id), vector=vec, metadata=meta))
            yield batch

    def write_batch(self, records: list[VectorRecord]) -> int:
        from qdrant_client.models import PointStruct
        points = [
            PointStruct(
                id=r.id if r.id.isdigit() else abs(hash(r.id)) % (2**63),
                vector=r.vector.tolist(),
                payload=r.metadata,
            )
            for r in records
        ]
        self._client.upsert(collection_name=self.config.collection, points=points)
        return len(records)

    def create_index(self, dimension: int, **kwargs) -> None:
        from qdrant_client.models import Distance, VectorParams
        metric_map = {"cosine": Distance.COSINE, "euclidean": Distance.EUCLID, "dot": Distance.DOT}
        distance = metric_map.get(kwargs.get("metric", "cosine"), Distance.COSINE)
        existing = [c.name for c in self._client.get_collections().collections]
        if self.config.collection not in existing:
            self._client.create_collection(
                collection_name=self.config.collection,
                vectors_config=VectorParams(size=dimension, distance=distance),
            )
