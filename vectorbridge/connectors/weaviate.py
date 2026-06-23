from typing import Iterator
import numpy as np

from .base import VectorConnector, VectorRecord, ConnectorConfig


class WeaviateConnector(VectorConnector):
    """Read/write vectors from Weaviate."""

    def connect(self) -> None:
        import weaviate
        auth = None
        if self.config.weaviate_api_key:
            auth = weaviate.auth.AuthApiKey(api_key=self.config.weaviate_api_key)
        self._client = weaviate.connect_to_custom(
            http_host=self.config.weaviate_url.replace("http://", "").replace("https://", "").split(":")[0],
            http_port=int(self.config.weaviate_url.split(":")[-1]) if ":" in self.config.weaviate_url else 8080,
            http_secure="https" in self.config.weaviate_url,
            grpc_host=self.config.weaviate_url.replace("http://", "").replace("https://", "").split(":")[0],
            grpc_port=50051,
            grpc_secure=False,
            auth_credentials=auth,
        )
        self._collection = self._client.collections.get(self.config.class_name)

    def disconnect(self) -> None:
        if self._client:
            self._client.close()

    def count(self) -> int:
        result = self._collection.aggregate.over_all(total_count=True)
        return result.total_count or 0

    def read_batches(self, batch_size: int = 256, offset: int = 0) -> Iterator[list[VectorRecord]]:
        cursor = None
        while True:
            query = self._collection.query.fetch_objects(
                limit=batch_size,
                after=cursor,
                include_vector=True,
            )
            if not query.objects:
                break
            batch = []
            for obj in query.objects:
                vec = np.array(obj.vector.get("default", []), dtype=np.float32)
                meta = dict(obj.properties) if obj.properties else {}
                batch.append(VectorRecord(id=str(obj.uuid), vector=vec, metadata=meta))
            yield batch
            cursor = query.objects[-1].uuid

    def write_batch(self, records: list[VectorRecord]) -> int:
        from weaviate.classes.data import DataObject
        objects = [
            DataObject(
                properties=r.metadata,
                vector=r.vector.tolist(),
                uuid=r.id if len(r.id) == 36 else None,
            )
            for r in records
        ]
        self._collection.data.insert_many(objects)
        return len(records)

    def create_index(self, dimension: int, **kwargs) -> None:
        import weaviate.classes.config as wc
        existing = [c.name for c in self._client.collections.list_all().values()]
        if self.config.class_name not in existing:
            metric_map = {
                "cosine": wc.VectorDistances.COSINE,
                "euclidean": wc.VectorDistances.L2_SQUARED,
                "dot": wc.VectorDistances.DOT,
            }
            self._client.collections.create(
                name=self.config.class_name,
                vectorizer_config=wc.Configure.Vectorizer.none(),
                vector_index_config=wc.Configure.VectorIndex.hnsw(
                    distance_metric=metric_map.get(kwargs.get("metric", "cosine"), wc.VectorDistances.COSINE)
                ),
            )
            self._collection = self._client.collections.get(self.config.class_name)
