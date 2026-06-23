from typing import Iterator
import numpy as np

from .base import VectorConnector, VectorRecord, ConnectorConfig


class ChromaDBConnector(VectorConnector):
    """Read/write vectors from ChromaDB (local or server)."""

    def connect(self) -> None:
        import chromadb
        if self.config.chroma_path:
            self._client = chromadb.PersistentClient(path=self.config.chroma_path)
        else:
            self._client = chromadb.HttpClient(
                host=self.config.chroma_host,
                port=self.config.chroma_port,
            )
        self._col = self._client.get_or_create_collection(self.config.collection_name)

    def disconnect(self) -> None:
        self._client = None

    def count(self) -> int:
        return self._col.count()

    def read_batches(self, batch_size: int = 256, offset: int = 0) -> Iterator[list[VectorRecord]]:
        total = self.count()
        current = offset
        while current < total:
            result = self._col.get(
                limit=batch_size,
                offset=current,
                include=["embeddings", "metadatas", "documents"],
            )
            if not result["ids"]:
                break
            batch = []
            for i, vid in enumerate(result["ids"]):
                vec = np.array(result["embeddings"][i], dtype=np.float32)
                meta = result["metadatas"][i] or {}
                batch.append(VectorRecord(id=vid, vector=vec, metadata=meta))
            yield batch
            current += batch_size

    def write_batch(self, records: list[VectorRecord]) -> int:
        self._col.upsert(
            ids=[r.id for r in records],
            embeddings=[r.vector.tolist() for r in records],
            metadatas=[r.metadata or {} for r in records],
        )
        return len(records)

    def create_index(self, dimension: int, **kwargs) -> None:
        import chromadb
        if self.config.chroma_path:
            self._client = chromadb.PersistentClient(path=self.config.chroma_path)
        else:
            self._client = chromadb.HttpClient(
                host=self.config.chroma_host,
                port=self.config.chroma_port,
            )
        self._col = self._client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": kwargs.get("metric", "cosine")},
        )
