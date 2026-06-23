from typing import Iterator
import numpy as np

from .base import VectorConnector, VectorRecord, ConnectorConfig


class FaissConnector(VectorConnector):
    """Read-only connector for FAISS index files."""

    def connect(self) -> None:
        import faiss
        self._index = faiss.read_index(self.config.faiss_path)
        if self.config.faiss_ids_path:
            self._ids = np.load(self.config.faiss_ids_path, allow_pickle=True)
        else:
            self._ids = np.array([str(i) for i in range(self._index.ntotal)])

    def disconnect(self) -> None:
        self._index = None

    def count(self) -> int:
        return self._index.ntotal

    def read_batches(self, batch_size: int = 256, offset: int = 0) -> Iterator[list[VectorRecord]]:
        total = self._index.ntotal
        current = offset
        while current < total:
            end = min(current + batch_size, total)
            # reconstruct extracts raw vectors by index position
            vectors = self._index.reconstruct_n(current, end - current)
            batch = []
            for i, vec in enumerate(vectors):
                vid = str(self._ids[current + i])
                batch.append(VectorRecord(
                    id=vid,
                    vector=vec.astype(np.float32),
                ))
            yield batch
            current += batch_size

    def write_batch(self, records: list[VectorRecord]) -> int:
        raise NotImplementedError("FAISS connector is read-only. Use as source only.")

    def create_index(self, dimension: int, **kwargs) -> None:
        raise NotImplementedError("FAISS connector is read-only.")
