from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator
import numpy as np


@dataclass
class VectorRecord:
    id: str
    vector: np.ndarray          # float32, shape (dim,)
    metadata: dict = field(default_factory=dict)
    namespace: str = ""


@dataclass
class ConnectorConfig:
    """Generic config — each connector reads what it needs."""
    # pgvector
    host: str = "localhost"
    port: int = 5432
    database: str = ""
    user: str = ""
    password: str = ""
    table: str = "embeddings"
    vector_column: str = "embedding"
    id_column: str = "id"
    metadata_columns: list = field(default_factory=list)

    # Pinecone
    api_key: str = ""
    index_name: str = ""
    environment: str = ""
    namespace: str = ""

    # ChromaDB
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    collection_name: str = "embeddings"
    chroma_path: str = ""         # local persistent path

    # Weaviate
    weaviate_url: str = ""
    weaviate_api_key: str = ""
    class_name: str = "Embedding"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    collection: str = "embeddings"

    # FAISS
    faiss_path: str = ""          # path to .index file
    faiss_ids_path: str = ""      # path to id mapping .npy file

    # Distance metric — must match between source and target
    # Accepted values: "cosine" | "l2" | "dot"
    distance_metric: str = "cosine"

    # General
    batch_size: int = 256
    dimension: int = 0            # 0 = auto-detect


class VectorConnector(ABC):
    """Base class every connector must implement."""

    def __init__(self, config: ConnectorConfig):
        self.config = config
        self._client = None

    @abstractmethod
    def connect(self) -> None:
        """Open connection / validate credentials."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection and release resources."""

    @abstractmethod
    def count(self) -> int:
        """Total number of vectors in the source/target."""

    @abstractmethod
    def read_batches(self, batch_size: int = 256, offset: int = 0) -> Iterator[list[VectorRecord]]:
        """Yield batches of VectorRecord from the store."""

    @abstractmethod
    def write_batch(self, records: list[VectorRecord]) -> int:
        """Write a batch of VectorRecord. Returns count written."""

    def supports_incremental(self) -> bool:
        return False

    def read_changes(self, since_checkpoint: str, batch_size: int = 256) -> Iterator[list[VectorRecord]]:
        raise NotImplementedError(f"{self.__class__.__name__} does not support incremental sync")

    def create_index(self, dimension: int, **kwargs) -> None:
        """Create target index/collection/table if it does not exist."""
        pass

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()
