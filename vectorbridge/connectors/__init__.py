from .base import VectorConnector, VectorRecord, ConnectorConfig
from .pgvector import PgvectorConnector
from .pinecone import PineconeConnector
from .chromadb import ChromaDBConnector
from .qdrant import QdrantConnector
from .weaviate import WeaviateConnector
from .faiss import FaissConnector

CONNECTOR_MAP = {
    "pgvector": PgvectorConnector,
    "pinecone": PineconeConnector,
    "chromadb": ChromaDBConnector,
    "qdrant": QdrantConnector,
    "weaviate": WeaviateConnector,
    "faiss": FaissConnector,
}


def get_connector(name: str, config: ConnectorConfig) -> VectorConnector:
    name = name.lower()
    if name not in CONNECTOR_MAP:
        raise ValueError(f"Unknown connector '{name}'. Available: {list(CONNECTOR_MAP)}")
    return CONNECTOR_MAP[name](config)
