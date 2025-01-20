# db.py
import chromadb
from chromadb.config import Settings


def get_chroma_client():
    return chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(
            allow_reset=True, anonymized_telemetry=False, is_persistent=True
        ),
    )


def init_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(
        name="pydantic_ai_docs",
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,  # We're providing our own embeddings
    )
