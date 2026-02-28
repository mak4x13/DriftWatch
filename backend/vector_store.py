"""ChromaDB persistence and source retrieval helpers for DriftWatch."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from backend.logging_utils import get_run_logger
from backend.mistral_client import MistralClient

logger = logging.getLogger(__name__)

CHROMA_PATH = Path(__file__).resolve().parent.parent / "chroma_data"
_client: chromadb.PersistentClient | None = None
_collections: dict[int, Any] = {}
_mistral = MistralClient()


def init_chroma(dimension: int | None = None) -> Any:
    """Create or load the persistent Chroma client and optional dimension-scoped collection."""

    global _client
    if _client is None:
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(CHROMA_PATH),
            settings=Settings(anonymized_telemetry=False),
        )

    if dimension is None:
        return _client

    collection = _collections.get(dimension)
    if collection is not None:
        return collection

    collection = _client.get_or_create_collection(
        name=_collection_name(dimension),
        metadata={"hnsw:space": "cosine"},
    )
    _collections[dimension] = collection
    return collection


async def ingest_sources(texts: list[str], run_id: str) -> None:
    """Embed and persist a set of source documents for one pipeline run."""

    if not texts:
        return

    run_logger = get_run_logger(logger, run_id)
    embeddings = await _mistral.embed(texts, run_id=run_id)
    dimension = len(embeddings[0]) if embeddings else 0
    collection = init_chroma(dimension)
    ids = [f"{run_id}:{index}" for index, _ in enumerate(texts)]
    metadatas = [{"run_id": run_id, "doc_index": index} for index, _ in enumerate(texts)]

    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    run_logger.info("Ingested %s source documents.", len(texts))


async def query_sources(claim: str, run_id: str, n_results: int = 3) -> list[str]:
    """Query the Chroma collection for source passages relevant to a claim."""

    if not claim.strip():
        return []

    query_embedding = await _mistral.embed([claim], run_id=run_id)
    dimension = len(query_embedding[0]) if query_embedding else 0
    collection = init_chroma(dimension)
    result = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        where={"run_id": run_id},
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])
    if not documents:
        return []
    return [document for document in documents[0] if isinstance(document, str)]


def _collection_name(dimension: int) -> str:
    """Build a stable collection name for one embedding dimensionality."""

    return f"sources_{dimension}"


def reset_source_collections() -> list[str]:
    """Delete all persistent source collections so the demo can start clean."""

    client = init_chroma()
    deleted: list[str] = []
    collections = client.list_collections()
    for collection in collections:
        name = collection if isinstance(collection, str) else getattr(collection, "name", "")
        if not name.startswith("sources_"):
            continue
        client.delete_collection(name=name)
        deleted.append(name)

    _collections.clear()
    return deleted
