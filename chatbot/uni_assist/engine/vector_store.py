"""
Vector Storage Module
======================
Handles persistence and loading of FAISS indexes and segment metadata.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from .segmentation import TextSegment


def create_directory_if_missing(directory: Path) -> None:
    """Ensure the specified directory exists, creating parent folders as needed."""
    directory.mkdir(parents=True, exist_ok=True)


def persist_vector_index(output_dir: Path, vector_index: faiss.Index, segments: list[TextSegment]) -> None:
    """Save the FAISS index and segment metadata to disk."""
    create_directory_if_missing(output_dir)

    faiss.write_index(vector_index, str(output_dir / "index.faiss"))

    with (output_dir / "chunks.jsonl").open("w", encoding="utf-8") as output_file:
        for segment in segments:
            output_file.write(json.dumps(asdict(segment), ensure_ascii=False) + "\n")


def restore_vector_index(storage_dir: Path) -> tuple[faiss.Index, list[TextSegment]]:
    """Load a previously saved FAISS index and its associated segments."""
    index_file = storage_dir / "index.faiss"
    metadata_file = storage_dir / "chunks.jsonl"
    
    if not index_file.exists() or not metadata_file.exists():
        raise FileNotFoundError(
            f"Index files not found in {storage_dir}. Execute ingestion first (python -m rag.ingest)."
        )

    vector_index = faiss.read_index(str(index_file))
    segments: list[TextSegment] = []
    
    with metadata_file.open("r", encoding="utf-8") as input_file:
        for json_line in input_file:
            record = json.loads(json_line)
            segments.append(TextSegment(**record))

    return vector_index, segments


def convert_to_faiss_array(embedding_vectors: list[list[float]]) -> np.ndarray:
    """Transform a list of embeddings into a FAISS-compatible numpy array."""
    array = np.array(embedding_vectors, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Embedding vectors must form a 2-dimensional matrix")
    return array
