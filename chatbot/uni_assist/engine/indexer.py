"""
Document Ingestion Pipeline
============================
Processes Markdown documents and builds the vector search index incrementally.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import faiss
from openai import OpenAI

# NOTE: Added restore_vector_index to imports
from .segmentation import segment_text
from .vector_store import persist_vector_index, convert_to_faiss_array, restore_vector_index
from .config import initialize_config


def _sanitize_extracted_text(raw_content: str) -> str:
    """Remove artifacts and normalize whitespace from text content."""
    cleaned = re.sub(r'\s+', ' ', raw_content)
    cleaned = re.sub(r'\x00', '', cleaned)
    cleaned = re.sub(r'[\u200b-\u200f\ufeff]', '', cleaned)
    return cleaned.strip()


def _get_markdown_metadata(md_filename: str) -> tuple[str, str]:
    """Extract or generate document identifier and title from Markdown filename."""
    extracted_title = md_filename.replace(".md", "").replace("_", " ").strip()
    identifier = md_filename.replace(".md", "").replace(" ", "-").replace("_", "-")
    return identifier, extracted_title


def _process_markdown_document(md_path: Path) -> tuple[str, str, str]:
    """Read and extract all text content from a Markdown file."""
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content or len(content.strip()) < 50:
            print(f"Warning: {md_path.name} contains insufficient content")
            return "", "", ""
        
        doc_id, doc_title = _get_markdown_metadata(md_path.name)
        sanitized = _sanitize_extracted_text(content)
        
        return sanitized, doc_id, doc_title
        
    except Exception as read_error:
        print(f"Error: Unable to process {md_path.name}: {read_error}")
        return "", "", ""


def create_search_index() -> None:
    """Build or update the vector search index from Markdown documents."""
    config = initialize_config()

    if not config.source_docs_path.exists():
        raise FileNotFoundError(
            f"Source documents folder missing: {config.source_docs_path}"
        )

    # Setup paths
    faiss_index_path = config.vector_store_path / "index.faiss"
    metadata_path = config.vector_store_path / "chunks.jsonl"
    
    # State variables
    existing_segments = []
    processed_filenames = set()
    search_index = None
    
    # ---------------------------------------------------------
    # 1. LOAD EXISTING INDEX (Incremental Logic)
    # ---------------------------------------------------------
    force_rebuild = os.getenv("FYP_REBUILD_INDEX", "0") == "1"

    if not force_rebuild and faiss_index_path.exists() and metadata_path.exists():
        print(f"\nLoading existing index from: {config.vector_store_path}")
        try:
            search_index, existing_segments = restore_vector_index(config.vector_store_path)
            # Create a set of filenames we have already finished
            processed_filenames = {seg.source_filename for seg in existing_segments}
            print(f"Loaded {len(existing_segments)} existing segments from {len(processed_filenames)} files.")
        except Exception as e:
            print(f"Error loading index ({e}). Starting fresh...")
            search_index = None
            existing_segments = []

    # ---------------------------------------------------------
    # 2. IDENTIFY NEW FILES
    # ---------------------------------------------------------
    all_md_files = sorted(config.source_docs_path.glob("*.md"))
    new_files = [f for f in all_md_files if f.name not in processed_filenames]

    if not new_files:
        print("\nAll documents are already indexed. No new work to do.")
        return

    print(f"\n{'='*60}")
    print(f"Found {len(new_files)} new documents to process")
    print(f"{'='*60}\n")

    # ---------------------------------------------------------
    # 3. PROCESS NEW FILES
    # ---------------------------------------------------------
    openai_client = OpenAI(api_key=config.llm_api_key)
    new_segments = []
    
    for md_file in new_files:
        print(f"Processing: {md_file.name}")
        extracted_text, doc_id, doc_title = _process_markdown_document(md_file)
        
        if not extracted_text:
            print(f"  Skipped (no content)\n")
            continue
        
        segments_before = len(new_segments)
        new_segments.extend(
            segment_text(
                source_filename=md_file.name,
                full_text=extracted_text,
                doc_identifier=doc_id,
                doc_name=doc_title,
                max_segment_length=config.segment_size,
                overlap_length=config.segment_overlap,
            )
        )
        print(f"  Created {len(new_segments) - segments_before} new segments\n")

    if not new_segments:
        print("No valid text segments found in new files.")
        return

    print(f"Total NEW segments to embed: {len(new_segments)}")

    # ---------------------------------------------------------
    # 4. GENERATE EMBEDDINGS (Only for new stuff)
    # ---------------------------------------------------------
    embedding_vectors: list[list[float]] = []
    batch_limit = 64
    
    for batch_start in range(0, len(new_segments), batch_limit):
        batch_segments = new_segments[batch_start : batch_start + batch_limit]
        embedding_response = openai_client.embeddings.create(
            model=config.embedding_model_name,
            input=[seg.content for seg in batch_segments],
        )
        embedding_vectors.extend([item.embedding for item in embedding_response.data])
        print(f"Embedded {min(batch_start + batch_limit, len(new_segments))}/{len(new_segments)}")

    # ---------------------------------------------------------
    # 5. MERGE AND SAVE
    # ---------------------------------------------------------
    new_vector_matrix = convert_to_faiss_array(embedding_vectors)
    faiss.normalize_L2(new_vector_matrix)

    if search_index is None:
        # If we started fresh, create the index now
        vector_dimension = new_vector_matrix.shape[1]
        search_index = faiss.IndexFlatIP(vector_dimension)
        search_index.add(new_vector_matrix)
    else:
        # If we loaded an index, just add to it
        search_index.add(new_vector_matrix)

    # Combine old and new segments for the metadata file
    total_segments = existing_segments + new_segments

    persist_vector_index(config.vector_store_path, search_index, total_segments)
    print(f"\nSuccess! Index updated. Total documents: {len(processed_filenames) + len(new_files)}")


if __name__ == "__main__":
    create_search_index()