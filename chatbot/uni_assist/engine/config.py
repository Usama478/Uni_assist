"""
Configuration Management Module
================================
Handles environment variables, paths, and application settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfiguration:
    """Immutable configuration container for the application."""
    project_root: Path
    source_docs_path: Path
    vector_store_path: Path
    llm_api_key: str
    chat_model_name: str
    embedding_model_name: str
    segment_size: int
    segment_overlap: int
    retrieval_count: int


def initialize_config() -> AppConfiguration:
    """Load and validate application configuration from environment."""
    load_dotenv()
    
    llm_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not llm_key:
        raise SystemExit(
            "Missing OPENAI_API_KEY environment variable. Add it to your .env file."
        )
    
    base_path = Path(__file__).resolve().parents[1]
    # Try to use documents folder relative to project, otherwise use external path
    default_docs = (base_path / ".." / "docs" / "documents").resolve()
    external_docs = Path("c:/Users/Dell/Desktop/scraper/umt_complete_data").resolve()
    docs_folder = default_docs if default_docs.exists() else external_docs
    store_folder = (base_path / "index").resolve()

    return AppConfiguration(
        project_root=base_path,
        source_docs_path=docs_folder,
        vector_store_path=store_folder,
        llm_api_key=llm_key,
        chat_model_name=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        embedding_model_name=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        segment_size=int(os.getenv("FYP_CHUNK_SIZE_CHARS", "2000")),
        segment_overlap=int(os.getenv("FYP_CHUNK_OVERLAP_CHARS", "400")),
        retrieval_count=int(os.getenv("FYP_TOP_K", "8")),
    )
