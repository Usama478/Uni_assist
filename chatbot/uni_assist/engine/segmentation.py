"""
Text Segmentation Module
=========================
Provides utilities for splitting documents into searchable segments.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextSegment:
    """Represents a portion of a document with associated metadata."""
    segment_id: str
    source_filename: str
    doc_identifier: str
    doc_name: str
    content: str


def _parse_header_field(content: str, field_name: str) -> str:
    """
    Parse a header field from document content.
    Searches for field name and returns the value from the next non-empty line.
    """
    content_lines = [line.rstrip("\n") for line in content.splitlines()]
    for idx, line in enumerate(content_lines):
        if line.strip().lower() == field_name.strip().lower():
            for next_idx in range(idx + 1, min(idx + 10, len(content_lines))):
                if content_lines[next_idx].strip():
                    return content_lines[next_idx].strip()
    return ""


def segment_text(
    *,
    source_filename: str,
    full_text: str,
    doc_identifier: str,
    doc_name: str,
    max_segment_length: int,
    overlap_length: int,
) -> list[TextSegment]:
    """
    Divide document content into overlapping segments for vector indexing.
    
    Uses intelligent boundary detection to split at natural break points
    like paragraph endings or sentence terminators rather than mid-word.
    """
    normalized_text = full_text.strip()
    if not normalized_text:
        return []

    segments: list[TextSegment] = []
    position = 0
    counter = 0

    while position < len(normalized_text):
        boundary = min(position + max_segment_length, len(normalized_text))
        
        # Attempt to find natural break points when not at document end
        if boundary < len(normalized_text):
            # Search for paragraph boundaries in the trailing portion
            search_begin = max(position, boundary - 200)
            paragraph_end = normalized_text.rfind('\n\n', search_begin, boundary)
            if paragraph_end > position:
                boundary = paragraph_end + 2
            else:
                # Fall back to sentence-ending punctuation
                terminators = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
                for terminator in terminators:
                    sentence_end = normalized_text.rfind(terminator, search_begin, boundary)
                    if sentence_end > position:
                        boundary = sentence_end + len(terminator)
                        break
        
        segment_content = normalized_text[position:boundary].strip()

        # Only include segments with meaningful content
        if segment_content and len(segment_content) > 50:
            seg_id = f"{doc_identifier}::seg_{counter:04d}"
            segments.append(
                TextSegment(
                    segment_id=seg_id,
                    source_filename=source_filename,
                    doc_identifier=doc_identifier,
                    doc_name=doc_name,
                    content=segment_content,
                )
            )
            counter += 1

        if boundary >= len(normalized_text):
            break

        # Move forward with overlap for context continuity
        position = max(position + 1, boundary - overlap_length)

    return segments
