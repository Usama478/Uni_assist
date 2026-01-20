"""
Conversational AI Module
=========================
Implements the RAG-based question answering system with context retrieval.
"""

from __future__ import annotations

from typing import Iterable

from openai import OpenAI

from .vector_store import restore_vector_index
from .semantic_search import generate_query_embedding, find_relevant_segments
from .config import initialize_config


ASSISTANT_INSTRUCTIONS = """You are a helpful University Assistant, providing accurate information and guidance based on official university resources and documents.

Your knowledge comes EXCLUSIVELY from the provided university documents and knowledge base. These documents contain comprehensive information about academic programs, policies, procedures, calendars, faculty information, and guidelines.

CRITICAL RULES:
1. **Strict Grounding**: Answer ONLY using information from the retrieved context. If the answer isn't in the provided sources, clearly state: "I don't have information about this topic in my knowledge base. Please check with the relevant university department or office."

2. **No External Knowledge**: Do NOT use general internet knowledge, common sense assumptions, or information not explicitly stated in the retrieved context.

3. **Complete Citations**: Always cite your sources in this format: (Document Title) or (Document ID — Document Title). When using multiple sources, cite all relevant ones.

4. **Comprehensive Answers**: Provide thorough, well-structured responses using:
   - Clear headings and subheadings
   - Bullet points for lists
   - Specific details, examples, and data from the sources
   - Direct quotes when particularly relevant

5. **Context Integration**: Multiple retrieved chunks may cover different aspects of the same topic. Synthesize information across chunks to provide complete, cohesive answers.

6. **Accurate Details**: Include specific information such as:
   - Dates and deadlines
   - Names, titles, and contact information
   - Academic requirements and procedures
   - Policy details and exceptions

7. **Professional Tone**: Maintain a respectful, helpful, and professional tone appropriate for a university assistant.

8. **Out-of-Scope Questions**: For questions unrelated to university operations, academic matters, or topics covered in the knowledge base, politely explain that you can only assist with university-related queries.

Remember: Your value comes from providing accurate, citation-backed answers from the university's official knowledge base. Never guess, hallucinate, or provide information beyond what is explicitly stated in the retrieved context."""


def _build_context_string(search_results: Iterable) -> str:
    """Format retrieved segments into a structured context block."""
    formatted_parts = []
    for index, result in enumerate(search_results, start=1):
        seg = result.segment
        formatted_parts.append(
            f"[Source {index}] {seg.doc_identifier} — {seg.doc_name} (file: {seg.source_filename})\n{seg.content}"
        )
    return "\n\n".join(formatted_parts)


def generate_response(*, user_query: str, conversation_history: list[tuple[str, str]]) -> str:
    """
    Generate an AI response using retrieval-augmented generation.
    
    Retrieves relevant context from the knowledge base and uses it
    to generate a grounded, accurate response to the user's question.
    """
    try:
        # Input validation
        if not user_query or not user_query.strip():
            return "Please enter a question to get started."
        
        cleaned_query = user_query.strip()
        if len(cleaned_query) > 2000:
            return "Your question is too long. Please limit it to 2000 characters."
        
        config = initialize_config()
        api_client = OpenAI(api_key=config.llm_api_key)

        # Load the vector index
        try:
            vector_index, all_segments = restore_vector_index(config.vector_store_path)
        except FileNotFoundError as index_error:
            return f"Error: Knowledge base not initialized. Run: python -m rag.ingest\n{index_error}"
        
        if not all_segments:
            return "Error: The knowledge base is empty. Please add documents to the source folder."

        # Perform semantic search
        try:
            query_vector = generate_query_embedding(
                api_client, 
                model_name=config.embedding_model_name, 
                query_text=cleaned_query
            )
            search_results = find_relevant_segments(
                vector_index=vector_index, 
                all_segments=all_segments, 
                query_embedding=query_vector, 
                num_results=config.retrieval_count
            )
        except Exception as search_error:
            return f"Search error: {str(search_error)}"
        
        if not search_results:
            return "I couldn't find relevant information in the knowledge base for your question. Try rephrasing or asking about different topics."

        context_block = _build_context_string(search_results)

        chat_messages = [
            {"role": "system", "content": ASSISTANT_INSTRUCTIONS},
        ]

        # Include recent conversation history for context continuity
        for prev_user_msg, prev_assistant_msg in conversation_history[-6:]:
            chat_messages.append({"role": "user", "content": prev_user_msg})
            chat_messages.append({"role": "assistant", "content": prev_assistant_msg})

        chat_messages.append(
            {
                "role": "user",
                "content": f"Question: {cleaned_query}\n\nRetrieved Context:\n{context_block}",
            }
        )

        # Generate the response
        try:
            completion = api_client.chat.completions.create(
                model=config.chat_model_name,
                messages=chat_messages,
                temperature=0.2,
                max_tokens=2000,
            )
            response_text = completion.choices[0].message.content or "Unable to generate a response."
            return response_text
        except Exception as generation_error:
            return f"Response generation failed: {str(generation_error)}. Please verify your API key and quota."
    
    except Exception as unexpected_error:
        return f"An unexpected error occurred: {str(unexpected_error)}. Please try again."
