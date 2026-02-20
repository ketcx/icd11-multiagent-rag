"""Smart chunking for ICD-11 text."""

from dataclasses import dataclass


@dataclass
class ChunkConfig:
    chunk_size: int = 1000  # target tokens
    chunk_overlap: int = 150  # overlap matching
    min_chunk_size: int = 200  # do not create tiny chunks
    max_chunk_size: int = 1500  # hard limit
    respect_sections: bool = True  # do not break within semantic sections
    respect_codes: bool = True  # do not break within diagnosis codes


def chunk_documents(pages: list[dict], config: ChunkConfig) -> list[dict]:
    """Divides pages into chunks respecting structure.

    Priority:
    1. Break by section/heading (if detected)
    2. Break by ICD-11 code (if detected)
    3. Fallback: break by size with overlap

    Returns:
        List of chunks with metadata:
        {content, metadata: {source_pdf, page, section, code, uri, chunk_id}}
    """
    # Placeholder implementation based on LangChain RecursiveCharacterTextSplitter for now
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = []
    chunk_id_counter = 0
    for page in pages:
        text = page["text"]

        # Simple split
        page_chunks = splitter.split_text(text)
        for i, chunk_text in enumerate(page_chunks):
            # Try to associate codes and headings
            associated_codes = [c for c in page.get("codes", []) if c in chunk_text]
            metadata = {
                "source_pdf": page["source_pdf"],
                "page": page["page_number"],
                "chunk_id": chunk_id_counter,
            }
            if associated_codes:
                # Just use the first one as primary for now
                metadata["code"] = associated_codes[0]

            chunks.append({"content": chunk_text, "metadata": metadata})
            chunk_id_counter += 1

    return chunks
