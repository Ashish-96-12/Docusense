import os
from typing import List, Dict, Any
from pdfminer.high_level import extract_text
from docx import Document


def parse_and_chunk(filepath: str, chunking_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse document and split into chunks"""
    # Extract text based on file type
    text = ""
    file_ext = os.path.splitext(filepath)[1].lower()

    if file_ext == ".pdf":
        text = extract_text(filepath)
    elif file_ext == ".docx":
        doc = Document(filepath)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    elif file_ext in [".txt", ".md"]:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")

    # Get chunking parameters
    chunk_size = chunking_config.get("chunk_size", 500)
    chunk_overlap = chunking_config.get("chunk_overlap", 50)

    # Create chunks
    chunks = []
    words = text.split()

    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)

        if chunk_text.strip():  # Only add non-empty chunks
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "start_index": i,
                    "end_index": min(i + chunk_size, len(words)),
                    "source": filepath
                }
            })

    return chunks