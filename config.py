"""
Configuration settings for DocuSense
"""

CONFIG = {
    "chunking": {
        "chunk_size": 500,
        "overlap": 50
    },

    "retrieval": {
        "top_k": 6,
        "min_score": 0.1
    },

    "storage": {
        "corpus_dir": "storage/corpus",
        "chunks_dir": "storage/chunks",
        "faiss_dir": "storage/faiss",
        "logs_dir": "storage/logs"
    },

    "summarization": {
        "max_sentences": 3,
        "algorithm": "textrank"
    },

    # Answer generation method: "smart" for intelligent rule-based answers or "textrank"
    "answer_method": "smart",  # Use smart rule-based answering

    # OpenAI settings (if using GPT - requires API key)
    "openai": {
        "model": "gpt-3.5-turbo",
        "max_tokens": 300,
        "temperature": 0.3
    },

    # Ollama settings (for local LLM)
    "ollama": {
        "base_url": "http://localhost:11434",
        "model": "llama2"
    }
}