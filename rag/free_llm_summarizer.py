import requests
import subprocess
import json
from typing import Dict


def check_ollama_running():
    """Check if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=3)
        return response.status_code == 200
    except:
        return False


def start_ollama():
    """Try to start Ollama if it's installed"""
    try:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except:
        return False


def ollama_generate_answer(query: str, context: str, model: str = "llama2") -> str:
    """Generate answer using free Ollama LLM"""

    # Check if Ollama is running
    if not check_ollama_running():
        print("Starting Ollama...")
        if not start_ollama():
            return fallback_answer(context)

        # Wait a moment for startup
        import time
        time.sleep(2)

        if not check_ollama_running():
            return fallback_answer(context)

    try:
        prompt = f"""You are a helpful assistant. Based on the provided context, answer the user's question clearly and accurately.

Context:
{context[:1500]}

Question: {query}

Answer:"""

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 150
            }
        }

        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            answer = result.get("response", "").strip()
            if answer:
                return answer

        return fallback_answer(context)

    except Exception as e:
        print(f"Ollama error: {e}")
        return fallback_answer(context)


def fallback_answer(context: str) -> str:
    """Simple fallback when LLM isn't available"""
    sentences = context.split('.')[:3]
    clean_sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    return '. '.join(clean_sentences[:2]) + '.'