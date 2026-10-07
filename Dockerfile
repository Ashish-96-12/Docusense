FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# CPU-only torch keeps the image ~2 GB smaller than the default CUDA build
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding and reranker models into the image so the container
# works offline and starts fast.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY docusense ./docusense
COPY ui ./ui
COPY data/samples ./data/samples

RUN useradd --create-home appuser && mkdir -p /app/data/store && chown -R appuser /app
USER appuser

EXPOSE 8000 8501
CMD ["uvicorn", "docusense.api:app", "--host", "0.0.0.0", "--port", "8000"]
