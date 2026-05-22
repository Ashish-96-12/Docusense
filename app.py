from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from rag.pipeline import RagPipeline
import os
import yaml
import json
from typing import Dict

app = FastAPI(title="DocuSense (RAG-only, no Pydantic)")

with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f)

PIPE = RagPipeline(CFG)

# Store active document sessions
ACTIVE_SESSIONS: Dict[str, str] = {}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Ingest a new document and return a session ID"""
    os.makedirs("storage/corpus", exist_ok=True)
    path = os.path.join("storage/corpus", file.filename)

    with open(path, "wb") as f:
        f.write(await file.read())

    doc_id = PIPE.ingest(path)

    # Create a session for this document
    session_id = f"session_{doc_id}"
    ACTIVE_SESSIONS[session_id] = doc_id

    return {
        "ok": True,
        "doc_id": doc_id,
        "session_id": session_id,
        "filename": file.filename
    }


@app.post("/query")
async def query(request: Request):
    """Query a specific document using session ID"""
    payload = await request.json()
    q = payload.get("query", "")
    session_id = payload.get("session_id", "")

    if not session_id or session_id not in ACTIVE_SESSIONS:
        raise HTTPException(status_code=400, detail="Invalid or missing session_id")

    doc_id = ACTIVE_SESSIONS[session_id]
    answer, evidence = PIPE.answer(q, doc_id=doc_id)

    return {
        "answer": answer,
        "evidence": evidence,
        "doc_id": doc_id
    }


@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    return {"sessions": ACTIVE_SESSIONS}


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a specific session"""
    if session_id in ACTIVE_SESSIONS:
        doc_id = ACTIVE_SESSIONS[session_id]
        PIPE.clear_document(doc_id)
        del ACTIVE_SESSIONS[session_id]
        return {"ok": True, "message": f"Session {session_id} cleared"}
    return {"ok": False, "message": "Session not found"}