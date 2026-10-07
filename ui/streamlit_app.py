"""DocuSense chat UI. Talks to the FastAPI backend.

Run with:  streamlit run ui/streamlit_app.py
"""
import os
import re

import httpx
import streamlit as st

API_URL = os.getenv("DOCUSENSE_API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="DocuSense", page_icon="📄", layout="wide")


# ---------- API helpers ----------

def api(method: str, path: str, **kwargs):
    try:
        resp = httpx.request(method, f"{API_URL}{path}", timeout=180, **kwargs)
    except httpx.HTTPError:
        st.error(f"Can't reach the DocuSense API at {API_URL}. Is it running?")
        st.stop()
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise RuntimeError(str(detail))
    return resp.json() if resp.content else None


@st.cache_data(ttl=30, show_spinner=False)
def get_providers():
    return api("GET", "/providers")


def get_documents():
    return api("GET", "/documents")


# ---------- sidebar: documents and settings ----------

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("📄 DocuSense")
    st.caption("Ask questions about your documents. Answers cite their sources.")

    uploads = st.file_uploader(
        "Add documents", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True
    )
    if uploads and st.button("Upload and index", type="primary", use_container_width=True):
        for f in uploads:
            with st.spinner(f"Indexing {f.name}..."):
                try:
                    info = api("POST", "/documents", files={"file": (f.name, f.getvalue())})
                    st.success(f"{info['filename']}: {info['num_chunks']} chunks")
                except RuntimeError as exc:
                    st.error(f"{f.name}: {exc}")

    docs = get_documents()
    st.subheader(f"Documents ({len(docs)})")
    labels = {d["doc_id"]: d["filename"] for d in docs}
    selected = st.multiselect(
        "Search in", options=list(labels), format_func=labels.get, placeholder="All documents"
    )
    for d in docs:
        col1, col2 = st.columns([5, 1])
        pages = f", {d['num_pages']} pages" if d.get("num_pages") else ""
        col1.caption(f"**{d['filename']}**  \n{d['num_words']:,} words{pages}")
        if col2.button("🗑", key=f"del-{d['doc_id']}", help="Remove document"):
            api("DELETE", f"/documents/{d['doc_id']}")
            st.rerun()

    st.divider()
    st.subheader("Settings")
    providers = get_providers()
    usable = [p["name"] for p in providers if p["available"]]
    provider = st.selectbox(
        "LLM",
        ["auto"] + [p["name"] for p in providers],
        format_func=lambda n: n if n == "auto" else f"{n} {'✓' if n in usable else '(unavailable)'}",
        help="auto picks the first available: Claude, OpenAI, Ollama, then the offline extractive mode.",
    )
    for p in providers:
        if p["name"] == provider and not p["available"]:
            st.warning(p["detail"])
    top_k = st.slider("Sources per answer", 1, 10, 5)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------- chat ----------

def render_answer(text: str) -> str:
    # Make [1] citations stand out
    return re.sub(r"\[(\d+)\]", r"**[\1]**", text)


def render_sources(citations, meta):
    if not citations:
        return
    with st.expander(f"Sources · {meta}"):
        for c in citations:
            where = f"{c['filename']}" + (f", page {c['page']}" if c.get("page") else "")
            mark = "🟢 cited" if c["cited"] else "⚪ retrieved"
            st.markdown(f"**[{c['index']}] {where}** · {mark}")
            snippet = c["text"] if len(c["text"]) < 700 else c["text"][:700] + "..."
            st.caption(snippet)


if not docs:
    st.info("Upload a PDF, Word doc, text or Markdown file in the sidebar to get started.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(render_answer(m["content"]) if m["role"] == "assistant" else m["content"])
        if m["role"] == "assistant":
            render_sources(m.get("citations", []), m.get("meta", ""))

if question := st.chat_input("Ask about your documents", disabled=not docs):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking..."):
            try:
                res = api(
                    "POST",
                    "/query",
                    json={
                        "question": question,
                        "doc_ids": selected or None,
                        "top_k": top_k,
                        "provider": provider,
                        "history": history,
                    },
                )
            except RuntimeError as exc:
                st.error(str(exc))
                st.stop()
        meta = f"{res['provider']} · {res['model']} · {res['latency_ms']} ms"
        st.markdown(render_answer(res["answer"]))
        for w in res.get("warnings", []):
            st.warning(w)
        render_sources(res["citations"], meta)
    st.session_state.messages.append(
        {"role": "assistant", "content": res["answer"], "citations": res["citations"], "meta": meta}
    )
