import os
import tempfile

import streamlit as st

from core import INDEX_DIR, STYLES, DocuMind, load_embeddings

st.set_page_config(page_title="DocuMind", page_icon="📚", layout="wide")
st.title("📚 DocuMind — chat with your PDFs")


@st.cache_resource(show_spinner="Loading embedding model (first run only)…")
def embeddings():
    return load_embeddings()


ss = st.session_state
ss.setdefault("history", [])
ss.setdefault("engine", None)
ss.setdefault("quiz", None)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    provider = st.selectbox("LLM provider", ["ollama", "openai", "gemini"],
                            help="Ollama is free & local. OpenAI/Gemini need API keys in env vars.")
    style = st.selectbox("Answer style", list(STYLES))
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if st.button("⚙️ Process documents", use_container_width=True) and files:
        eng = DocuMind(embeddings(), provider)
        with st.spinner("Reading and indexing…"):
            for f in files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as t:
                    t.write(f.getvalue())
                    path = t.name
                try:
                    n = eng.add_pdf(path, f.name)
                    st.success(f"{f.name}: {n} chunks")
                except Exception as e:
                    st.error(f"{f.name}: {e}")
                finally:
                    os.remove(path)
        ss.engine, ss.history, ss.quiz = eng, [], None

    if (INDEX_DIR / "index.faiss").exists() and st.button("📂 Load saved index", use_container_width=True):
        saved = DocuMind(embeddings(), provider)
        saved.load()
        ss.engine, ss.history, ss.quiz = saved, [], None

    eng: DocuMind | None = ss.engine
    only = []
    if eng and eng.chunks:
        eng.set_provider(provider)
        if st.button("💾 Save index", use_container_width=True):
            eng.save()
            st.toast("Index saved to ./index")
        only = st.multiselect("Search only in…", eng.sources, default=eng.sources)

    if ss.history:
        md = "\n\n".join(f"**{m['role'].title()}:** {m['content']}" for m in ss.history)
        st.download_button("⬇️ Export chat (.md)", md, "chat.md", use_container_width=True)
        if st.button("🗑️ Clear chat", use_container_width=True):
            ss.history = []
            st.rerun()

if not (eng and eng.chunks):
    st.info("Upload one or more PDFs in the sidebar and click **Process documents**.")
    st.stop()

chat_tab, sum_tab, quiz_tab = st.tabs(["💬 Chat", "📝 Summary", "🎯 Quiz me"])


def show_sources(sources, label, conf):
    icon = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}[label]
    st.caption(f"{icon} Confidence: **{label}** ({conf:.2f})")
    if sources:
        with st.expander("Sources"):
            for d in sources:
                st.markdown(f"**{d.metadata['source']} — page {d.metadata['page']}**")
                st.caption(d.page_content[:350] + "…")


# --------------------------------------------------------------------- chat
with chat_tab:
    for m in ss.history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("label"):
                show_sources(m["sources"], m["label"], m["conf"])

    if q := st.chat_input("Ask something about your documents…"):
        with st.chat_message("user"):
            st.markdown(q)
        try:
            res = eng.ask(q, ss.history, style, only or None)
            with st.chat_message("assistant"):
                answer = st.write_stream(res.stream)
                show_sources(res.sources, res.label, res.confidence)
            ss.history += [
                {"role": "user", "content": q},
                {"role": "assistant", "content": answer, "sources": res.sources,
                 "label": res.label, "conf": res.confidence},
            ]
        except Exception as e:
            st.error(f"LLM error: {e}")

# ------------------------------------------------------------------ summary
with sum_tab:
    if st.button("Summarise selected documents"):
        st.write_stream(eng.summarize(only or None))

# --------------------------------------------------------------------- quiz
with quiz_tab:
    c1, c2 = st.columns([3, 1])
    topic = c1.text_input("Topic (optional)", placeholder="e.g. normalization, OSI model")
    n = c2.slider("Questions", 3, 10, 5)
    if st.button("Generate quiz"):
        try:
            with st.spinner("Writing questions…"):
                ss.quiz = eng.quiz(n, topic, only or None)
        except Exception as e:
            st.error(str(e))

    if ss.quiz:
        picks = [st.radio(f"**{i+1}. {q['question']}**", q["options"], index=None, key=f"q{i}")
                 for i, q in enumerate(ss.quiz)]
        if st.button("Check answers"):
            score = 0
            for q, p in zip(ss.quiz, picks):
                right = q["options"][q["answer"]]
                ok = p == right
                score += ok
                (st.success if ok else st.error)(f"{'✅' if ok else '❌'} {right} — {q['explanation']}")
            st.metric("Score", f"{score} / {len(ss.quiz)}")
