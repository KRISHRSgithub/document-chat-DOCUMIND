"""DocuMind core: a hybrid-RAG engine over one or many PDFs (LangChain)."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()  # reads API keys / model names from a local .env file if present

# ----------------------------------------------------------------- settings
INDEX_DIR = Path(__file__).parent / "index"
GUARD_THRESHOLD = 0.18   # below this similarity -> "not in the documents"
HIGH, MEDIUM = 0.50, 0.30  # confidence labels (tune for your embedding model)

STYLES = {
    "Standard": "Answer clearly and concisely.",
    "ELI5": "Explain like I'm 12: simple words and one everyday analogy.",
    "Exam mode": "Write a crisp exam answer: definition, 3-5 key points, one example.",
    "Interview mode": "Answer like in a technical interview: direct, structured, mention trade-offs.",
}


def load_embeddings() -> HuggingFaceEmbeddings:
    """Free, local embeddings (no API key). Normalised so we can read cosine similarity."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )


def get_llm(provider: str, temperature: float = 0.2):
    """Provider-agnostic LLM. Ollama = free & local; OpenAI / Gemini need API keys in env."""
    if provider == "none":  # retrieval-only (used by eval/tests, no LLM needed)
        return None
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=temperature)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"), temperature=temperature
        )
    raise ValueError(f"Unknown provider: {provider}")


# ------------------------------------------------------------------ prompts
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are DocuMind, a precise document assistant. Answer ONLY from the context below. "
     "Cite every claim like [file.pdf p.3]. If the context does not contain the answer, say so "
     "instead of guessing.\nStyle: {style}"),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

REWRITE_PROMPT = ChatPromptTemplate.from_template(
    "Rewrite the last question as a standalone search query using the chat history. "
    "Return ONLY the rewritten question.\n\nHistory:\n{history}\n\nLast question: {question}"
)

SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    "Summarise the document excerpts below.\nFormat:\n**TL;DR** (2 sentences)\n"
    "**Key points** (5-7 bullets)\n**Key terms** (5 terms with one-line definitions)\n\n{context}"
)

QUIZ_PROMPT = ChatPromptTemplate.from_template(
    "From the excerpts below, write {n} multiple-choice questions that test understanding.\n"
    "Return ONLY a JSON list. Each item: {{\"question\": str, \"options\": [4 strings], "
    "\"answer\": index 0-3, \"explanation\": str}}.\n\n{context}"
)


@dataclass
class Result:
    stream: Iterator[str]
    sources: list          # LangChain Documents used as evidence
    confidence: float      # best cosine similarity
    label: str             # High / Medium / Low
    standalone: str        # the rewritten question actually searched


class DocuMind:
    def __init__(self, embeddings, provider: str = "ollama"):
        self.emb = embeddings
        self.set_provider(provider)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        self.chunks: list = []
        self.vs: FAISS | None = None
        self.bm25: BM25Retriever | None = None

    def set_provider(self, provider: str):
        self.llm = get_llm(provider)

    # ---------------------------------------------------------- ingestion
    @property
    def sources(self) -> list[str]:
        return sorted({c.metadata["source"] for c in self.chunks})

    def add_pdf(self, path: str, display_name: str) -> int:
        pages = PyPDFLoader(path).load()
        for p in pages:
            p.metadata["source"] = display_name
            p.metadata["page"] = int(p.metadata.get("page", 0)) + 1  # 1-indexed for humans
        new = self.splitter.split_documents(pages)
        if not new:
            raise ValueError(f"No extractable text in {display_name} (scanned PDF? try OCR).")
        self.chunks += new
        if self.vs is None:
            self.vs = FAISS.from_documents(new, self.emb)
        else:
            self.vs.add_documents(new)
        self.bm25 = BM25Retriever.from_documents(self.chunks)
        return len(new)

    # -------------------------------------------------------- persistence
    def save(self, folder: Path = INDEX_DIR) -> None:
        """Save the FAISS index so big PDFs don't need re-embedding next time."""
        Path(folder).mkdir(exist_ok=True)
        self.vs.save_local(str(folder))

    def load(self, folder: Path = INDEX_DIR) -> None:
        # allow_dangerous_deserialization is fine here: we only load an index we saved ourselves.
        self.vs = FAISS.load_local(str(folder), self.emb, allow_dangerous_deserialization=True)
        self.chunks = list(self.vs.docstore._dict.values())
        self.bm25 = BM25Retriever.from_documents(self.chunks)

    # ---------------------------------------------------------- retrieval
    @staticmethod
    def _key(d):
        return (d.metadata["source"], d.metadata["page"], d.page_content[:80])

    def _search(self, query: str, k: int, only: list[str] | None):
        """Hybrid search = FAISS (meaning) + BM25 (exact keywords), merged by reciprocal rank fusion."""
        fetch = k * 4
        vec = self.vs.similarity_search_with_score(query, k=fetch)
        self.bm25.k = fetch
        kw = self.bm25.invoke(query)
        ok = lambda d: not only or d.metadata["source"] in only
        vec = [(d, s) for d, s in vec if ok(d)]
        kw = [d for d in kw if ok(d)]

        scores, docs = {}, {}
        for lst in ([d for d, _ in vec], kw):
            for rank, d in enumerate(lst):
                key = self._key(d)
                scores[key] = scores.get(key, 0) + 1 / (60 + rank + 1)
                docs[key] = d
        top = [docs[x] for x in sorted(scores, key=scores.get, reverse=True)[:k]]
        # FAISS returns squared L2; for unit vectors cosine = 1 - d/2
        best = max((1 - s / 2 for _, s in vec), default=0.0)
        return top, best

    def _rewrite(self, question: str, history: list[dict]) -> str:
        if not history:
            return question
        hist = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
        chain = REWRITE_PROMPT | self.llm | StrOutputParser()
        return chain.invoke({"history": hist, "question": question}).strip() or question

    # ----------------------------------------------------------------- QA
    def ask(self, question: str, history: list[dict], style: str = "Standard",
            only: list[str] | None = None, k: int = 5) -> Result:
        standalone = self._rewrite(question, history)
        docs, conf = self._search(standalone, k, only)
        label = "High" if conf >= HIGH else "Medium" if conf >= MEDIUM else "Low"

        if conf < GUARD_THRESHOLD or not docs:  # hallucination guard: don't even call the LLM
            msg = "I couldn't find this in the uploaded documents, so I won't guess."
            return Result(iter([msg]), [], conf, "Low", standalone)

        context = "\n\n".join(
            f"[{d.metadata['source']} p.{d.metadata['page']}]\n{d.page_content}" for d in docs
        )
        chain = QA_PROMPT | self.llm | StrOutputParser()
        stream = chain.stream({"style": STYLES[style], "context": context, "question": question})
        return Result(stream, docs, conf, label, standalone)

    # ------------------------------------------------- study-tool features
    def _sample(self, only: list[str] | None, n: int = 12, topic: str = "") -> str:
        if topic:
            docs, _ = self._search(topic, n, only)
        else:
            pool = [c for c in self.chunks if not only or c.metadata["source"] in only]
            step = max(1, len(pool) // n)
            docs = pool[::step][:n]
        return "\n\n".join(f"[{d.metadata['source']} p.{d.metadata['page']}]\n{d.page_content}" for d in docs)

    def summarize(self, only: list[str] | None = None) -> Iterator[str]:
        return (SUMMARY_PROMPT | self.llm | StrOutputParser()).stream(
            {"context": self._sample(only)}
        )

    def quiz(self, n: int = 5, topic: str = "", only: list[str] | None = None) -> list[dict]:
        raw = (QUIZ_PROMPT | self.llm | StrOutputParser()).invoke(
            {"n": n, "context": self._sample(only, topic=topic)}
        )
        match = re.search(r"\[.*\]", raw, re.S)  # models sometimes wrap JSON in prose / fences
        if not match:
            raise ValueError("Model did not return valid JSON. Try again or use a stronger model.")
        return json.loads(match.group(0))
