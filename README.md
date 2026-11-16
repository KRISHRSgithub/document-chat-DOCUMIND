# DocuMind – PDF Question-Answering Chatbot (RAG) | Python, LangChain, FAISS, BM25, Streamlit, Ollama, Gemini 

• Built a retrieval-augmented (RAG) chatbot that answers questions from multiple PDFs with page-level citations, using hybrid retrieval: FAISS semantic search       plus BM25 keyword search merged by reciprocal rank fusion.

• Added a similarity-based guard that refuses questions the documents cannot answer (2/2 off-topic test questions refused), a confidence indicator, and rewriting    of follow-up questions using chat history.

• Wrote an evaluation script and pytest tests; on a hand-written 11-question test set, 9/9 answers included page citations and average keyword recall was 0.89       with a local Llama 3.2 model.

• Built a Streamlit interface with multi-PDF filtering, summary and quiz generation, and switchable local (Ollama) or cloud (Gemini) LLMs; documented setup and      results in the README with a demo video.

# DocuMind: chat with your PDFs (Python + LangChain)

A RAG chatbot that answers questions from one or many PDFs, cites page numbers, refuses to guess,
and doubles as a study tool.

## Features
1. **Hybrid retrieval**: FAISS (semantic) + BM25 (keyword) merged with reciprocal rank fusion.
2. **Page-level citations**: answers say `[file.pdf p.3]`; the UI shows source snippets.
3. **Hallucination guard**: weak retrieval means the LLM is never called and the bot says it can't find the answer.
4. **Confidence badge**: High / Medium / Low from cosine similarity.
5. **Follow-up aware**: "explain that simpler" is rewritten into a standalone query using chat history.
6. **Multi-PDF** with a "search only in..." filter.
7. **Answer styles**: Standard, ELI5, Exam mode, Interview mode.
8. **Summary** (TL;DR, key points, key terms) and **Quiz mode** (MCQs, scoring, explanations).
9. **Save / load index** so large PDFs are embedded only once.
10. **Provider-agnostic**: Ollama (free, local), OpenAI or Gemini.
11. **Export chat** to Markdown.

## Project structure
```
documind/
├── app.py              # Streamlit UI (chat, summary, quiz, save/load index)
├── core.py             # RAG engine: ingestion, hybrid search, guard, summary, quiz
├── requirements.txt
├── .env.example        # copy to .env and add keys
├── .gitignore / .dockerignore
├── Dockerfile
├── data/sample.pdf     # 3-page demo PDF (databases) used by the eval and tests
├── index/              # saved FAISS index (created by the "Save index" button)
├── eval/
│   ├── questions.json  # in-scope and out-of-scope test questions
│   └── run_eval.py     # Hit@k, guard accuracy, keyword recall, citation rate
├── tests/test_core.py  # pytest unit tests
└── assets/             # put demo.gif / screenshots here for the README
```

## Run
```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # only needed for OpenAI / Gemini
ollama pull llama3.2                                 # free local option
streamlit run app.py                                 # http://localhost:8501
```
Try it with `data/sample.pdf`, for example: "Who proposed the relational model?", "Explain 3NF",
then a follow-up like "give an example of that". Ask "What is the capital of France?" to see the guard.

## Test and evaluate
```bash
pip install pytest && pytest -q                       # unit tests
python eval/run_eval.py                               # retrieval-only, no LLM needed
python eval/run_eval.py --provider ollama             # adds keyword recall + citation rate
```
Results are written to `eval/results.json`. Report numbers like Hit@3 on your own PDFs in your resume or README.
If the guard refuses valid questions (or answers off-topic ones), tune `GUARD_THRESHOLD` in `core.py`.

## Docker
```bash
docker build -t documind .
docker run -p 8501:8501 --env-file .env documind
```
For Ollama on the host machine, add `-e OLLAMA_BASE_URL=http://host.docker.internal:11434`.

## Architecture
```
PDFs -> PyPDFLoader -> splitter -> FAISS + BM25
question -> rewrite (history) -> hybrid search -> RRF -> guard -> prompt -> LLM (stream) -> answer + sources
```

## Ideas to extend
OCR for scanned PDFs, a cross-encoder reranker, RAGAS metrics, user accounts, a deployed demo link.

## Evaluation
Hand-written test set: 9 in-scope and 2 off-topic questions on a 3-page sample PDF.

| Metric | Result |
|---|---|
| Hit@1 (right page ranked first) | 9/9 |
| Hit@3 | 9/9 |
| Guard accuracy (off-topic refused) | 2/2 |
| Avg keyword recall (Llama 3.2 via Ollama) | 0.89 |
| Answers with page citation | 9/9 |

Limitations: small test set and a short sample PDF, so these numbers show the pipeline works but are not a benchmark.

Hit@1: 9/9 (one question label corrected after checking the PDF; 8/9 on first run)
Hit@3: 9/9
Guard accuracy: 2/2
Avg keyword recall (Llama 3.2, Ollama): 0.89
Citation rate: 9/9
Test set: hand-written, 9 in-scope + 2 off-topic, on a 3-page sample PDF
