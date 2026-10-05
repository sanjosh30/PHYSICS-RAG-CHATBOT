# ⚛️ Physics RAG Chatbot

> A production-grade Retrieval-Augmented Generation chatbot for undergraduate physics.  
> Every answer is **grounded in cited passages** from The Feynman Lectures on Physics and OpenStax University Physics — zero hallucinations tolerated.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-green.svg)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.6-purple.svg)](https://chromadb.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45-red.svg)](https://streamlit.io)
[![Ollama](https://img.shields.io/badge/LLM-Llama_3.1-orange.svg)](https://ollama.ai)

---

## 📐 Architecture

```
Physics PDFs (6 volumes)
        │
        ▼
Document Ingestion
  ├── PyMuPDF — text + metadata extraction
  ├── Text Cleaner — remove headers, captions, artifacts
  └── Chunker — RecursiveCharacterTextSplitter (512 tok / 64 overlap)
        │
        ▼
Embeddings: BAAI/bge-small-en-v1.5 (384-dim, local)
        │
        ▼
ChromaDB (persistent, cosine similarity)
        │
   ┌────┴────┐
   │         │
   ▼         ▼
 MMR Search  Metadata Filter
   (diversity)  (feynman | openstax)
        │
        ▼
Top-10 Candidates
        │
        ▼
Cross-Encoder Reranker → Top-3 Docs + Confidence Score
(cross-encoder/ms-marco-MiniLM-L-6-v2)
        │
        ▼
Out-of-scope check (confidence < threshold → refuse)
        │
        ▼
Prompt Builder → [System + Context + Chat History + Question]
        │
        ▼
Ollama Llama 3.1 (local, streaming)
        │
        ▼
Citation Engine → [Source N] → book, chapter, page, snippet
        │
        ▼
Streamlit UI
  ├── Chat history
  ├── Expandable citation cards
  ├── Colour-coded confidence meter
  └── Out-of-scope refusal display
```

---

## 📚 Corpus

| Book | Volumes | Format | Source |
|------|---------|--------|--------|
| The Feynman Lectures on Physics | Vol 1 (Mechanics), Vol 2 (E&M), Vol 3 (QM) | PDF | [feynmanlectures.caltech.edu](https://www.feynmanlectures.caltech.edu/) |
| OpenStax University Physics | Vol 1 (Mechanics), Vol 2 (Thermodynamics & Waves), Vol 3 (Optics & Modern) | PDF | [openstax.org](https://openstax.org/subjects/science) |

**Total corpus:** ~6 PDFs, ~250 MB raw, ~50,000 chunks after processing.

---

## 🚀 Quickstart (Local)

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- ~8 GB RAM (for Llama 3.1)

### 1. Clone and install

```bash
git clone <repo-url>
cd physics-rag-chatbot

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env if needed (defaults work out-of-the-box with Ollama)
```

### 3. Pull the Llama 3.1 model

```bash
ollama pull llama3.1
```

### 4. Add your PDFs

Place PDFs in:
```
data/raw/feynman/    ← Feynman Lectures Vol 1, 2, 3
data/raw/openstax/   ← OpenStax University Physics Vol 1, 2, 3
```

### 5. Build the vector index

```bash
python scripts/ingest.py
```

This will:
- Parse all 6 PDFs with PyMuPDF
- Clean and chunk the text (~50,000 chunks)
- Embed with BGE-small (384-dim)
- Store in ChromaDB at `vectorstore/chroma_db/`

⏱️ Estimated time: **5–15 minutes** (CPU), **1–3 minutes** (GPU)

### 6. Run the chatbot

```bash
python -m streamlit run app/streamlit_app.py
```

Open http://localhost:8501 in your browser.

---

## 🐳 Docker Deployment

```bash
# Start both Ollama + Streamlit app
docker-compose up -d

# Build the vector index (first time only)
docker-compose exec app python scripts/ingest.py

# View logs
docker-compose logs -f app
```

The app is available at http://localhost:8501.

---

## 🧪 Evaluation

### Run the benchmark

> **Windows users:** Prefix commands with `$env:PYTHONUTF8="1";` to enable emoji output in the terminal, or run from Windows Terminal which supports UTF-8 by default.

```bash
# Run 20-question benchmark
python scripts/evaluate.py

# Include LLM-as-judge hallucination check
python scripts/evaluate.py --hallucination-check
```

### Targets & Results

| Metric | Target | Result | Description |
|--------|--------|--------|-------------|
| Citation Accuracy | ≥ 85% | **88.9%** | Retrieved passages contain expected physics keywords |
| Hallucination Rate | < 10% | **0.0%** | Answers not contradicted by retrieved context |
| OOS Refusal Rate | ≥ 90% | **100.0%** | Out-of-scope questions correctly refused |

### Run unit tests

```bash
pytest tests/ -v --tb=short
```

---

## 📁 Project Structure

```
physics-rag-chatbot/
├── app/
│   └── streamlit_app.py          # Streamlit chat UI
├── src/
│   ├── ingestion/
│   │   ├── pdf_loader.py         # PyMuPDF loader with metadata
│   │   ├── text_cleaner.py       # Noise removal
│   │   ├── chunker.py            # RecursiveCharacterTextSplitter
│   │   └── build_index.py        # Pipeline orchestrator
│   ├── embeddings/
│   │   └── embedding_model.py    # BGE-small-en-v1.5
│   ├── retrieval/
│   │   ├── vector_store.py       # ChromaDB wrapper
│   │   ├── retriever.py          # MMR retriever
│   │   └── reranker.py           # Cross-encoder reranker
│   ├── generation/
│   │   ├── prompts.py            # System prompt + context formatter
│   │   ├── rag_chain.py          # Main RAG chain (Ollama/OpenAI)
│   │   └── citation_engine.py    # [Source N] → metadata mapper
│   ├── evaluation/
│   │   ├── benchmark.py          # 20-Q benchmark runner
│   │   ├── metrics.py            # Citation acc / hallucination / refusal
│   │   └── hallucination_test.py # LLM-as-judge
│   └── utils/
│       ├── config.py             # Pydantic settings
│       ├── logger.py             # Loguru rotating logs
│       └── helpers.py            # Shared utilities
├── data/
│   ├── raw/feynman/              # Feynman Lectures PDFs
│   ├── raw/openstax/             # OpenStax PDFs
│   └── benchmark/questions.json # 20 test questions
├── vectorstore/chroma_db/        # ChromaDB (git-ignored)
├── logs/                         # App, retrieval, evaluation logs
├── tests/                        # pytest test suite
├── scripts/
│   ├── ingest.py                 # Run ingestion pipeline
│   ├── evaluate.py               # Run benchmark
│   └── rebuild_index.py          # Wipe + rebuild index
├── Dockerfile                    # Multi-stage Docker build
├── docker-compose.yml            # Ollama + Streamlit services
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Configuration

All settings are in `.env` (copied from `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BACKEND` | `ollama` | `ollama` or `openai` |
| `OLLAMA_MODEL` | `llama3.1` | Any model available in Ollama |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `RETRIEVAL_TOP_K` | `10` | Candidates before reranking |
| `RERANK_TOP_K` | `3` | Final docs sent to LLM |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `OUT_OF_SCOPE_THRESHOLD` | `0.35` | Min reranker score to answer |

---

## 🛠 Development

```bash
# Rebuild index from scratch
python scripts/rebuild_index.py --yes

# Run specific test file
pytest tests/test_citations.py -v

# Check logs
tail -f logs/app.log
tail -f logs/retrieval.log
```

---

## 📄 License

MIT License. Corpus content (Feynman Lectures, OpenStax) is open-access under their respective licenses.
