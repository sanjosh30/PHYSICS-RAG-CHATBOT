# ============================================================
# Physics RAG Chatbot — Dockerfile
# Multi-stage build optimised for production deployment.
#
# Build:  docker build -t physics-rag-chatbot .
# Run:    docker-compose up
# ============================================================

# ── Stage 1: Builder (install Python deps) ─────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps for PyMuPDF and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ───────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Runtime system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source code
COPY src/ ./src/
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/benchmark/ ./data/benchmark/
COPY .env.example .env.example

# Create directories
RUN mkdir -p logs data/raw/feynman data/raw/openstax data/processed vectorstore/chroma_db

# Streamlit config
RUN mkdir -p /root/.streamlit
RUN echo '\
[general]\n\
email = ""\n\
\n\
[server]\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
port = 8501\n\
\n\
[theme]\n\
base = "dark"\n\
primaryColor = "#63b3ed"\n\
backgroundColor = "#0a0a1a"\n\
secondaryBackgroundColor = "#0d1117"\n\
textColor = "#e2e8f0"\n\
' > /root/.streamlit/config.toml

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default command: run Streamlit app
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
