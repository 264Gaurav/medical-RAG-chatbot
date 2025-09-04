# RAG Pipeline — README

## Project overview

This repository implements a modular Retrieval-Augmented Generation (RAG) pipeline built for **local / on-premise** LLMs and embedding models. The pipeline covers:

- Knowledge base preparation (data ingestion → chunking → embedding → vector DB upsert)
- Prompt design and query embedding
- Retrieval (top‑k similarity search in Pinecone)
- Augmentation (system prompt + user query + retrieved context)
- Generation with a local LLM (LLama3.1 via Ollama)
- Dataset generation for evaluation (Q/A + context)
- Evaluation (retrieval metrics + generation metrics) with `ragas`, traced via `LangSmith`, and experiments tracked with `MLflow`
- Data versioning with `DVC`

---

## High-level architecture

1. **Data source(s)** — raw files (PDF, TXT, HTML, DOCX, DB exports, etc.)
2. **Data loader** — normalize text, remove boilerplate
3. **Chunking** — split into chunks (tokens/characters) with overlap
4. **Embedding** — Nomic local embed model (batching, dimension noted)
5. **Vector DB** — Pinecone (index creation, HNSW/ANN index behind the scenes)
6. **Retrieval** — query embedding → top‑k similarity search
7. **Augmentation** — build prompt: `system prompt` + `user query` + `retrieved context` (top‑k)
8. **Generation** — feed augmented prompt to LLM (llama3.1 via Ollama local); produce structured output
9. **Evaluation & tracking** — retrieval metrics (Part‑1) and generation metrics (Part‑2) logged in MLflow; LangSmith traces & ragas computes RAG metrics

---

## Prerequisites

- Python 3.10+
- Pinecone account & API key (or self-hosted Pinecone-compatible vector DB)
- Local Nomic embedding model set up and accessible (or equivalent local embedder)
- Ollama installed and running with Llama 3.1 model loaded for local inference
- DVC installed and initialized for data versioning
- MLflow server or local MLflow tracking
- LangSmith account and API key (for tracing LangChain flows)
- `ragas` installed for RAG evaluation

---

## Environment variables (example)

```bash
# Pinecone
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENV=your-pinecone-env
PINECONE_INDEX=my-rag-index

# Local embedding model (Nomic) — path or endpoint
NOMIC_EMBED_ENDPOINT=http://localhost:8000

# Ollama (LLM)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1

# LangSmith
LANGCHAIN_ENDPOINT = https://api.smith.langchain.com
LANGSMITH_TRACING = true
LANGSMITH_API_KEY=your-langsmith-key

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000    -> OR setup Dagshub for DVC

# General
CHUNK_SIZE=800           # tokens (recommended range: 256-1200 depending on docs)
CHUNK_OVERLAP=200        # tokens
EMBED_BATCH_SIZE=64
TOP_K=5
```

---

## Knowledge base preparation

### 1. Data loading

- Accept multiple file types (PDF, DOCX, HTML, TXT). Normalize text (remove headers/footers, cleanup whitespace and repeated boilerplate).

### 2. Chunking

- Recommended defaults: `CHUNK_SIZE=512..800 tokens`, `OVERLAP=128..256 tokens` for many corpora.
- Keep chunk sizes small enough so retrieved contexts fit LLM input budget when concatenated with system prompt.
- Store metadata per chunk: `source_id`, `document_id`, `chunk_id`, `start_char`, `end_char`, `page_number` (if available), `embedding_status`.

### 3. Embedding

- Use the local Nomic embed model; batch for speed (`EMBED_BATCH_SIZE`, e.g. 32–128).
- Persist raw embeddings and metadata locally (parquet/JSONL) before upsert to Pinecone so DVC can version them.

### 4. Upsert to Pinecone

- Create index with appropriate vector dimension and metric (cosine or dot product depending on embedder).
- Upsert in batches. Include metadata so retrieval results are traceable to original sources.

Example upsert pseudocode (Python):

```python
# embedder.py (pseudocode)
def embed_documents(docs: List[str]) -> np.ndarray:
    # call local nomic embedder in batches
    return embeddings

# indexer.py
def upsert_to_pinecone(embeddings, metadatas, ids):
    # pinecone.upsert(vectors=list(zip(ids, embeddings, metadatas)))
    pass
```

---

## Prompt design

- Split prompt into two parts: `system prompt` (instructions, expected output structure, constraints) and `user prompt` (user question).
- When adding retrieved context, follow a stable schema — e.g., `CONTEXT (1/k): <text> — source: <doc_id>` — to help LLM cite sources.
- Limit total context length; prefer top‑k with `k` tuned (common values: 3–10).

Example prompt scaffold:

```
SYSTEM: You are an assistant that answers using only the provided context. Output in JSON with keys: answer, sources.

USER: <user question>

CONTEXT START
---
[1] <text chunk 1> (source: docA#chunk3)
[2] <text chunk 2> (source: docB#chunk4)
...
CONTEXT END

Please answer now.
```

---

## Retrieval

- Embed user query using the **same** embedder (Nomic) and same preprocessing.
- Execute Pinecone `query` with `top_k` and optionally `filter` on metadata.
- Use HNSW/ANN search parameters when creating the index; tune ef/ef_construction if configurable.
- Return `top_k` chunks ranked with scores and metadata.

---

## Augmentation & Generation

- Build augmented prompt from `system prompt`, `user query`, and retrieved `top_k` context.
- Send to local Ollama LLM (llama3.1). Use temperature and max_tokens tuned for structured output.
- Enforce output format in `system prompt` (JSON/markdown table) so downstream parsing is deterministic.

Example generation call (pseudocode):

```python
from ollama_client import OllamaClient
client = OllamaClient(host=os.getenv('OLLAMA_HOST'))
resp = client.generate(model=os.getenv('OLLAMA_MODEL'), prompt=augmented_prompt, temperature=0.0, max_tokens=512)
```

---

## Dataset generation pipeline

1. Use an LLM-based Q/A generator to create `question` + `reference answer` pairs from documents.
2. For each generated question, run your retrieval pipeline to collect `top_k` context and the model's generated answer.
3. Save records as `(question, reference_answer, retrieved_context, generated_answer, metadata)` for use in evaluation and for `ragas` ground-truth comparison.

This produces a `qas` dataset that enables automated RAG evaluation.

---

## Evaluation pipeline (two-part separation)

**Part‑1 — Retrieval metrics**

**Part‑2 — Generation metrics**

- Use `ragas` to compute standard RAG metrics comparing generated answers ➜ ground-truth; store ragas outputs as artifacts

**Logging:** Send both metric sets and artifacts to MLflow. Use separate MLflow runs for retrieval and generation evaluations or tag runs accordingly.

---

## Tooling integration (quick notes)

- **DVC**: track large files (processed chunks, embeddings, index metadata). Use `dvc add data/processed/*` and commit `dvc.lock` and pipelines.
- **MLflow**: track experiments, params, metrics, and artifacts (model outputs, ragas reports). Use `mlflow.set_experiment(...)` and `mlflow.log_*` in evaluation scripts.
- **LangSmith**: instrument LangChain chains (or custom pipeline instrumentation) to record chain runs, inputs/outputs for debugging.
- **ragas**: run evaluation using your `(question, reference_answer, retrieved_context, generated_answer)` dataset. Save ragas report and log to MLflow.

---

## Configuration & hyperparameters (suggested defaults)

- CHUNK_SIZE: 512–800 tokens
- CHUNK_OVERLAP: 128–256
- EMBED_BATCH_SIZE: 32–128
- TOP_K (retrieval): 3–8 (tune)
- LLM temperature: 0.0–0.2 for deterministic / factual outputs
- Max generation tokens: depends on downstream needs (recommend <= 1024 for Llama3.1 local)

Tune these using MLflow experiments.

---

## Metrices for evaluation of RAG pipeline using RAGAS

1. Context Precision

2. Context Recall

3. Context Entities Recall

4. Noise Sensitivity

5. Response Relevancy

6. Faithfulness

7. Multimodal Faithfulness

8. Multimodal Relevance

---

## Security, privacy, and compliance

- Sanitize sensitive PII before indexing (or store reducted versions). Keep raw data access restricted.
- Use metadata-driven filters to exclude confidential documents from production retrievals.
- Secure API keys (Pinecone, LangSmith) via environment variables and secrets manager.

---

## Troubleshooting & tips

- If retrieval returns low-quality results: try increasing `TOP_K`, adjust chunk size/overlap, or tune indexing metric (cosine vs dot).
- If LLM hallucinates: lower temperature, constrain output format in system prompt, and reduce irrelevant context by stricter source filtering.
- Monitor pipeline bottlenecks: embedding throughput (batch size), Pinecone upsert latency, LLM generation time.

---

## Contributing

- Follow the repo style guides. Add tests for new functionality. Open PRs against `main` with clear descriptions and MLflow experiment ids if applicable.

---

## Appendix — Quick commands

```bash
pip install -r requirements.txt
# DVC
dvc init
dvc add data/

# MLflow (example to run experiments locally) ## or setup MLFlow with dagshub
mlflow ui --port 5000

# Run the notebook -> ragas.ipynb cell by cell

```

---
