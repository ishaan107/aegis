# 🔍 Project Aegis — Advanced Enterprise RAG System

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-compatible-green)](https://langchain.com)
[![Pinecone](https://img.shields.io/badge/Vector_DB-Pinecone-purple)](https://pinecone.io)
[![Gemini](https://img.shields.io/badge/LLM-Gemini_Pro-orange?logo=google)](https://ai.google.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A production-grade, multi-stage Retrieval-Augmented Generation (RAG) system engineered for high-accuracy document QA over complex corporate policy documents. Built with a 5-step intelligent retrieval pipeline including query expansion, HyDE, metadata filtering, version-aware post-filtering, and cross-encoder reranking.

---

## 📌 Project Overview

Standard RAG systems retrieve chunks by naive semantic similarity — resulting in poor accuracy when users ask vague questions or when documents share overlapping vocabulary.

**Project Aegis** solves this with a fully orchestrated retrieval pipeline:

```
User Query
    │
    ▼
┌─────────────────────────────┐
│  1. Query Transformation    │  Multi-query expansion + HyDE
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  2. Intent Detection        │  LLM-powered doc_type router → pre-filter
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  3. Broad Retrieval         │  Top-25 per query variant from Pinecone
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  4. Post-Filter             │  Drop stale document versions
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  5. Cross-Encoder Reranking │  Score (query, chunk) pairs → Top-5
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  6. Grounded Generation     │  Gemini Pro answers from Top-5 only
└─────────────────────────────┘
```

---

## 🧠 Key Technical Decisions

### Chunking — Sliding Window over Sentences
Rather than arbitrary character splits, documents are split at sentence boundaries then grouped into overlapping sliding windows (`window=5, step=4`). This guarantees:
- No concept is ever cut mid-sentence
- A 1-sentence overlap acts as a semantic bridge between adjacent chunks

### Keyword Extraction — TF-IDF over Raw Frequency
Each chunk carries 8 TF-IDF keywords as metadata. TF-IDF down-weights terms common across *all* documents and surfaces terms that uniquely identify *this* chunk — making metadata filtering far more discriminative than raw term frequency.

### Retrieval — Bi-Encoder + Cross-Encoder Two-Stage Pipeline
| Stage | Model | Speed | Accuracy |
|---|---|---|---|
| Broad retrieval | `all-MiniLM-L6-v2` (bi-encoder) | Fast | Moderate |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Slower | High |

The cross-encoder reads the query and each chunk *together*, enabling full attention over both texts. This catches semantic matches that pure embedding cosine similarity misses.

### Query Transformation
Two techniques run in parallel before any retrieval:
- **Multi-Query Expansion**: Gemini rewrites the query in 3 different styles (formal, procedural, keyword-heavy)
- **HyDE**: Gemini generates a *hypothetical* policy-like answer; its embedding often matches real chunks better than the raw question embedding

---

## 🗂️ Repository Structure

```
advanced-rag-project/
├── M9_ASSIGNMENT.ipynb       # Full pipeline notebook (Colab-ready)
├── rag_pipeline/
│   ├── __init__.py
│   ├── chunker.py            # Sentence splitting + sliding window
│   ├── embedder.py           # SentenceTransformer wrapper
│   ├── indexer.py            # Pinecone upsert logic
│   ├── retriever.py          # Multi-query + HyDE retrieval
│   ├── reranker.py           # Cross-encoder scoring
│   └── generator.py          # Gemini prompt + answer
├── data/                     # Place your .txt policy documents here
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Setup & Usage

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/advanced-rag-project.git
cd advanced-rag-project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
# Add your GOOGLE_API_KEY and PINECONE_API_KEY to .env
```

### 4. Run on Google Colab (recommended)
Open `M9_ASSIGNMENT.ipynb` in Colab, set your secrets (`gemini`, `pinecone`), mount Google Drive with your `.txt` documents in `/data`, and run all cells.

### 5. Ask a question
```python
from rag_pipeline import answer

answer("What is the travel reimbursement policy for international trips?")
```

---

## 🔬 Pipeline Deep-Dive

### Step 1 — Query Expansion + HyDE
```python
query_variants = expand_query(user_query)     # 4 variants via Gemini
hyde_passage   = generate_hyde_passage(query) # 1 hypothetical passage
search_texts   = query_variants + [hyde_passage]  # 5 total search vectors
```

### Step 2 — Intent-Aware Pre-Filtering
```python
inferred_type = detect_doc_type_intent(query)
# → "travel" | "security" | "training" | "work policies" | None
```
If a category is detected, Pinecone filters are applied *before* vector search — eliminating entire irrelevant document sets.

### Step 3 — Broad Retrieval (Top-25 per variant)
All 5 search texts are embedded and queried against Pinecone independently. Results are pooled, deduplicated (max score wins), giving ~50-80 candidate chunks.

### Step 4 — Post-Filtering (Version Staleness)
Regex detects version suffixes (`_v2`, `_2024`) in filenames and retains only the most recent version's chunks.

### Step 5 — Cross-Encoder Reranking (Top-5)
```python
pairs  = [(query, chunk.text) for chunk in candidates]
scores = cross_encoder.predict(pairs)
top_5  = sorted(zip(scores, candidates))[-5:]
```

### Step 6 — Grounded Generation
Gemini Pro is instructed to answer *only* from the Top-5 chunks, with explicit source attribution in the context block.

---

## 📊 Performance Characteristics

| Metric | Naive RAG | Project Aegis |
|---|---|---|
| Query types handled | Exact match only | Vague, multi-intent, typos |
| Context window pollution | High | Low (Top-5 only) |
| Multi-version handling | None | Version-aware post-filter |
| Irrelevant doc cross-talk | Common | Eliminated by pre-filter |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | Google Gemini Pro (`gemini-1.5-flash`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Vector DB | Pinecone (Serverless, AWS us-east-1) |
| Runtime | Python 3.10+, Google Colab |

---

## 📄 License

MIT — see [LICENSE](LICENSE)
