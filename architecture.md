# System Architecture — Project Aegis

## Overview

Project Aegis is designed as a modular, stage-separated RAG pipeline where each stage can be independently swapped, tuned, or benchmarked.

---

## Component Map

```
┌──────────────────────────────────────────────────────────┐
│                     INGESTION LAYER                      │
│                                                          │
│  .txt files  →  Sentence Splitter  →  Sliding Window    │
│                      Chunker            (w=5, step=4)    │
│                          │                               │
│                    TF-IDF Keywords                       │
│                    (8 per chunk)                         │
│                          │                               │
│                  all-MiniLM-L6-v2                        │
│                    Bi-Encoder                            │
│                    Embeddings                            │
│                          │                               │
│               Pinecone Serverless Index                  │
│               (cosine, 384-dim, AWS)                     │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                     QUERY LAYER                          │
│                                                          │
│  User Query                                              │
│      │                                                   │
│      ├─→ Multi-Query Expansion (3 variants, Gemini)      │
│      └─→ HyDE Passage Generation (Gemini)                │
│                          │                               │
│              5 search vectors total                      │
│                          │                               │
│          LLM Intent Router → doc_type tag                │
│          (security/travel/training/policy/None)          │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                   RETRIEVAL LAYER                        │
│                                                          │
│  Pre-filter (doc_type == inferred_type)                  │
│      │                                                   │
│  Top-25 per search vector → Pinecone                     │
│      │                                                   │
│  Pool + Deduplicate (max-score wins on conflict)         │
│      │                                                   │
│  Post-filter: drop stale doc versions (_v1, _2023, …)    │
│      │                                                   │
│  Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)         │
│  Scores (query, chunk) pairs → Top-5                     │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                  GENERATION LAYER                        │
│                                                          │
│  Build labelled context block (source + scores)         │
│      │                                                   │
│  Gemini Pro — grounded generation prompt                 │
│  "Answer ONLY from the excerpts below."                  │
│      │                                                   │
│  Final Answer with full traceability                     │
└──────────────────────────────────────────────────────────┘
```

---

## Design Decisions

### Why sliding-window sentence chunking over character splits?

Character splits are language-agnostic and fast, but they regularly cut words, sentences, and tabular structures mid-stream. Sentence-boundary splits guarantee semantic integrity. The sliding window ensures that any idea spanning two chunks is represented fully in at least one.

### Why TF-IDF keywords over raw frequency?

Raw frequency in a corpus of policy documents promotes generic policy vocabulary ("shall", "employee", "policy") to keyword status for every chunk. TF-IDF's IDF component penalises corpus-wide frequent terms. The resulting keywords are chunk-discriminative and serve as effective pre-filter metadata.

### Why 25 candidates into the reranker?

The bi-encoder (all-MiniLM) retrieves broadly but shallowly — it can miss fine-grained logical relevance. The cross-encoder re-scores every (query, chunk) pair with full attention over both texts, recovering this depth at the cost of latency. Retrieving 25 then pruning to 5 balances recall with precision and solves the "Lost in the Middle" problem (LLMs ignore content buried deep in long contexts).

### Why HyDE alongside multi-query expansion?

Multi-query expansion diversifies the *question* space. HyDE diversifies the *answer* space — it generates a document-like passage whose embedding tends to sit geometrically closer to real policy chunks than the raw question embedding. Together they maximise recall before the reranker prunes.

---

## Latency Profile (approximate, Colab T4)

| Stage | Latency |
|---|---|
| Query expansion (Gemini) | ~1–2 s |
| HyDE generation (Gemini) | ~1 s |
| 5× Pinecone queries | ~0.5 s |
| Cross-encoder (25 pairs) | ~0.3 s |
| Gemini generation | ~2–3 s |
| **Total** | **~5–7 s** |
