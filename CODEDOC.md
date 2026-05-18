# RAGAS Evaluation Demo — Code Documentation

A simple guide to what every file does and how they talk to each other.

---

## What does this app do?

It demonstrates how to evaluate a **RAG (Retrieval-Augmented Generation)** system using the **RAGAS** framework. In plain English:

1. You put documents into a database.
2. You ask questions — the app finds relevant chunks of text and sends them to an AI to answer.
3. RAGAS measures *how good* those answers are using four scores.
4. You can see where all the chunks live in space using pretty 2D and 3D plots.

---

## File Map

```
RAGAS/
├── app.py                  ← The web UI (Streamlit). Everything the user sees.
├── config.py               ← Settings (model names, AWS region, chunk size, etc.)
├── ingest.py               ← Loads documents, chunks them, embeds them, saves to ChromaDB
├── retriever.py            ← Searches ChromaDB to find relevant chunks for a question
├── evaluate.py             ← Runs the full RAGAS evaluation on 10 test questions
├── visualize.py            ← Computes t-SNE and builds Plotly 2D/3D plots
├── documents/              ← The knowledge base — plain .txt files the app learns from
├── test_data/
│   └── test_questions.jsonl  ← 10 questions with expected answers for evaluation
├── chroma_db/              ← Auto-created folder where ChromaDB stores embeddings on disk
├── requirements.txt        ← Python packages to install
├── deploy.py               ← One-click AWS deployment script
└── CODEDOC.md              ← This file
```

---

## config.py — Settings

Think of this as the "control panel". Change things here and the whole app adjusts.

| Setting | What it means |
|---|---|
| `AWS_REGION` | Which AWS data centre to use (default: us-east-1) |
| `AWS_PROFILE` | Which AWS credentials profile to use locally. Set to empty on EC2 (uses IAM role) |
| `BEDROCK_LLM_MODEL` | The Claude Haiku 4.5 model ID on Bedrock (inference profile format) |
| `BEDROCK_EMBED_MODEL` | Amazon Titan Text Embeddings v2 — turns text into numbers |
| `CHROMA_PATH` | Where ChromaDB saves its data on disk |
| `CHUNK_SIZE` | How many words per chunk when splitting documents (250) |
| `CHUNK_OVERLAP` | How many words overlap between adjacent chunks (40) |
| `TOP_K` | How many chunks to retrieve per question (3) |
| `EMBED_DIMENSIONS` | Size of each embedding vector (1024 numbers) |

---

## ingest.py — Loading the Knowledge Base

**What it does:** Reads every `.txt` file in `documents/`, splits each file into overlapping chunks of text, converts each chunk to a vector (embedding) using AWS Bedrock, and saves everything to ChromaDB.

**Key function: `ingest_documents()`**
1. Creates a boto3 session (using the AWS profile or IAM role)
2. Opens a ChromaDB persistent store on disk
3. For each document file:
   - Reads the text
   - Calls `_chunk_text()` to split it into word-limited pieces with overlap
   - Calls `_embed_single()` once per chunk — this sends the text to Bedrock Titan Embeddings and gets back a list of 1024 numbers
   - Calls `collection.add()` to store the chunk text + its embedding vector in ChromaDB
4. Accepts a `log_callback` so the UI can display live progress

**Why overlap?** If a sentence spans the boundary between two chunks, overlap ensures it appears in at least one complete chunk so it can be retrieved.

---

## retriever.py — Finding Relevant Chunks

**What it does:** Given a question, it converts the question to an embedding vector and asks ChromaDB to find the closest stored chunk vectors. Returns the actual text of the top-K matches.

**Key function: `retrieve(question)`**
1. Embeds the question using Bedrock Titan (same model used during ingest)
2. Passes the question embedding to `collection.query()` — ChromaDB uses HNSW index to find nearest neighbours
3. Returns a list of text strings (the matching chunks)

**Key function: `get_all_embeddings()`**
Used by the visualisation step — returns every stored embedding and its metadata (document name, chunk number) so they can be plotted with t-SNE.

---

## evaluate.py — Running RAGAS

**What it does:** Runs all 10 test questions through the full RAG pipeline and scores the results with four RAGAS metrics.

**Key function: `run_evaluation()`**
1. Creates a LangChain `ChatBedrock` LLM and `BedrockEmbeddings` — RAGAS requires LangChain wrappers
2. Loads questions and ground-truth answers from `test_questions.jsonl`
3. For each question:
   - Calls `retrieve()` to get the 3 most relevant chunks
   - Calls `_generate_answer()` — sends question + chunks to Claude Haiku and gets an answer
   - Wraps everything in a `SingleTurnSample` object
4. Passes all samples to RAGAS `evaluate()` which runs four LLM-judge metrics
5. Returns scores as a dict and a pandas DataFrame for display

---

## visualize.py — t-SNE Plots

**What it does:** Fetches all stored embeddings from ChromaDB, runs scikit-learn t-SNE to squash 1024-dimensional vectors into 2D or 3D, and builds interactive Plotly scatter plots.

**Why t-SNE?** Embedding vectors are 1024 numbers — impossible to visualise directly. t-SNE finds a 2D/3D arrangement that preserves the neighbourhood structure: chunks that were close in 1024D stay close in 2D/3D. Chunks about the same topic end up in clusters.

**Expected clusters with the 10 documents:**
- **AI / Deep Learning cluster** — ai_overview, deep_learning
- **NLP / Transformers cluster** — transformers, attention_mechanisms, nlp_fundamentals
- **Retrieval / RAG cluster** — rag_systems, vector_databases, dense_retrieval
- **Evaluation cluster** — llm_evaluation, ragas_deep_dive

---

## app.py — The Web Interface

Built with Streamlit. The sidebar has a radio button navigator for six steps:

| Step | What happens |
|---|---|
| 1 Ingest | Click a button, watch live logs as documents are chunked and embedded |
| 2 Test Data | Preview the 10 JSONL test questions |
| 3 Chat with RAG | Type any question, get an AI answer backed by retrieved chunks. Add "expected answer" to unlock all 4 metrics |
| 4 Evaluate | Run full evaluation (10 fixed questions, 4 metrics) or chat evaluation (your questions, 2–4 metrics) |
| 5 Metrics | Score cards, comparison bar chart, per-question heatmap |
| 6 Visualise | Generate 2D or 3D t-SNE scatter plots |

**Chat evaluation metrics:**
- Without expected answers: Faithfulness + Answer Relevancy (2 metrics — no ground truth needed)
- With expected answers: All 4 metrics including Context Precision and Context Recall

---

## RAGAS Metrics — Plain English

| Metric | Plain English |
|---|---|
| **Faithfulness** | Did the AI stick to the retrieved text? (1.0 = no hallucination) |
| **Answer Relevancy** | Did the AI actually answer the question asked? |
| **Context Precision** | Were the most useful chunks ranked at the top? |
| **Context Recall** | Did we retrieve *all* the information needed to answer correctly? |

All scores are 0–1 where 1 is best. NaN means the metric could not be computed (usually because the answer was too short or the LLM judge call failed).

---

## Data Flow Diagram

```
documents/*.txt
      │
      ▼
  ingest.py  ──► Bedrock Titan Embeddings ──► ChromaDB (chroma_db/)
                                                    │
                              ┌─────────────────────┤
                              │                     │
                         retriever.py          visualize.py
                              │                     │
                         (semantic              t-SNE 2D/3D
                          search)               Plotly charts
                              │
                         evaluate.py / app.py (chat)
                              │
                         Claude Haiku 4.5
                              │
                          RAGAS scores
                              │
                          app.py (display)
```

---

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
# Open http://localhost:8501
```

AWS credentials must be configured (`aws configure --profile brijesh` or an IAM role).

## Deploying to AWS

```bash
python deploy.py
# Provisions EC2 t3.medium, uploads code, starts the app
# Prints the public URL when ready
```

---

## Packages Used

| Package | Why |
|---|---|
| `streamlit` | Web UI |
| `boto3` | AWS SDK — calls Bedrock for embeddings and LLM |
| `langchain-aws` | LangChain wrappers for Bedrock (required by RAGAS) |
| `chromadb` | Local vector database |
| `ragas` | Evaluation framework |
| `scikit-learn` | t-SNE for dimensionality reduction |
| `plotly` | Interactive 2D/3D charts |
| `pandas` | Data tables |
