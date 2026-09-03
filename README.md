# Opti-MedRAG

**Adaptive, retrieval-aware multi-agent reasoning for medical question answering.**

Opti-MedRAG integrates **MedRAG** (retrieval-augmented generation over medical corpora) with a
**multi-agent clinical debate framework** (adapted from MDAgents). Instead of feeding retrieved
documents into a single model, Opti-MedRAG first judges whether the retrieved evidence is
*sufficient*, and then **adaptively decides how much reasoning compute to spend** — from a single
expert to a full multidisciplinary panel. This trades cost for accuracy only when a question
actually needs it.

---

## Table of contents

- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [API key setup](#api-key-setup)
- [Datasets](#datasets)
  - [1. QA benchmark (MedQA)](#1-qa-benchmark-medqa)
  - [2. Retrieval corpus (auto-downloaded)](#2-retrieval-corpus-auto-downloaded)
- [Running](#running)
- [Output](#output)
- [Configuration reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [Notes & caveats](#notes--caveats)

---

## How it works

For each question, Opti-MedRAG runs an **adaptive routing pipeline**:

```
                        ┌─────────────────────────────┐
   question ───────────▶│  MedRAG retrieval (top-10)  │   MedCPT dense retriever
                        └──────────────┬──────────────┘   over a medical corpus
                                       │
                                       ▼
                        ┌─────────────────────────────┐
                        │      determine_relevance      │  Is the evidence sufficient?
                        └───────┬─────────────┬────────┘
                       relevant │             │ irrelevant
                                ▼             ▼
                      ┌──────────────┐  ┌────────────────────┐
                      │    BASIC     │  │ determine_difficulty│
                      │ single expert│  └────┬──────────┬────┘
                      │  RAG answer  │  interm.│          │advanced
                      └──────────────┘        ▼          ▼
                        (cheapest)   ┌──────────────┐ ┌──────────────┐
                                     │ INTERMEDIATE │ │   ADVANCED   │
                                     │ expert panel │ │ multi-team   │
                                     │    debate    │ │  (MDT) + mod │
                                     └──────────────┘ └──────────────┘
                                       (moderate)        (most costly)
```

1. **Retrieve** the top-*k* snippets from a medical corpus using the MedCPT dense retriever.
2. **Relevance gate** (`determine_relevance`) — an LLM judges whether the retrieved context is
   relevant and sufficient, and ranks the most useful documents.
3. **Adaptive routing:**
   - **Basic** — evidence is sufficient → a single expert answers directly from the top-ranked
     documents (cheapest).
   - **Intermediate** — evidence is insufficient → recruit a panel of specialists who each give an
     opinion, optionally debate, and a **moderator takes a majority vote**.
   - **Advanced** — for the most complex queries → organize multiple **multidisciplinary teams
     (MDTs)**; each deliberates internally, then a chief moderator synthesizes a final decision.

The intuition: strong retrieval makes cheap answering reliable, so the expensive multi-agent
machinery is reserved for cases where retrieval falls short.

---

## Repository layout

```
Opti-MedRAG/
├── Opti-MedRAG.py            # Entry point: adaptive routing pipeline
├── run_demo.sh               # Convenience runner (venv + env vars + key loading)
├── requirements.txt          # Verified dependency set
├── .env.example              # Template for your OpenAI API key
├── src/                      # MedRAG retrieval library
│   ├── medrag.py             #   MedRAG class (retrieval + generation)
│   ├── utils.py              #   RetrievalSystem / Retriever / DocExtracter (FAISS, RRF)
│   ├── template.py           #   Prompt templates
│   ├── config.py             #   API config (no secrets committed)
│   └── data/                 #   Corpus chunkers (pubmed, textbooks, statpearls, wikipedia)
├── MD_Agents/
│   └── utils_og.py           # Multi-agent framework: Agent/Group + the query paths
├── MDAgents/data/<dataset>/  # QA benchmark data (test.jsonl / train.jsonl)
└── corpus/                   # Auto-downloaded corpora + FAISS indexes (git-ignored)
```

---

## Prerequisites

| Requirement | Why | Install (macOS) |
|-------------|-----|-----------------|
| **Python 3.11** | Pinned ML deps build cleanly on 3.11 | `brew install python@3.11` |
| **git-lfs** | Corpora are stored as Git LFS objects on Hugging Face | `brew install git-lfs && git lfs install` |
| **wget** | Precomputed embeddings are fetched via `wget` | `brew install wget` |
| **unzip** | Unpacks the downloaded embedding archives | preinstalled on macOS |
| **OpenAI API key** | The agents and gates call the OpenAI API | — |
| **~3–5 GB free disk** | Textbooks corpus + MedCPT embeddings + index | — |
| *(optional)* **Java 11+** | Only for the BM25 retriever (`pyserini`) | `brew install openjdk@11` |

> **Note:** [`uv`](https://github.com/astral-sh/uv) is used below for fast environment setup. If you
> don't have it: `curl -LsSf https://astral.sh/uv/install.sh | sh` (or use plain `python -m venv`).

---

## Installation

```bash
# 1. Clone
git clone https://github.com/vasavamsi/Opti-MedRAG.git
cd Opti-MedRAG

# 2. System tools (macOS / Homebrew)
brew install git-lfs wget
git lfs install

# 3. Create an isolated Python 3.11 environment
uv venv --python python3.11 .venv
source .venv/bin/activate

# 4. Install dependencies
uv pip install -r requirements.txt
#   (plain pip works too: pip install -r requirements.txt)
```

The dependency set is pinned to a **verified working combination**. Two pins matter in particular:

- `httpx==0.27.2` — `openai==1.14.2` breaks with `httpx>=0.28` (which removed the `proxies` arg).
- `huggingface_hub<0.26` — `sentence_transformers==2.2.2` relies on the removed `cached_download`.

Verify the install:

```bash
python -c "import openai, torch, faiss, sentence_transformers, transformers; print('OK')"
```

---

## API key setup

Your key is loaded from a **git-ignored `.env` file** — it is never committed.

```bash
cp .env.example .env
# then edit .env and set your key:
#   OPENAI_API_KEY=sk-...
```

Alternatively, export it in your shell: `export OPENAI_API_KEY=sk-...`

> ⚠️ **Never commit an API key.** `.env` is listed in `.gitignore`. If a key is ever exposed,
> rotate it immediately at <https://platform.openai.com/api-keys>.

---

## Datasets

Opti-MedRAG needs **two** kinds of data: a **QA benchmark** (the questions) and a **retrieval
corpus** (the medical knowledge base). The corpus is downloaded automatically; the QA benchmark is
a small set of files you place on disk.

### 1. QA benchmark (MedQA)

QA data lives at `MDAgents/data/<dataset>/` as two JSON-Lines files:

```
MDAgents/data/medqa/test.jsonl     # questions to answer
MDAgents/data/medqa/train.jsonl    # few-shot exemplars
```

Each line is one question in this schema:

```json
{
  "question": "A 23-year-old pregnant woman ... Which of the following is the best treatment?",
  "options": {"A": "Ampicillin", "B": "Ceftriaxone", "C": "Doxycycline",
              "D": "Nitrofurantoin", "E": "Trimethoprim-sulfamethoxazole"},
  "answer": "Nitrofurantoin",
  "answer_idx": "D",
  "meta_info": "step2&3"
}
```

**A small sample set is already included** in this repo so you can run immediately.

**To use the full MedQA benchmark**, drop the official files into the same folder. The MedQA
(USMLE) dataset is available from:

- The MedRAG benchmark on Hugging Face: [`MedRAG/MedQA`](https://huggingface.co/datasets/MedRAG)
- The original MedQA release: [`jind11/MedQA`](https://github.com/jind11/MedQA)

```bash
# Example: place full MedQA files as:
MDAgents/data/medqa/test.jsonl
MDAgents/data/medqa/train.jsonl
```

Any dataset following the schema above works — pass its folder name via `--dataset`.

### 2. Retrieval corpus (auto-downloaded)

The retrieval corpus is downloaded **automatically on the first run** into `./corpus/`
(git-ignored). No manual steps are needed — the code clones the corpus from Hugging Face and
fetches precomputed MedCPT embeddings, then builds a FAISS index.

The corpus is selected in `Opti-MedRAG.py`:

```python
medrag = MedRAG(llm_name="OpenAI/gpt-4o-mini", rag=True,
                retriever_name="MedCPT", corpus_name="Textbooks")
```

Available corpora (via the `MedRAG/*` Hugging Face datasets):

| `corpus_name` | Contents | Approx. size | Notes |
|---------------|----------|--------------|-------|
| **`Textbooks`** | 18 medical textbooks | **small (~GBs)** | **Recommended default** — fast to download & index |
| `StatPearls`  | StatPearls articles | medium | Embeddings computed locally (slower first run) |
| `Wikipedia`   | Medical Wikipedia | large | |
| `PubMed`      | PubMed abstracts | **very large (tens of GB)** | Not recommended for a laptop |
| `MedText`     | Textbooks + StatPearls | medium | |
| `MedCorp`     | All of the above | very large | |

Retrievers (`retriever_name`): `MedCPT` (default, dense), `Contriever`, `SPECTER`, `BM25`
(needs Java), `RRF-2`, `RRF-4` (fusion). **`MedCPT` + `Textbooks`** is the recommended combination
and the one this project is tested with.

> The first run downloads a few GB and builds the index; **subsequent runs load the cached index
> and start in seconds.**

---

## Running

The simplest way is the provided runner, which activates the venv, sets the stabilizing threading
environment variables, loads your key, and starts the pipeline:

```bash
./run_demo.sh
```

Or run directly:

```bash
source .venv/bin/activate
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python Opti-MedRAG.py --dataset medqa --model gpt-4o-mini --difficulty adaptive
```

### Demonstrating each path

```bash
# Adaptive (default): the relevance gate decides the route per question
./run_demo.sh

# Force the multi-agent expert-panel debate on every question
./run_demo.sh --difficulty intermediate
```

On easy, well-covered questions the relevance gate usually routes to the **basic** path, so use
`--difficulty intermediate` to showcase the multi-agent debate.

---

## Output

Results are appended (one JSON object per line) to:

```
Opti-MedRAG-2 medqa output_gpt-4o-mini.json
```

Each record contains the question, the routing decision, and the final answer:

```json
{
  "question": "...",
  "difficulty": "basic",
  "answer": {"majority": {"0.0": "... Answer: (D) Nitrofurantoin"}}
}
```

Progress is also printed live to the console, including the retrieval step, the relevance/difficulty
routing decision, the recruited experts, the debate, and the moderator's final vote.

---

## Configuration reference

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `medqa` | Folder name under `MDAgents/data/` |
| `--model` | `gpt-4o-mini` | OpenAI model used by the agents |
| `--difficulty` | `adaptive` | `adaptive` uses the relevance gate; `intermediate` / `advanced` force a path |
| `--num_samples` | `100` | Reserved (the current loop iterates over all test items) |

Change the retrieval corpus/retriever by editing the `MedRAG(...)` call in `Opti-MedRAG.py`.

---

## Troubleshooting

| Symptom | Cause & fix |
|---------|-------------|
| **Segfault / `exit 139`** | faiss/torch OpenMP clash. The env vars `KMP_DUPLICATE_LIB_OK=TRUE`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` fix it (already set by `run_demo.sh` and in the entry point). |
| **`TypeError: ... unexpected keyword argument 'proxies'`** | `httpx` too new for `openai==1.14.2`. `pip install "httpx==0.27.2"`. |
| **`cached_download` ImportError / `huggingface_hub`** | `pip install "huggingface_hub<0.26"` (needed by `sentence_transformers==2.2.2`). |
| **`OPENAI_API_KEY is not set`** | Create `.env` with your key (see [API key setup](#api-key-setup)). |
| **Corpus clone has tiny/empty files** | Git LFS not installed. `brew install git-lfs && git lfs install`, then delete `corpus/` and re-run. |
| **`wget: command not found`** | `brew install wget`. |
| **BM25 retriever errors** | `pyserini` needs a Java runtime. Install Java, or use `MedCPT` (recommended). |

---

## Notes & caveats

- **Threading env vars are required** on macOS/Apple Silicon to avoid faiss/torch segfaults; they
  are set automatically by `run_demo.sh` and inside `Opti-MedRAG.py`.
- **The relevance gate rarely escalates on easy questions.** With the Textbooks corpus, classic
  exam questions are judged "relevant" and take the cheap **basic** path. Use
  `--difficulty intermediate` to exercise the debate path, or supply harder/less-covered questions.
- **Cost & latency scale with the path.** Basic ≈ a couple of API calls; intermediate ≈ 20–30;
  advanced (multi-team) more. `gpt-4o-mini` keeps this inexpensive.
- **The advanced path falls back to intermediate** if team formation fails, so the run never breaks
  on a parsing hiccup.
- This is research code adapted from **MedRAG** (retrieval) and **MDAgents** (multi-agent
  collaboration); the adaptive relevance-gated routing is the Opti-MedRAG contribution.
