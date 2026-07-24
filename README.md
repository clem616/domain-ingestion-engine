# domain-ingestion-engine

Convert any document corpus into adversarial RAG evaluation datasets — in under one hour, for under five dollars.

Given a directory of PDFs or text files, the engine generates five ground-truth JSON files covering IR retrieval, out-of-domain refusal, poisoned context, and multi-turn conversation. These files are directly consumable by [`rag-adversarial-bench`](https://github.com/rag-adversarial-bench/rag-adversarial-bench) or any custom evaluation harness.

---

## Requirements

- Python 3.10 or higher
- An Anthropic API key (`sk-ant-...`) — the engine calls `claude-haiku-4-5` by default. Sign up at [console.anthropic.com](https://console.anthropic.com).

---

## Install

**Step 1 — clone the repository:**

```bash
git clone https://github.com/rag-adversarial-bench/domain-ingestion-engine
cd domain-ingestion-engine
```

**Step 2 — install the dependencies:**

```bash
pip install -r requirements.txt
```

That is all. No FAISS, no sentence-transformers, no PyTorch.

To also load from HuggingFace datasets (optional — only needed for `--hf-dataset`):

```bash
pip install datasets
```

**Step 3 — run from the repository root.** All commands below must be executed from inside the `domain-ingestion-engine/` directory you just cloned. The engine is a local Python module, not an installed package.

---

## Quickstart

**Step 1 — set your API key:**

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

If you use Azure AI Foundry instead of the Anthropic API directly:

```bash
export LLM_BASE_URL_CLAUDE=https://your-resource.services.ai.azure.com/anthropic
export LLM_API_KEY_CLAUDE=your-azure-key
```

**Step 2 — prepare your documents:**

Create a directory and put your PDF or `.txt` files inside it:

```bash
mkdir my_documents
# copy your PDFs or text files into my_documents/
```

**Step 3 — run the engine:**

```bash
python -m domain_ingestion_engine \
  --docs-dir ./my_documents \
  --domain-name "Legal Research" \
  --output-dir ./ground_truth/legal
```

The engine will:
1. Read every PDF and `.txt` file in `my_documents/`
2. Split each document into chunks
3. Call `claude-haiku-4-5` to generate QA pairs, OOD queries, poisoned passages, and multi-turn chains
4. Run a quality self-check on each generated case
5. Write five JSON files into `./ground_truth/legal/`

Expected runtime: 10–30 minutes for 50 documents. Estimated cost: $1–5 with Haiku.

---

## Output files

After the engine finishes, `./ground_truth/legal/` contains:

| File | Content |
|---|---|
| `ground_truth_ir.json` | QA pairs with expected document IDs and quality flags |
| `ground_truth_ood.json` | Out-of-domain refusal queries (15 unrelated domains) |
| `ground_truth_poisoned.json` | Counterfactual passage + original fact pairs |
| `ground_truth_multiturn.json` | Multi-turn conversation chains with standalone rewrites |
| `source_passages.json` | `doc_id` → full passage text map (enables evaluation without original PDFs) |
| `domain_metadata.json` | Provenance: model used, chunk count, doc count, quality filter rate, estimated cost |

All cases include a `ground_truth_context` field (the source passage text) so downstream evaluation requires no access to the original PDFs.

---

## Cost model

| Corpus size | Model | Estimated cost |
|---|---|---|
| 10–50 documents | `claude-haiku-4-5` | $1–3 |
| 50–200 documents | `claude-haiku-4-5` | $3–10 |
| Adversarial cases only (`--adversarial-only`) | `claude-haiku-4-5` | < $1 |

The engine generates 2 QA pairs per chunk by default. Pass `--n-qa-per-chunk 1` to halve cost on large corpora.

---

## Quality self-check

After generating each IR case the engine verifies that at least 2 of the `expected_chunk_keywords` appear verbatim in the source context. Cases that fail are flagged `"quality_flag": "low"` rather than silently included. The aggregate filter rate is reported in `domain_metadata.json`.

This is a proxy signal, not a substitute for human annotation. See the v2 roadmap for a human-audit protocol.

---

## CLI reference

```
python -m domain_ingestion_engine [OPTIONS]

Input (one required):
  --docs-dir PATH            Directory of PDFs or .txt files
  --hf-dataset TEXT          HuggingFace dataset ID (requires: pip install datasets)

Required:
  --domain-name TEXT         Human-readable domain label (used in metadata)
  --output-dir PATH          Where to write the ground-truth files

Optional:
  --model TEXT               LLM model name [default: claude-haiku-4-5]
  --n-qa-per-chunk INT       QA pairs generated per chunk [default: 2]
  --adversarial-only         Skip IR generation; produce only OOD/poisoned/multiturn
  --source-ir-path PATH      Seed multiturn chains from an existing IR JSON file
  --n-cases INT              Number of IR cases when loading from HuggingFace [default: 150]
  --hf-split TEXT            HuggingFace split to load [default: train]
  --n-ood INT                Number of OOD cases to generate [default: 50]
  --n-poisoned INT           Number of poisoned cases to generate [default: 20]
  --n-multiturn INT          Number of multi-turn chains to generate [default: 20]
```

---

## Endpoint compatibility

The engine auto-detects which client to use based on the URL:

| URL pattern | Client used |
|---|---|
| `api.anthropic.com` (default) | Anthropic SDK |
| `*.services.ai.azure.com/anthropic` | Azure AI Foundry SDK |
| anything else | OpenAI-compatible endpoint |

---

## Repository structure

```
domain_ingestion_engine/
├── engine.py      — DomainIngestionEngine class (chunking, generation, quality check, API routing)
├── __main__.py    — CLI entry point
└── __init__.py    — public API: exposes DomainIngestionEngine

tests/
├── conftest.py    — shared fixtures and mock helpers
├── test_engine.py — engine unit tests (100% branch coverage, no real API calls)
└── test_cli.py    — CLI unit tests

pyproject.toml     — project metadata and dependencies
```

---

## Development setup

To run the test suite locally:

```bash
git clone https://github.com/rag-adversarial-bench/domain-ingestion-engine
cd domain-ingestion-engine
pip install uv          # if not already installed
uv sync                 # installs dev dependencies (pytest, pytest-cov, datasets, requests)
uv run pytest tests/ -v # 100% branch coverage, no real API calls made
```

---

## v2 roadmap — call for contributions

This is a v1 release. Open problems we are inviting the community to address:

- **Human GT audit** — a 50-case spot-check with recorded inter-annotator agreement to validate the keyword self-check as a quality proxy
- **Additional input formats** — `.docx`, `.html`, `.md` ingestion alongside PDF
- **Second domain validation** — a Legal or Finance instantiation to demonstrate domain-agnostic generalisability empirically
- **Local model support** — replace the Haiku API call with a locally-served model (Ollama, llama.cpp) for air-gapped environments

Issues and PRs welcome.

---

## License

Apache 2.0. See `LICENSE`.

The engine itself carries no dataset license — ground truth files you generate from your own corpus inherit the license of your source documents.

---

## Citation

The accompanying paper is currently under review. Citation will be added here upon publication.
