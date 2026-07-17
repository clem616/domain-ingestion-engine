# domain-ingestion-engine

Convert any document corpus into adversarial RAG evaluation datasets — in under one hour, for under five dollars.

Given a directory of PDFs or text files, the engine generates five ground-truth JSON files covering IR retrieval, out-of-domain refusal, poisoned context, and multi-turn conversation. These files are directly consumable by `rag-adversarial-bench` or any custom evaluation harness.

---

## Install

```bash
pip install domain-ingestion-engine
```

Dependencies: `anthropic`, `PyPDF2`, `tqdm`. No FAISS, no sentence-transformers, no PyTorch.

---

## Quickstart

```bash
python -m domain_ingestion_engine \
  --docs-dir ./my_documents \
  --domain-name "Legal Research" \
  --output-dir ./ground_truth/legal
```

That is the complete command. The engine reads every PDF in `my_documents/`, chunks them, calls a Haiku-class model once per batch, and writes five files to `./ground_truth/legal/`.

Set your API key once:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or for Azure AI Foundry:
export LLM_BASE_URL_CLAUDE=https://your-resource.services.ai.azure.com/anthropic
export LLM_API_KEY_CLAUDE=...
```

---

## Output files

| File | Content |
|---|---|
| `ground_truth_ir.json` | QA pairs with expected document IDs and quality flags |
| `ground_truth_ood.json` | Out-of-domain refusal queries (15 unrelated domains) |
| `ground_truth_poisoned.json` | Counterfactual passage + original fact pairs |
| `ground_truth_multiturn.json` | Multi-turn conversation chains with standalone rewrites |
| `domain_metadata.json` | Provenance: model, timestamp, chunk count, quality filter rate |

All cases include a `ground_truth_context` field (the source passage text) so downstream evaluation requires no access to the original PDFs.

---

## Cost model

| Corpus size | Default model | Estimated cost |
|---|---|---|
| 10–50 documents | `claude-haiku-4-5` | $1–3 |
| 50–200 documents | `claude-haiku-4-5` | $3–10 |
| Adversarial cases only (`--adversarial-only`) | `claude-haiku-4-5` | < $1 |

The engine generates 2 QA pairs per chunk by default (`--n-qa-per-chunk`). Reduce to 1 to halve cost on large corpora.

---

## Quality self-check

After generating each IR case the engine verifies that at least 2 of the `expected_chunk_keywords` appear verbatim in the source context. Cases that fail are flagged `"quality_flag": "low"` rather than silently included. The aggregate filter rate is reported in `domain_metadata.json`.

This is a proxy signal, not a substitute for human annotation. See the v2 roadmap for a human-audit protocol.

---

## Repository structure

```
domain_ingestion_engine/
├── engine.py               — DomainIngestionEngine class, main entry point
├── chunker.py              — PDF and text chunking (600 tokens, 100-token overlap)
├── quality.py              — keyword self-check and quality_flag logic
├── endpoint.py             — endpoint detection (Anthropic / Azure Foundry / OpenAI-compat)
├── generators/
│   ├── ir_generator.py     — IR QA pair generation
│   ├── ood_generator.py    — OOD refusal query generation (15 domains)
│   ├── poisoned_generator.py   — counterfactual passage generation
│   └── multiturn_generator.py  — multi-turn chain generation
└── cli.py                  — __main__ entry point

examples/
├── quickstart_local.py     — local PDF directory example
├── quickstart_hf.py        — ingest from a HuggingFace dataset
└── ground_truth/           — sample outputs for a 5-document test corpus

pyproject.toml              — deps: anthropic, PyPDF2, tqdm only
```

---

## Endpoint compatibility

| URL pattern | Auth |
|---|---|
| `api.anthropic.com` | `Anthropic` SDK |
| `*.services.ai.azure.com/anthropic` | `AnthropicFoundry` SDK |
| anything else | OpenAI-compatible endpoint |

---

## CLI reference

```
python -m domain_ingestion_engine [OPTIONS]

  --docs-dir PATH            Directory of PDFs or .txt files
  --domain-name TEXT         Human-readable domain label (used in metadata)
  --output-dir PATH          Where to write the 5 ground-truth files
  --model TEXT               LLM model name [default: claude-haiku-4-5]
  --n-qa-per-chunk INT       QA pairs generated per chunk [default: 2]
  --adversarial-only         Skip IR generation; produce only OOD/poisoned/multiturn
  --source-ir-path PATH      Seed multiturn chains from an existing IR JSON file
  --n-ood INT                Number of OOD cases to generate [default: 50]
  --n-poisoned INT           Number of poisoned cases to generate [default: 20]
  --n-multiturn INT          Number of multi-turn chains to generate [default: 20]
```

---

## v2 roadmap — call for contributions

This is a v1 release. The following are open problems we are inviting the community to address:

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

```bibtex
<!-- TODO: add arXiv citation after paper is submitted -->
```
