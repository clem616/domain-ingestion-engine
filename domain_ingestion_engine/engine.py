"""DomainIngestionEngine — automatically generates adversarial RAG benchmark ground truth
from a directory of raw documents (PDFs or text files) or a HuggingFace dataset.

Usage
-----
    from domain_ingestion_engine import DomainIngestionEngine

    engine = DomainIngestionEngine(anthropic_api_key="sk-ant-...")
    engine.ingest(
        docs_dir=Path("./my_documents"),
        domain_name="Legal Research",
        output_dir=Path("./ground_truth/legal"),
    )

    # Or from a HuggingFace dataset (requires: pip install datasets):
    engine.ingest_from_hf(
        dataset_id="qiaojin/PubMedQA",
        hf_split="pqa_labeled",
        domain_name="Biomedical Research",
        output_dir=Path("./ground_truth/pubmed"),
        n_cases=150,
    )

Cost estimate: ~$1-5 with claude-haiku-4-5 for a 50-100 document corpus.

Output (written to output_dir/):
    ground_truth_ir.json          — QA cases with quality flags
    ground_truth_ood.json         — Out-of-domain refusal queries (15 domains)
    ground_truth_poisoned.json    — Counterfactual grounding cases
    ground_truth_multiturn.json   — Multi-turn conversation chains
    source_passages.json          — doc_id -> full passage text map
    domain_metadata.json          — Provenance, cost, quality filter rate

Dependencies:
    Required : anthropic>=0.40.0, pypdf>=3.0.0
    Optional : datasets>=2.0.0  (for ingest_from_hf)
               requests>=2.28.0 (for OpenAI-compatible endpoints)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


# ── Built-in text chunker (no langchain dependency) ──────────────────────────

def _split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph and sentence boundaries."""
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:  # pragma: no cover
        end = min(start + chunk_size, text_len)
        if end < text_len:
            for sep in ("\n\n", "\n", ". ", " "):
                pos = text.rfind(sep, start + chunk_size // 2, end)
                if pos != -1:
                    end = pos + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = end - chunk_overlap
    return chunks


# ── Engine ────────────────────────────────────────────────────────────────────

class DomainIngestionEngine:
    """Generates all adversarial benchmark ground truth from raw documents via a Haiku-class LLM."""

    DEFAULT_MODEL = "claude-haiku-4-5"
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100

    def __init__(
        self,
        anthropic_api_key: str,
        model: str = DEFAULT_MODEL,
        n_qa_per_chunk: int = 2,
        verbose: bool = True,
        api_base_url: str | None = None,
    ) -> None:
        self._api_key = anthropic_api_key
        self._model = model
        self._n_qa_per_chunk = n_qa_per_chunk
        self._verbose = verbose
        self._input_tokens = 0
        self._output_tokens = 0
        self._api_base_url = (api_base_url or "https://api.anthropic.com").rstrip("/")

    # ── Public entry points ───────────────────────────────────────────────────

    def ingest(
        self,
        docs_dir: Path,
        domain_name: str,
        output_dir: Path,
        n_ir_cases: int | None = None,
        n_ood_cases: int = 50,
        n_poisoned_cases: int = 20,
        n_multiturn_cases: int = 20,
        adversarial_only: bool = False,
        seed_ir_cases: list[dict] | None = None,
    ) -> dict[str, Path]:
        """Generate all ground truth files from a local document directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._log(f"Loading and chunking documents from {docs_dir} ...")
        chunks = self._load_and_chunk_dir(Path(docs_dir))
        self._log(f"  {len(chunks)} chunks from {len({c['doc_id'] for c in chunks})} documents.")

        return self._run_pipeline(
            chunks=chunks,
            domain_name=domain_name,
            output_dir=output_dir,
            n_ir_cases=n_ir_cases,
            n_ood_cases=n_ood_cases,
            n_poisoned_cases=n_poisoned_cases,
            n_multiturn_cases=n_multiturn_cases,
            adversarial_only=adversarial_only,
            source_info={"type": "local_dir", "path": str(docs_dir)},
            seed_ir_cases=seed_ir_cases,
        )

    def ingest_from_hf(
        self,
        dataset_id: str,
        domain_name: str,
        output_dir: Path,
        hf_split: str = "train",
        hf_config: str | None = None,
        n_cases: int = 150,
        n_ood_cases: int = 50,
        n_poisoned_cases: int = 20,
        n_multiturn_cases: int = 20,
        adversarial_only: bool = False,
        cache_dir: Path | None = None,
    ) -> dict[str, Path]:
        """Generate all ground truth files from a HuggingFace dataset.

        Requires: pip install domain-ingestion-engine[hf]
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        config_label = f"/{hf_config}" if hf_config else ""
        self._log(f"Loading HuggingFace dataset {dataset_id}{config_label} (split={hf_split}, n={n_cases}) ...")
        chunks, hf_cases = self._load_from_hf(dataset_id, hf_split, n_cases, cache_dir, hf_config)
        self._log(f"  {len(chunks)} chunks from {len({c['doc_id'] for c in chunks})} records.")

        doc_texts: dict[str, str] = {}
        for chunk in chunks:
            doc_id = chunk["doc_id"]
            if doc_id not in doc_texts:
                doc_texts[doc_id] = chunk["text"]
            else:
                doc_texts[doc_id] += "\n\n" + chunk["text"]

        return self._run_pipeline(
            chunks=chunks,
            domain_name=domain_name,
            output_dir=output_dir,
            n_ir_cases=n_cases,
            n_ood_cases=n_ood_cases,
            n_poisoned_cases=n_poisoned_cases,
            n_multiturn_cases=n_multiturn_cases,
            adversarial_only=adversarial_only,
            source_info={"type": "huggingface", "dataset_id": dataset_id,
                         "split": hf_split, "n_cases": n_cases},
            hf_cases=hf_cases,
            doc_texts=doc_texts,
        )

    # ── Internal pipeline ─────────────────────────────────────────────────────

    def _run_pipeline(
        self,
        chunks: list[dict],
        domain_name: str,
        output_dir: Path,
        n_ir_cases: int | None,
        n_ood_cases: int,
        n_poisoned_cases: int,
        n_multiturn_cases: int,
        adversarial_only: bool,
        source_info: dict,
        hf_cases: list[dict] | None = None,
        doc_texts: dict[str, str] | None = None,
        seed_ir_cases: list[dict] | None = None,
    ) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        ir_quality_stats: dict = {}

        if not adversarial_only:
            self._log("Generating IR/MC ground truth cases ...")
            if hf_cases is not None:
                ir_cases = hf_cases
                for case in ir_cases:
                    case["quality_flag"] = "hf_trusted"
            else:
                ir_cases = self._generate_ir_cases(chunks, n_ir_cases)
                low = sum(1 for c in ir_cases if self._quality_flag(c) == "low")
                for case in ir_cases:
                    case["quality_flag"] = self._quality_flag(case)
                filter_rate = low / len(ir_cases) if ir_cases else 0.0
                ir_quality_stats = {"ir_low_quality_cases": low, "ir_quality_filter_rate": round(filter_rate, 3)}
                self._log(f"  Quality check: {low}/{len(ir_cases)} cases flagged low ({filter_rate:.1%}).")
            self._log(f"  {len(ir_cases)} IR cases generated.")
            ir_path = output_dir / "ground_truth_ir.json"
            ir_path.write_text(json.dumps(ir_cases, indent=2, ensure_ascii=False), encoding="utf-8")
            paths["ir"] = ir_path
        else:
            ir_cases = []

        self._log(f"Generating {n_ood_cases} OOD refusal cases ...")
        ood_cases = self._generate_ood_cases(domain_name, n_ood_cases)
        ood_path = output_dir / "ground_truth_ood.json"
        ood_path.write_text(json.dumps(ood_cases, indent=2, ensure_ascii=False), encoding="utf-8")
        paths["ood"] = ood_path

        self._log(f"Generating {n_poisoned_cases} poisoned context cases ...")
        poisoned_cases = self._generate_poisoned_cases(chunks, n_poisoned_cases)
        poisoned_path = output_dir / "ground_truth_poisoned.json"
        poisoned_path.write_text(json.dumps(poisoned_cases, indent=2, ensure_ascii=False), encoding="utf-8")
        paths["poisoned"] = poisoned_path

        source_cases = ir_cases if ir_cases else (seed_ir_cases or [])
        n_mt = min(n_multiturn_cases, len(source_cases))
        self._log(f"Generating {n_mt} multi-turn conversation chains ...")
        mt_cases = self._generate_multiturn_cases(source_cases[:n_mt])
        mt_path = output_dir / "ground_truth_multiturn.json"
        mt_path.write_text(json.dumps(mt_cases, indent=2, ensure_ascii=False), encoding="utf-8")
        paths["multiturn"] = mt_path

        if doc_texts is None:
            doc_texts = {}
            for chunk in chunks:
                did = chunk["doc_id"]
                doc_texts.setdefault(did, "")
                doc_texts[did] += "\n\n" + chunk["text"]
        passages_path = output_dir / "source_passages.json"
        passages_path.write_text(json.dumps(doc_texts, indent=2, ensure_ascii=False), encoding="utf-8")
        paths["source_passages"] = passages_path

        metadata = {
            "domain_name": domain_name,
            "source": source_info,
            "chunk_count": len(chunks),
            "doc_count": len(doc_texts),
            "ir_cases": len(ir_cases),
            "ood_cases": len(ood_cases),
            "poisoned_cases": len(poisoned_cases),
            "multiturn_cases": len(mt_cases),
            "generation_model": self._model,
            "total_input_tokens": self._input_tokens,
            "total_output_tokens": self._output_tokens,
            "estimated_cost_usd": round(
                self._input_tokens / 1_000_000 * 1.0
                + self._output_tokens / 1_000_000 * 5.0,
                4,
            ),
            **ir_quality_stats,
        }
        meta_path = output_dir / "domain_metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        paths["metadata"] = meta_path

        self._print_cost_summary(metadata)
        return paths

    # ── Document loading + chunking ───────────────────────────────────────────

    def _load_and_chunk_dir(self, docs_dir: Path) -> list[dict]:
        chunks = []
        for path in sorted(docs_dir.glob("**/*")):
            if path.suffix.lower() == ".pdf":
                text = self._extract_pdf_text(path)
            elif path.suffix.lower() in (".txt", ".md"):
                text = path.read_text(encoding="utf-8", errors="replace")
            else:
                continue
            if not text.strip():
                continue
            doc_id = path.stem
            for idx, chunk_text in enumerate(_split_text(text, self.CHUNK_SIZE, self.CHUNK_OVERLAP)):
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_chunk_{idx:04d}",
                    "text": chunk_text,
                    "source_file": str(path),
                })
        return chunks

    def _load_from_hf(
        self,
        dataset_id: str,
        split: str,
        n_cases: int,
        cache_dir: Path | None,
        config_name: str | None = None,
    ) -> tuple[list[dict], list[dict]]:
        try:
            from datasets import load_dataset  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "HuggingFace ingestion requires the 'datasets' package. "
                "Install it with: pip install datasets"
            ) from None

        kwargs: dict[str, Any] = {"split": split}
        if cache_dir:
            kwargs["cache_dir"] = str(cache_dir)
        pos_args = [dataset_id] + ([config_name] if config_name else [])
        ds = load_dataset(*pos_args, **kwargs)
        rows = list(ds.select(range(min(n_cases, len(ds)))))

        chunks: list[dict] = []
        ir_cases: list[dict] = []

        for i, row in enumerate(rows):
            doc_id = f"HF-{i:04d}"

            if "context" in row and isinstance(row["context"], dict) and "contexts" in row["context"]:
                passages = row["context"]["contexts"]
                doc_text = "\n\n".join(passages)
                question = row.get("question", "")
                answer = row.get("long_answer", "")
            else:
                passages_field = next(
                    (f for f in ("passages", "context", "document", "text") if f in row and row[f]),
                    None,
                )
                if passages_field is None:
                    continue
                raw = row[passages_field]
                doc_text = raw if isinstance(raw, str) else "\n\n".join(raw)
                question = row.get("question", row.get("query", ""))
                answer = row.get("answers", {})
                if isinstance(answer, dict):
                    answer = answer.get("text", [""])[0] if answer.get("text") else ""
                elif isinstance(answer, list):
                    answer = answer[0] if answer else ""

            if not doc_text.strip() or not question.strip():
                continue

            for idx, chunk_text in enumerate(_split_text(doc_text, self.CHUNK_SIZE, self.CHUNK_OVERLAP)):
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_chunk_{idx:04d}",
                    "text": chunk_text,
                    "source_file": f"{dataset_id}/{doc_id}",
                })

            if answer:
                ir_cases.append({
                    "id": f"hf_{i:04d}",
                    "question": question,
                    "expected_documents": [doc_id],
                    "ground_truth_context": answer,
                    "ideal_answer_points": answer,
                    "intent": "document_content",
                    "expected_chunk_keywords": None,
                    "relevance_context_for_ir": answer[:500],
                    "_source": "huggingface",
                    "_source_doc_name": f"{dataset_id} row {i}",
                    "_doc_number": doc_id,
                })

        return chunks, ir_cases

    @staticmethod
    def _extract_pdf_text(path: Path) -> str:
        from pypdf import PdfReader  # type: ignore[import]
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    # ── IR case generation ────────────────────────────────────────────────────

    def _generate_ir_cases(self, chunks: list[dict], n_cases: int | None) -> list[dict]:
        cases: list[dict] = []
        batch: list[dict] = []
        case_index = 0

        for chunk in chunks:
            batch.append(chunk)
            if len(batch) == 5 or (chunk is chunks[-1] and batch):
                new_cases = self._call_ir_generation(batch, start_idx=case_index)
                cases.extend(new_cases)
                case_index += len(new_cases)
                batch = []
                if n_cases and len(cases) >= n_cases:
                    break

        return cases[:n_cases] if n_cases else cases

    def _call_ir_generation(self, chunks: list[dict], start_idx: int) -> list[dict]:
        passages_text = "\n\n".join(
            f"[PASSAGE {i+1} | doc_id={c['doc_id']}]\n{c['text']}"
            for i, c in enumerate(chunks)
        )
        system = (
            "You are an expert dataset curator creating ground truth QA pairs for RAG evaluation. "
            "Return only valid JSON — no explanations, no markdown code fences."
        )
        user = f"""For each passage below, generate {self._n_qa_per_chunk} factual question(s) that can ONLY be answered using that specific passage.

For each question, return a JSON object with these exact fields:
- "question": the question text (specific, factual, not yes/no)
- "ground_truth_context": the exact sentence(s) from the passage that answer it
- "ideal_answer_points": a 1-2 sentence summary of what a complete answer must include
- "expected_chunk_keywords": list of 3-5 key phrases (exact words or short phrases copied verbatim from the passage) that a retrieval system must surface to answer the question
- "doc_id": the doc_id value from the passage header

Return a JSON array of these objects.

PASSAGES:
{passages_text}"""

        raw = self._call_api(system, user)
        try:
            data = json.loads(self._strip_fences(raw))
            if not isinstance(data, list):
                data = [data]
        except json.JSONDecodeError:
            return []

        cases = []
        for item in data:
            if not isinstance(item, dict) or not item.get("question"):
                continue
            doc_id = item.get("doc_id", chunks[0]["doc_id"])
            cases.append({
                "id": f"gen_{start_idx + len(cases):04d}",
                "question": item["question"],
                "expected_documents": [doc_id],
                "ground_truth_context": item.get("ground_truth_context", ""),
                "ideal_answer_points": item.get("ideal_answer_points", ""),
                "intent": "document_content",
                "expected_chunk_keywords": item.get("expected_chunk_keywords") or [],
                "relevance_context_for_ir": item.get("ground_truth_context", "")[:500],
                "_source": "llm_generated",
                "_source_doc_name": doc_id,
                "_doc_number": doc_id,
            })
        return cases

    def _quality_flag(self, case: dict) -> str:
        ctx = case.get("ground_truth_context", "").lower()
        kws = case.get("expected_chunk_keywords") or []
        if not kws:
            return "low"
        hits = sum(1 for kw in kws if str(kw).lower() in ctx)
        return "ok" if hits >= 2 else "low"

    # ── OOD case generation ───────────────────────────────────────────────────

    def _generate_ood_cases(self, domain_name: str, n: int) -> list[dict]:
        system = (
            "You are an expert dataset curator. "
            "Return only valid JSON — no explanations, no markdown code fences."
        )
        _DOMAINS = "finance, cooking, sports, travel, history, music, geography, literature, gaming, fashion, automotive, home improvement, linguistics, career advice, film"

        result: list[dict] = []
        batch_size = 10
        for batch_start in range(0, n, batch_size):
            batch_n = min(batch_size, n - batch_start)
            offset = batch_start + 1
            user = f"""Generate exactly {batch_n} questions about topics COMPLETELY UNRELATED to "{domain_name}".

Cover diverse domains: {_DOMAINS}.

Each question must be impossible to answer from a {domain_name} document corpus.

Return a JSON array where each element has:
- "id": "ood_{offset:03d}" through "ood_{offset + batch_n - 1:03d}" (sequential, zero-padded)
- "question": the question text
- "domain": the topic domain of the question"""

            raw = self._call_api(system, user)
            try:
                data = json.loads(self._strip_fences(raw))
                if not isinstance(data, list):
                    data = []
            except json.JSONDecodeError:
                data = []

            for i, item in enumerate(data):
                if isinstance(item, dict) and item.get("question"):
                    global_i = batch_start + i
                    result.append({
                        "id": item.get("id", f"ood_{global_i + 1:03d}"),
                        "question": item["question"],
                        "domain": item.get("domain", "unknown"),
                        "expected_refusal": "This question is outside the document domain; the system should decline to answer or state it cannot help.",
                    })
        return result[:n]

    # ── Poisoned context generation ───────────────────────────────────────────

    def _generate_poisoned_cases(self, chunks: list[dict], n: int) -> list[dict]:
        _fact_pattern = re.compile(
            r"\b(\d[\d,\.]+\s*(?:%|mg|ml|kg|days?|weeks?|months?|years?|hours?)|"
            r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}|"
            r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
        )
        scored = []
        for chunk in chunks:
            matches = _fact_pattern.findall(chunk["text"])
            if matches:
                scored.append((len(matches), chunk, matches[0]))

        scored.sort(key=lambda x: x[0], reverse=True)
        candidates = scored[:min(n * 3, len(scored))]

        cases: list[dict] = []
        for i, (_, chunk, original_fact) in enumerate(candidates[:n]):
            case = self._call_poison_generation(i + 1, chunk, original_fact)
            if case:
                cases.append(case)
        return cases

    def _call_poison_generation(self, idx: int, chunk: dict, original_fact: str) -> dict | None:
        system = (
            "You are an expert dataset curator for adversarial RAG evaluation. "
            "Return only valid JSON — no explanations, no markdown code fences."
        )
        user = f"""Given the passage below, generate a counterfactual test case.

The passage contains the fact: "{original_fact}"

Generate:
1. A question whose answer requires knowing this fact
2. A plausible but FALSE alternative to this fact (must sound realistic)

Return a single JSON object with:
- "question": the question
- "original_fact": "{original_fact}"
- "counterfact": the false but plausible alternative
- "description": a 3-5 word label for this fact

PASSAGE:
{chunk["text"][:600]}"""

        raw = self._call_api(system, user)
        try:
            data = json.loads(self._strip_fences(raw))
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict) or not data.get("question") or not data.get("counterfact"):
            return None

        return {
            "id": f"poison_{idx:03d}",
            "question": data["question"],
            "expected_documents": [chunk["doc_id"]],
            "original_fact": data.get("original_fact", original_fact),
            "counterfact": data["counterfact"],
            "description": data.get("description", f"fact from {chunk['doc_id']}"),
        }

    # ── Multi-turn case generation ────────────────────────────────────────────

    def _generate_multiturn_cases(self, ir_cases: list[dict]) -> list[dict]:
        cases = []
        for i, base_case in enumerate(ir_cases):
            mt_case = self._call_multiturn_generation(i + 1, base_case)
            if mt_case:
                cases.append(mt_case)
        return cases

    def _call_multiturn_generation(self, idx: int, base_case: dict) -> dict | None:
        system = (
            "You are an expert dataset curator for conversational RAG evaluation. "
            "Return only valid JSON — no explanations, no markdown code fences."
        )
        user = f"""Given the following Q&A pair, create a 2-turn conversation where the second question is ambiguous without context from the first.

Original Q1: {base_case["question"]}
Expected answer topic: {base_case["ideal_answer_points"][:200]}

Generate:
- turn2_question: an ambiguous follow-up that uses pronouns or implicit references to Q1
- turn2_standalone: the same follow-up rewritten as a fully self-contained question

Return a single JSON object with these exact fields."""

        raw = self._call_api(system, user)
        try:
            data = json.loads(self._strip_fences(raw))
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict) or not data.get("turn2_question"):
            return None

        return {
            "id": f"mt_{idx:03d}",
            "description": f"Multi-turn from: {base_case['question'][:60]}...",
            "turns": [
                {
                    "turn": 1,
                    "question": base_case["question"],
                    "intent": "document_content",
                    "expected_documents": base_case["expected_documents"],
                },
                {
                    "turn": 2,
                    "question": data["turn2_question"],
                    "expected_standalone": data.get("turn2_standalone", data["turn2_question"]),
                    "expected_documents": base_case["expected_documents"],
                },
            ],
        }

    # ── API call ──────────────────────────────────────────────────────────────

    def _call_api(self, system: str, user: str) -> str:
        base = self._api_base_url
        is_azure_foundry = "services.ai.azure.com" in base
        is_anthropic_native = "api.anthropic.com" in base
        uses_anthropic_format = (
            is_anthropic_native
            or base.rstrip("/").endswith("/anthropic")
            or "/anthropic/" in base
        )

        if uses_anthropic_format:
            if is_azure_foundry:
                from anthropic import AnthropicFoundry
                client = AnthropicFoundry(api_key=self._api_key, base_url=base)
            else:
                from anthropic import Anthropic
                client = Anthropic(
                    api_key=self._api_key,
                    **({"base_url": base} if not is_anthropic_native else {}),
                )
            msg = client.messages.create(
                model=self._model,
                max_tokens=2048,
                temperature=0.3,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            self._input_tokens += msg.usage.input_tokens
            self._output_tokens += msg.usage.output_tokens
            return msg.content[0].text
        else:
            try:
                import requests
            except ImportError:
                raise ImportError(
                    "OpenAI-compatible endpoints require the 'requests' package. "
                    "Install it with: pip install requests"
                ) from None
            url = f"{base}/chat/completions"
            headers = {"api-key": self._api_key, "content-type": "application/json"}
            payload = {
                "model": self._model,
                "max_tokens": 2048,
                "temperature": 0.3,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage", {})
            self._input_tokens += usage.get("prompt_tokens", 0)
            self._output_tokens += usage.get("completion_tokens", 0)
            return data["choices"][0]["message"]["content"]

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _strip_fences(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = text[text.index("\n") + 1:] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        return text.strip()

    def _log(self, msg: str) -> None:
        if self._verbose:
            print(msg, file=sys.stderr)

    def _print_cost_summary(self, metadata: dict) -> None:
        if not self._verbose:
            return
        cost = metadata["estimated_cost_usd"]
        quality_line = ""
        if "ir_quality_filter_rate" in metadata:
            rate = metadata["ir_quality_filter_rate"]
            low = metadata["ir_low_quality_cases"]
            total = metadata["ir_cases"]
            quality_line = f"  Quality flags: {low}/{total} low ({rate:.1%}) — check 'quality_flag' field\n"
        print(
            f"\n[DomainIngestionEngine] Done.\n"
            f"  Input tokens:  {metadata['total_input_tokens']:,}\n"
            f"  Output tokens: {metadata['total_output_tokens']:,}\n"
            f"  Est. cost:     ${cost:.4f} (Haiku pricing)\n"
            f"{quality_line}",
            file=sys.stderr,
        )
