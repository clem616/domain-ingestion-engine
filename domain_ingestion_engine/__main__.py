"""CLI entry point: python -m domain_ingestion_engine  (or: domain-ingestion-engine)"""

import argparse
import json
import os
import sys
from pathlib import Path

from domain_ingestion_engine import DomainIngestionEngine


def _load_env() -> None:
    """Load .env from cwd or parent directories. Best-effort, no hard dep on dotenv."""
    for candidate in (Path.cwd() / ".env", Path.cwd().parent / ".env"):
        if candidate.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(candidate, override=False)
            except ImportError:
                with open(candidate) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, _, v = line.partition("=")
                            k = k.strip()
                            if k in (
                                "ANTHROPIC_API_KEY",
                                "LLM_API_KEY_CLAUDE",
                                "LLM_BASE_URL_CLAUDE",
                                "LLM_MODEL_CLAUDE",
                                "LLM_MODEL_CLAUDE_HAIKU",
                            ):
                                os.environ.setdefault(k, v.strip())
            break


def main() -> None:
    _load_env()

    env_api_base = os.environ.get("LLM_BASE_URL_CLAUDE")
    env_model = os.environ.get("LLM_MODEL_CLAUDE_HAIKU") or os.environ.get("LLM_MODEL_CLAUDE")

    parser = argparse.ArgumentParser(
        prog="domain-ingestion-engine",
        description="Generate adversarial RAG benchmark ground truth from documents.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--docs-dir", type=Path, help="Local directory of PDFs / .txt files.")
    source.add_argument("--hf-dataset", type=str, help="HuggingFace dataset ID.")

    parser.add_argument("--hf-split", type=str, default="train")
    parser.add_argument("--hf-config", type=str, default=None)
    parser.add_argument("--n-cases", type=int, default=150)
    parser.add_argument("--domain-name", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-ood", type=int, default=50)
    parser.add_argument("--n-poisoned", type=int, default=20)
    parser.add_argument("--n-multiturn", type=int, default=20)
    parser.add_argument("--n-qa-per-chunk", type=int, default=2)
    parser.add_argument("--model", type=str, default=DomainIngestionEngine.DEFAULT_MODEL)
    parser.add_argument("--api-base-url", type=str, default=env_api_base)
    parser.add_argument("--adversarial-only", action="store_true")
    parser.add_argument("--source-ir-path", type=Path, default=None)
    parser.add_argument("--hf-cache-dir", type=Path, default=None)

    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("LLM_API_KEY_CLAUDE")
    if not api_key:
        print("ERROR: No API key found. Set ANTHROPIC_API_KEY (or LLM_API_KEY_CLAUDE).", file=sys.stderr)
        sys.exit(1)

    model = args.model
    if model == DomainIngestionEngine.DEFAULT_MODEL and env_model:
        model = env_model

    engine = DomainIngestionEngine(
        anthropic_api_key=api_key,
        model=model,
        n_qa_per_chunk=args.n_qa_per_chunk,
        verbose=True,
        api_base_url=args.api_base_url,
    )

    if args.hf_dataset:
        paths = engine.ingest_from_hf(
            dataset_id=args.hf_dataset,
            hf_split=args.hf_split,
            hf_config=args.hf_config,
            domain_name=args.domain_name,
            output_dir=args.output_dir,
            n_cases=args.n_cases,
            n_ood_cases=args.n_ood,
            n_poisoned_cases=args.n_poisoned,
            n_multiturn_cases=args.n_multiturn,
            adversarial_only=args.adversarial_only,
            cache_dir=args.hf_cache_dir,
        )
    else:
        seed_ir = None
        if args.adversarial_only and args.source_ir_path:
            seed_ir = json.loads(args.source_ir_path.read_text(encoding="utf-8"))
            print(f"  Loaded {len(seed_ir)} seed IR cases from {args.source_ir_path}.")
        paths = engine.ingest(
            docs_dir=args.docs_dir,
            domain_name=args.domain_name,
            output_dir=args.output_dir,
            n_ir_cases=args.n_cases if not args.adversarial_only else None,
            n_ood_cases=args.n_ood,
            n_poisoned_cases=args.n_poisoned,
            n_multiturn_cases=args.n_multiturn,
            adversarial_only=args.adversarial_only,
            seed_ir_cases=seed_ir,
        )

    print("\nGenerated files:")
    for name, path in paths.items():
        print(f"  {name:16s} → {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
