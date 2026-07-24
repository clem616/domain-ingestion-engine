"""Shared fixtures and helpers for domain-ingestion-engine tests."""

import json
from unittest.mock import MagicMock

import pytest

from domain_ingestion_engine.engine import DomainIngestionEngine


def make_anthropic_message(text: str) -> MagicMock:
    """Return a mock Anthropic message with usage tracking."""
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    msg.usage.input_tokens = 10
    msg.usage.output_tokens = 5
    return msg


def make_openai_response(text: str) -> MagicMock:
    """Return a mock requests.Response for an OpenAI-compat endpoint."""
    resp = MagicMock()
    resp.json.return_value = {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    return resp


def ir_json(n: int = 1) -> str:
    """Return a valid IR cases JSON string."""
    cases = [
        {
            "question": f"Question {i}",
            "ground_truth_context": f"The answer is value_{i}.",
            "ideal_answer_points": f"Answer must include value_{i}.",
            "expected_chunk_keywords": [f"value_{i}", "answer"],
            "doc_id": "doc-001",
        }
        for i in range(n)
    ]
    return json.dumps(cases)


def ood_json(n: int = 2) -> str:
    return json.dumps([
        {"id": f"ood_{i+1:03d}", "question": f"OOD question {i}", "domain": "cooking"}
        for i in range(n)
    ])


def poison_json() -> str:
    return json.dumps({
        "question": "What is the dosage?",
        "original_fact": "10mg",
        "counterfact": "100mg",
        "description": "dosage amount",
    })


def multiturn_json() -> str:
    return json.dumps({
        "turn2_question": "What about the side effects?",
        "turn2_standalone": "What are the side effects of the drug mentioned?",
    })


@pytest.fixture
def engine() -> DomainIngestionEngine:
    return DomainIngestionEngine(
        anthropic_api_key="test-key",
        model="claude-haiku-4-5",
        n_qa_per_chunk=1,
        verbose=False,
    )


@pytest.fixture
def verbose_engine() -> DomainIngestionEngine:
    return DomainIngestionEngine(
        anthropic_api_key="test-key",
        verbose=True,
    )


@pytest.fixture
def sample_chunks() -> list[dict]:
    return [
        {
            "doc_id": "doc-001",
            "chunk_id": "doc-001_chunk_0000",
            "text": "The approved dosage is 10mg daily for patients over 18 years old.",
            "source_file": "/docs/doc-001.txt",
        },
        {
            "doc_id": "doc-001",
            "chunk_id": "doc-001_chunk_0001",
            "text": "Maximum dosage must not exceed 20mg per day. January 2023 guidelines apply.",
            "source_file": "/docs/doc-001.txt",
        },
    ]


@pytest.fixture
def sample_ir_cases() -> list[dict]:
    return [
        {
            "id": "gen_0000",
            "question": "What is the approved dosage?",
            "expected_documents": ["doc-001"],
            "ground_truth_context": "The approved dosage is 10mg daily.",
            "ideal_answer_points": "10mg daily for patients over 18.",
            "intent": "document_content",
            "expected_chunk_keywords": ["dosage", "10mg", "daily"],
            "relevance_context_for_ir": "The approved dosage is 10mg daily.",
            "_source": "llm_generated",
            "_source_doc_name": "doc-001",
            "_doc_number": "doc-001",
        }
    ]
