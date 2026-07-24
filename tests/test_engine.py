"""Tests for engine.py — DomainIngestionEngine and _split_text."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from domain_ingestion_engine.engine import DomainIngestionEngine, _split_text
from tests.conftest import (
    make_anthropic_message,
    make_openai_response,
    ir_json,
    ood_json,
    poison_json,
    multiturn_json,
)


# ── _split_text ───────────────────────────────────────────────────────────────

class TestSplitText:
    def test_empty_string(self):
        assert _split_text("") == []

    def test_short_text_single_chunk(self):
        result = _split_text("Hello world.", chunk_size=800, chunk_overlap=100)
        assert result == ["Hello world."]

    def test_splits_on_double_newline(self):
        text = ("A" * 500) + "\n\n" + ("B" * 400)
        chunks = _split_text(text, chunk_size=600, chunk_overlap=50)
        assert len(chunks) >= 2
        assert any("B" * 10 in c for c in chunks)

    def test_splits_on_single_newline(self):
        text = ("A" * 500) + "\n" + ("B" * 400)
        chunks = _split_text(text, chunk_size=600, chunk_overlap=50)
        assert len(chunks) >= 2

    def test_splits_on_sentence_boundary(self):
        text = ("word " * 100) + ". " + ("other " * 80)
        chunks = _split_text(text, chunk_size=300, chunk_overlap=30)
        assert len(chunks) >= 2

    def test_splits_on_space_when_no_better_separator(self):
        # Build a string with only spaces as separators
        text = "a" * 200 + " " + "b" * 200 + " " + "c" * 200
        chunks = _split_text(text, chunk_size=250, chunk_overlap=20)
        assert len(chunks) >= 2

    def test_no_separator_found_splits_at_chunk_size(self):
        # Solid block of text with no separators in the back half search window
        text = "x" * 1000
        chunks = _split_text(text, chunk_size=300, chunk_overlap=50)
        assert len(chunks) >= 3
        for chunk in chunks:
            assert len(chunk) <= 300

    def test_overlap_produces_shared_content(self):
        text = "word " * 200
        chunks = _split_text(text, chunk_size=100, chunk_overlap=30)
        # The end of chunk N should overlap with the start of chunk N+1
        assert len(chunks) >= 2

    def test_empty_chunks_discarded(self):
        text = "\n\n".join(["   ", "content here", "   "])
        chunks = _split_text(text, chunk_size=50, chunk_overlap=5)
        for chunk in chunks:
            assert chunk.strip()

    def test_whitespace_only_window_discarded(self):
        # Forces a chunk window that is entirely whitespace after strip,
        # covering the `if chunk:` False branch.
        text = "aaaaa" + " " * 10 + "bbbbb"
        chunks = _split_text(text, chunk_size=10, chunk_overlap=2)
        assert all(chunk for chunk in chunks)
        assert any("a" in c for c in chunks)
        assert any("b" in c for c in chunks)

    def test_exact_chunk_size_no_overflow(self):
        text = "a" * 800
        chunks = _split_text(text, chunk_size=800, chunk_overlap=100)
        assert len(chunks) == 1
        assert chunks[0] == "a" * 800

    def test_terminates_when_start_reaches_end(self):
        # Very short text should not loop infinitely
        chunks = _split_text("hi", chunk_size=10, chunk_overlap=5)
        assert chunks == ["hi"]


# ── __init__ ──────────────────────────────────────────────────────────────────

class TestInit:
    def test_defaults(self):
        e = DomainIngestionEngine(anthropic_api_key="key")
        assert e._api_key == "key"
        assert e._model == DomainIngestionEngine.DEFAULT_MODEL
        assert e._n_qa_per_chunk == 2
        assert e._verbose is True
        assert e._api_base_url == "https://api.anthropic.com"

    def test_custom_api_base_url_trailing_slash_stripped(self):
        e = DomainIngestionEngine(anthropic_api_key="key", api_base_url="https://example.com/")
        assert e._api_base_url == "https://example.com"

    def test_none_api_base_url_uses_default(self):
        e = DomainIngestionEngine(anthropic_api_key="key", api_base_url=None)
        assert "api.anthropic.com" in e._api_base_url

    def test_token_counters_start_at_zero(self):
        e = DomainIngestionEngine(anthropic_api_key="key")
        assert e._input_tokens == 0
        assert e._output_tokens == 0


# ── _quality_flag ─────────────────────────────────────────────────────────────

class TestQualityFlag:
    def test_ok_when_two_keywords_found(self, engine):
        case = {
            "ground_truth_context": "The dosage is 10mg daily for patients.",
            "expected_chunk_keywords": ["dosage", "10mg", "patients"],
        }
        assert engine._quality_flag(case) == "ok"

    def test_low_when_one_keyword_found(self, engine):
        case = {
            "ground_truth_context": "The dosage is 10mg daily.",
            "expected_chunk_keywords": ["dosage", "missing_word", "another_missing"],
        }
        assert engine._quality_flag(case) == "low"

    def test_low_when_no_keywords_found(self, engine):
        case = {
            "ground_truth_context": "Some unrelated text here.",
            "expected_chunk_keywords": ["alpha", "beta", "gamma"],
        }
        assert engine._quality_flag(case) == "low"

    def test_low_when_keywords_empty_list(self, engine):
        assert engine._quality_flag({"ground_truth_context": "text", "expected_chunk_keywords": []}) == "low"

    def test_low_when_keywords_none(self, engine):
        assert engine._quality_flag({"ground_truth_context": "text", "expected_chunk_keywords": None}) == "low"

    def test_low_when_keywords_missing(self, engine):
        assert engine._quality_flag({"ground_truth_context": "text"}) == "low"

    def test_case_insensitive_matching(self, engine):
        case = {
            "ground_truth_context": "DOSAGE is TEN milligrams.",
            "expected_chunk_keywords": ["dosage", "ten"],
        }
        assert engine._quality_flag(case) == "ok"


# ── _strip_fences ─────────────────────────────────────────────────────────────

class TestStripFences:
    def test_no_fences(self):
        assert DomainIngestionEngine._strip_fences('{"key": "value"}') == '{"key": "value"}'

    def test_json_fence_with_newline(self):
        text = "```json\n{\"key\": \"value\"}\n```"
        assert DomainIngestionEngine._strip_fences(text) == '{"key": "value"}'

    def test_plain_fence_no_newline(self):
        result = DomainIngestionEngine._strip_fences("```data```")
        assert "```" not in result

    def test_only_end_fence(self):
        text = '{"key": "value"}\n```'
        result = DomainIngestionEngine._strip_fences(text)
        assert not result.endswith("```")

    def test_strips_surrounding_whitespace(self):
        result = DomainIngestionEngine._strip_fences("  hello  ")
        assert result == "hello"


# ── _log and _print_cost_summary ─────────────────────────────────────────────

class TestLog:
    def test_verbose_prints_to_stderr(self, verbose_engine, capsys):
        verbose_engine._log("test message")
        captured = capsys.readouterr()
        assert "test message" in captured.err

    def test_silent_prints_nothing(self, engine, capsys):
        engine._log("should not appear")
        captured = capsys.readouterr()
        assert captured.err == ""


class TestPrintCostSummary:
    def test_verbose_false_no_output(self, engine, capsys):
        engine._print_cost_summary({"estimated_cost_usd": 0.5, "total_input_tokens": 100, "total_output_tokens": 50})
        assert capsys.readouterr().err == ""

    def test_with_quality_stats(self, verbose_engine, capsys):
        verbose_engine._print_cost_summary({
            "estimated_cost_usd": 0.5,
            "total_input_tokens": 1000,
            "total_output_tokens": 500,
            "ir_quality_filter_rate": 0.1,
            "ir_low_quality_cases": 2,
            "ir_cases": 20,
        })
        out = capsys.readouterr().err
        assert "Quality flags" in out
        assert "2/20" in out

    def test_without_quality_stats(self, verbose_engine, capsys):
        verbose_engine._print_cost_summary({
            "estimated_cost_usd": 0.25,
            "total_input_tokens": 500,
            "total_output_tokens": 250,
        })
        out = capsys.readouterr().err
        assert "Done" in out
        assert "Quality flags" not in out


# ── _call_api ─────────────────────────────────────────────────────────────────

class TestCallApi:
    def _make_client_mock(self, text: str) -> MagicMock:
        client = MagicMock()
        client.messages.create.return_value = make_anthropic_message(text)
        return client

    def test_anthropic_native_no_base_url_kwarg(self, engine):
        engine._api_base_url = "https://api.anthropic.com"
        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value = self._make_client_mock("native response")
            result = engine._call_api("sys", "user")
        assert result == "native response"
        _, kwargs = MockAnthropic.call_args
        assert "base_url" not in kwargs

    def test_anthropic_custom_url_passes_base_url(self, engine):
        engine._api_base_url = "https://custom.example.com/anthropic"
        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value = self._make_client_mock("custom response")
            result = engine._call_api("sys", "user")
        assert result == "custom response"
        _, kwargs = MockAnthropic.call_args
        assert kwargs.get("base_url") == "https://custom.example.com/anthropic"

    def test_azure_foundry_uses_anthropic_foundry(self, engine):
        engine._api_base_url = "https://myresource.services.ai.azure.com/anthropic"
        with patch("anthropic.AnthropicFoundry") as MockFoundry:
            MockFoundry.return_value = self._make_client_mock("foundry response")
            result = engine._call_api("sys", "user")
        assert result == "foundry response"
        MockFoundry.assert_called_once()

    def test_openai_compat_uses_requests(self, engine):
        engine._api_base_url = "https://openai-compat.example.com/v1"
        with patch("requests.post") as mock_post:
            mock_post.return_value = make_openai_response("compat response")
            result = engine._call_api("sys", "user")
        assert result == "compat response"
        assert engine._input_tokens == 10
        assert engine._output_tokens == 5

    def test_openai_compat_no_requests_raises(self, engine):
        engine._api_base_url = "https://openai-compat.example.com/v1"
        with patch.dict(sys.modules, {"requests": None}):
            with pytest.raises(ImportError, match="openai-compat"):
                engine._call_api("sys", "user")

    def test_token_counts_accumulate(self, engine):
        engine._api_base_url = "https://api.anthropic.com"
        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value = self._make_client_mock("response")
            engine._call_api("sys", "user")
            engine._call_api("sys", "user")
        assert engine._input_tokens == 20
        assert engine._output_tokens == 10


# ── _call_ir_generation ───────────────────────────────────────────────────────

class TestCallIrGeneration:
    def test_success_returns_cases(self, engine, sample_chunks):
        with patch.object(engine, "_call_api", return_value=ir_json(2)):
            cases = engine._call_ir_generation(sample_chunks[:1], start_idx=0)
        assert len(cases) == 2
        assert cases[0]["id"] == "gen_0000"
        assert cases[0]["intent"] == "document_content"

    def test_json_decode_error_returns_empty(self, engine, sample_chunks):
        with patch.object(engine, "_call_api", return_value="not json"):
            cases = engine._call_ir_generation(sample_chunks[:1], start_idx=0)
        assert cases == []

    def test_non_list_wrapped_in_list(self, engine, sample_chunks):
        single = json.dumps({
            "question": "Q?",
            "ground_truth_context": "ctx",
            "ideal_answer_points": "points",
            "expected_chunk_keywords": ["kw1", "kw2"],
            "doc_id": "doc-001",
        })
        with patch.object(engine, "_call_api", return_value=single):
            cases = engine._call_ir_generation(sample_chunks[:1], start_idx=0)
        assert len(cases) == 1

    def test_item_without_question_skipped(self, engine, sample_chunks):
        data = json.dumps([{"ground_truth_context": "no question here"}])
        with patch.object(engine, "_call_api", return_value=data):
            cases = engine._call_ir_generation(sample_chunks[:1], start_idx=0)
        assert cases == []

    def test_doc_id_from_item_used(self, engine, sample_chunks):
        data = json.dumps([{
            "question": "Q?",
            "ground_truth_context": "ctx",
            "ideal_answer_points": "pts",
            "expected_chunk_keywords": ["k"],
            "doc_id": "custom-doc",
        }])
        with patch.object(engine, "_call_api", return_value=data):
            cases = engine._call_ir_generation(sample_chunks[:1], start_idx=0)
        assert cases[0]["expected_documents"] == ["custom-doc"]

    def test_doc_id_fallback_to_first_chunk(self, engine, sample_chunks):
        data = json.dumps([{
            "question": "Q?",
            "ground_truth_context": "ctx",
            "ideal_answer_points": "pts",
            "expected_chunk_keywords": [],
        }])
        with patch.object(engine, "_call_api", return_value=data):
            cases = engine._call_ir_generation(sample_chunks[:1], start_idx=5)
        assert cases[0]["expected_documents"] == ["doc-001"]
        assert cases[0]["id"] == "gen_0005"

    def test_missing_optional_fields_use_defaults(self, engine, sample_chunks):
        data = json.dumps([{"question": "Q?", "doc_id": "doc-001"}])
        with patch.object(engine, "_call_api", return_value=data):
            cases = engine._call_ir_generation(sample_chunks[:1], start_idx=0)
        assert cases[0]["ground_truth_context"] == ""
        assert cases[0]["expected_chunk_keywords"] == []


# ── _call_poison_generation ───────────────────────────────────────────────────

class TestCallPoisonGeneration:
    def test_success(self, engine, sample_chunks):
        with patch.object(engine, "_call_api", return_value=poison_json()):
            result = engine._call_poison_generation(1, sample_chunks[0], "10mg")
        assert result is not None
        assert result["id"] == "poison_001"
        assert result["counterfact"] == "100mg"

    def test_json_decode_error_returns_none(self, engine, sample_chunks):
        with patch.object(engine, "_call_api", return_value="invalid"):
            assert engine._call_poison_generation(1, sample_chunks[0], "fact") is None

    def test_missing_question_returns_none(self, engine, sample_chunks):
        data = json.dumps({"counterfact": "something"})
        with patch.object(engine, "_call_api", return_value=data):
            assert engine._call_poison_generation(1, sample_chunks[0], "fact") is None

    def test_missing_counterfact_returns_none(self, engine, sample_chunks):
        data = json.dumps({"question": "Q?"})
        with patch.object(engine, "_call_api", return_value=data):
            assert engine._call_poison_generation(1, sample_chunks[0], "fact") is None

    def test_original_fact_fallback(self, engine, sample_chunks):
        data = json.dumps({"question": "Q?", "counterfact": "fake"})
        with patch.object(engine, "_call_api", return_value=data):
            result = engine._call_poison_generation(1, sample_chunks[0], "real_fact")
        assert result["original_fact"] == "real_fact"

    def test_description_fallback(self, engine, sample_chunks):
        data = json.dumps({"question": "Q?", "counterfact": "fake"})
        with patch.object(engine, "_call_api", return_value=data):
            result = engine._call_poison_generation(1, sample_chunks[0], "fact")
        assert "doc-001" in result["description"]


# ── _call_multiturn_generation ────────────────────────────────────────────────

class TestCallMultiturnGeneration:
    def test_success(self, engine, sample_ir_cases):
        with patch.object(engine, "_call_api", return_value=multiturn_json()):
            result = engine._call_multiturn_generation(1, sample_ir_cases[0])
        assert result is not None
        assert result["id"] == "mt_001"
        assert len(result["turns"]) == 2
        assert result["turns"][1]["expected_standalone"] == "What are the side effects of the drug mentioned?"

    def test_json_error_returns_none(self, engine, sample_ir_cases):
        with patch.object(engine, "_call_api", return_value="bad"):
            assert engine._call_multiturn_generation(1, sample_ir_cases[0]) is None

    def test_missing_turn2_question_returns_none(self, engine, sample_ir_cases):
        with patch.object(engine, "_call_api", return_value=json.dumps({"turn2_standalone": "x"})):
            assert engine._call_multiturn_generation(1, sample_ir_cases[0]) is None

    def test_standalone_fallback_to_turn2_question(self, engine, sample_ir_cases):
        data = json.dumps({"turn2_question": "What about it?"})
        with patch.object(engine, "_call_api", return_value=data):
            result = engine._call_multiturn_generation(1, sample_ir_cases[0])
        assert result["turns"][1]["expected_standalone"] == "What about it?"


# ── _generate_ood_cases ───────────────────────────────────────────────────────

class TestGenerateOodCases:
    def test_single_batch(self, engine):
        with patch.object(engine, "_call_api", return_value=ood_json(5)):
            cases = engine._generate_ood_cases("Medical", 5)
        assert len(cases) == 5
        assert cases[0]["expected_refusal"]

    def test_multiple_batches(self, engine):
        with patch.object(engine, "_call_api", return_value=ood_json(10)):
            cases = engine._generate_ood_cases("Medical", 15)
        # Two batches of 10 → 20 items, truncated to 15
        assert len(cases) == 15

    def test_json_error_produces_empty_batch(self, engine):
        with patch.object(engine, "_call_api", return_value="not json"):
            cases = engine._generate_ood_cases("Medical", 5)
        assert cases == []

    def test_non_list_response_treated_as_empty(self, engine):
        with patch.object(engine, "_call_api", return_value=json.dumps({"wrong": "format"})):
            cases = engine._generate_ood_cases("Medical", 5)
        assert cases == []

    def test_item_without_question_skipped(self, engine):
        data = json.dumps([{"id": "ood_001", "domain": "cooking"}])
        with patch.object(engine, "_call_api", return_value=data):
            cases = engine._generate_ood_cases("Medical", 5)
        assert cases == []

    def test_id_fallback_when_missing(self, engine):
        data = json.dumps([{"question": "Q?", "domain": "cooking"}])
        with patch.object(engine, "_call_api", return_value=data):
            cases = engine._generate_ood_cases("Medical", 5)
        assert cases[0]["id"].startswith("ood_")


# ── _generate_poisoned_cases ──────────────────────────────────────────────────

class TestGeneratePoisonedCases:
    def test_chunks_with_facts_generate_cases(self, engine, sample_chunks):
        with patch.object(engine, "_call_poison_generation", return_value={
            "id": "poison_001", "question": "Q?", "counterfact": "fake",
            "original_fact": "10mg", "description": "dosage", "expected_documents": ["doc-001"],
        }):
            cases = engine._generate_poisoned_cases(sample_chunks, 2)
        assert len(cases) == 2

    def test_chunks_without_facts_produce_no_cases(self, engine):
        chunks = [{"doc_id": "x", "chunk_id": "x_0", "text": "no numbers or entities here", "source_file": "x"}]
        with patch.object(engine, "_call_poison_generation", return_value=None):
            cases = engine._generate_poisoned_cases(chunks, 5)
        assert cases == []

    def test_none_results_from_generation_skipped(self, engine, sample_chunks):
        with patch.object(engine, "_call_poison_generation", return_value=None):
            cases = engine._generate_poisoned_cases(sample_chunks, 2)
        assert cases == []

    def test_candidate_pool_capped_at_n_times_3(self, engine):
        # 10 chunks with facts, n=2 → candidates capped at 6
        chunks = [
            {"doc_id": f"doc-{i}", "chunk_id": f"doc-{i}_0",
             "text": f"The value is {i*10}mg daily.", "source_file": f"doc-{i}"}
            for i in range(1, 11)
        ]
        call_count = []
        def fake_poison(idx, chunk, fact):
            call_count.append(1)
            return {"id": f"poison_{idx:03d}", "question": "Q?", "counterfact": "x",
                    "original_fact": fact, "description": "d", "expected_documents": [chunk["doc_id"]]}

        with patch.object(engine, "_call_poison_generation", side_effect=fake_poison):
            engine._generate_poisoned_cases(chunks, 2)
        # n=2, so we try up to min(6, 10) = 6 candidates but only take 2
        assert len(call_count) == 2


# ── _generate_ir_cases ────────────────────────────────────────────────────────

class TestGenerateIrCases:
    def _make_chunks(self, n: int) -> list[dict]:
        return [
            {"doc_id": "doc-001", "chunk_id": f"doc-001_chunk_{i:04d}",
             "text": f"Text chunk {i}.", "source_file": "doc.txt"}
            for i in range(n)
        ]

    def test_batches_of_five_processed(self, engine):
        chunks = self._make_chunks(7)
        with patch.object(engine, "_call_ir_generation", return_value=[{"id": "x"}]) as mock_call:
            engine._generate_ir_cases(chunks, n_cases=None)
        # 7 chunks → batch of 5, then batch of 2 = 2 calls
        assert mock_call.call_count == 2

    def test_last_partial_batch_processed(self, engine):
        chunks = self._make_chunks(3)
        with patch.object(engine, "_call_ir_generation", return_value=[{"id": "x"}]) as mock_call:
            engine._generate_ir_cases(chunks, n_cases=None)
        assert mock_call.call_count == 1

    def test_n_cases_limit_stops_early(self, engine):
        chunks = self._make_chunks(10)
        with patch.object(engine, "_call_ir_generation", return_value=[{"id": f"x{i}"} for i in range(5)]):
            cases = engine._generate_ir_cases(chunks, n_cases=5)
        assert len(cases) == 5

    def test_no_n_cases_returns_all(self, engine):
        chunks = self._make_chunks(3)
        with patch.object(engine, "_call_ir_generation", return_value=[{"id": "x"}, {"id": "y"}]):
            cases = engine._generate_ir_cases(chunks, n_cases=None)
        assert len(cases) == 2


# ── _extract_pdf_text ─────────────────────────────────────────────────────────

class TestExtractPdfText:
    def test_joins_pages(self, tmp_path):
        fake_path = tmp_path / "test.pdf"
        fake_path.write_bytes(b"")
        page1, page2, page3 = MagicMock(), MagicMock(), MagicMock()
        page1.extract_text.return_value = "Page one text."
        page2.extract_text.return_value = None
        page3.extract_text.return_value = "Page three text."
        with patch("pypdf.PdfReader") as MockReader:
            MockReader.return_value.pages = [page1, page2, page3]
            result = DomainIngestionEngine._extract_pdf_text(fake_path)
        assert "Page one text." in result
        assert "Page three text." in result
        # None page produces empty string, not error
        assert result.count("\n") == 2


# ── _load_and_chunk_dir ───────────────────────────────────────────────────────

class TestLoadAndChunkDir:
    def test_txt_file(self, engine, tmp_path):
        (tmp_path / "doc.txt").write_text("Word " * 50, encoding="utf-8")
        chunks = engine._load_and_chunk_dir(tmp_path)
        assert len(chunks) >= 1
        assert chunks[0]["doc_id"] == "doc"
        assert chunks[0]["source_file"].endswith("doc.txt")

    def test_md_file(self, engine, tmp_path):
        (tmp_path / "readme.md").write_text("Content " * 50, encoding="utf-8")
        chunks = engine._load_and_chunk_dir(tmp_path)
        assert any(c["doc_id"] == "readme" for c in chunks)

    def test_unknown_extension_skipped(self, engine, tmp_path):
        (tmp_path / "data.csv").write_text("a,b,c", encoding="utf-8")
        chunks = engine._load_and_chunk_dir(tmp_path)
        assert chunks == []

    def test_empty_file_skipped(self, engine, tmp_path):
        (tmp_path / "empty.txt").write_text("   ", encoding="utf-8")
        chunks = engine._load_and_chunk_dir(tmp_path)
        assert chunks == []

    def test_pdf_file_calls_extract(self, engine, tmp_path):
        pdf_path = tmp_path / "paper.pdf"
        pdf_path.write_bytes(b"")
        with patch.object(engine.__class__, "_extract_pdf_text", staticmethod(lambda p: "PDF content " * 20)):
            chunks = engine._load_and_chunk_dir(tmp_path)
        assert any(c["doc_id"] == "paper" for c in chunks)

    def test_chunk_ids_are_sequential(self, engine, tmp_path):
        (tmp_path / "long.txt").write_text("word " * 500, encoding="utf-8")
        chunks = engine._load_and_chunk_dir(tmp_path)
        ids = [c["chunk_id"] for c in chunks]
        assert ids[0] == "long_chunk_0000"
        assert ids[1] == "long_chunk_0001"


# ── _load_from_hf ─────────────────────────────────────────────────────────────

class TestLoadFromHf:
    def test_datasets_not_installed_raises(self, engine):
        with patch.dict(sys.modules, {"datasets": None}):
            with pytest.raises(ImportError, match="datasets"):
                engine._load_from_hf("some/dataset", "train", 5, None)

    def _make_hf_dataset(self, rows: list[dict]) -> MagicMock:
        ds = MagicMock()
        ds.__len__.return_value = len(rows)
        ds.select.return_value = rows
        return ds

    def test_pubmed_schema(self, engine):
        row = {
            "context": {"contexts": ["Abstract text about drugs."]},
            "question": "What is the drug?",
            "long_answer": "The drug is X.",
        }
        mock_ds = self._make_hf_dataset([row])
        with patch("datasets.load_dataset", return_value=mock_ds):
            chunks, ir_cases = engine._load_from_hf("qiaojin/PubMedQA", "pqa_labeled", 1, None, "pqa_labeled")
        assert len(ir_cases) == 1
        assert ir_cases[0]["question"] == "What is the drug?"
        assert ir_cases[0]["_doc_number"] == "HF-0000"

    def test_generic_schema_passages_field(self, engine):
        row = {"passages": "Generic document text.", "question": "Q?", "answers": ["Answer A"]}
        mock_ds = self._make_hf_dataset([row])
        with patch("datasets.load_dataset", return_value=mock_ds):
            chunks, ir_cases = engine._load_from_hf("some/dataset", "train", 1, None)
        assert len(ir_cases) == 1

    def test_answer_as_dict_with_text(self, engine):
        row = {"passages": "text", "question": "Q?", "answers": {"text": ["Answer A"]}}
        mock_ds = self._make_hf_dataset([row])
        with patch("datasets.load_dataset", return_value=mock_ds):
            _, ir_cases = engine._load_from_hf("some/dataset", "train", 1, None)
        assert ir_cases[0]["ground_truth_context"] == "Answer A"

    def test_answer_as_dict_without_text(self, engine):
        row = {"passages": "text", "question": "Q?", "answers": {"other_field": []}}
        mock_ds = self._make_hf_dataset([row])
        with patch("datasets.load_dataset", return_value=mock_ds):
            _, ir_cases = engine._load_from_hf("some/dataset", "train", 1, None)
        # Empty answer → no ir_case added
        assert ir_cases == []

    def test_answer_as_empty_list(self, engine):
        row = {"passages": "text", "question": "Q?", "answers": []}
        mock_ds = self._make_hf_dataset([row])
        with patch("datasets.load_dataset", return_value=mock_ds):
            _, ir_cases = engine._load_from_hf("some/dataset", "train", 1, None)
        assert ir_cases == []

    def test_answer_as_plain_string(self, engine):
        # answers is neither dict nor list — falls through both isinstance checks (branch 351→354)
        row = {"passages": "Some document text.", "question": "Q?", "answers": "plain answer"}
        mock_ds = self._make_hf_dataset([row])
        with patch("datasets.load_dataset", return_value=mock_ds):
            _, ir_cases = engine._load_from_hf("some/dataset", "train", 1, None)
        assert len(ir_cases) == 1

    def test_no_passages_field_row_skipped(self, engine):
        row = {"question": "Q?", "answers": ["A"]}
        mock_ds = self._make_hf_dataset([row])
        with patch("datasets.load_dataset", return_value=mock_ds):
            chunks, ir_cases = engine._load_from_hf("some/dataset", "train", 1, None)
        assert chunks == []
        assert ir_cases == []

    def test_empty_doc_text_row_skipped(self, engine):
        row = {"passages": "   ", "question": "Q?", "answers": ["A"]}
        mock_ds = self._make_hf_dataset([row])
        with patch("datasets.load_dataset", return_value=mock_ds):
            chunks, ir_cases = engine._load_from_hf("some/dataset", "train", 1, None)
        assert chunks == []

    def test_empty_question_row_skipped(self, engine):
        row = {"passages": "Some text content.", "question": "   ", "answers": ["A"]}
        mock_ds = self._make_hf_dataset([row])
        with patch("datasets.load_dataset", return_value=mock_ds):
            chunks, ir_cases = engine._load_from_hf("some/dataset", "train", 1, None)
        assert chunks == []

    def test_passages_as_list_joined(self, engine):
        row = {"passages": ["Part A.", "Part B."], "question": "Q?", "answers": ["A"]}
        mock_ds = self._make_hf_dataset([row])
        with patch("datasets.load_dataset", return_value=mock_ds):
            chunks, _ = engine._load_from_hf("some/dataset", "train", 1, None)
        combined = " ".join(c["text"] for c in chunks)
        assert "Part A." in combined
        assert "Part B." in combined

    def test_cache_dir_passed_to_load_dataset(self, engine, tmp_path):
        row = {"context": {"contexts": ["text"]}, "question": "Q?", "long_answer": "A."}
        mock_ds = self._make_hf_dataset([row])
        with patch("datasets.load_dataset", return_value=mock_ds) as mock_load:
            engine._load_from_hf("some/dataset", "train", 1, tmp_path)
        call_kwargs = mock_load.call_args[1]
        assert "cache_dir" in call_kwargs


# ── _run_pipeline ─────────────────────────────────────────────────────────────

class TestRunPipeline:
    def _all_mocked(self, engine, ir_return=None, ood_return=None, poisoned_return=None, mt_return=None):
        engine._generate_ir_cases = MagicMock(return_value=ir_return or [])
        engine._generate_ood_cases = MagicMock(return_value=ood_return or [])
        engine._generate_poisoned_cases = MagicMock(return_value=poisoned_return or [])
        engine._generate_multiturn_cases = MagicMock(return_value=mt_return or [])

    def test_full_pipeline_writes_all_files(self, engine, tmp_path, sample_chunks, sample_ir_cases):
        self._all_mocked(engine, ir_return=sample_ir_cases)
        paths = engine._run_pipeline(
            chunks=sample_chunks,
            domain_name="Test",
            output_dir=tmp_path,
            n_ir_cases=1,
            n_ood_cases=2,
            n_poisoned_cases=2,
            n_multiturn_cases=1,
            adversarial_only=False,
            source_info={"type": "local_dir"},
        )
        assert (tmp_path / "ground_truth_ir.json").exists()
        assert (tmp_path / "ground_truth_ood.json").exists()
        assert (tmp_path / "ground_truth_poisoned.json").exists()
        assert (tmp_path / "ground_truth_multiturn.json").exists()
        assert (tmp_path / "source_passages.json").exists()
        assert (tmp_path / "domain_metadata.json").exists()
        assert "ir" in paths

    def test_adversarial_only_no_ir_file(self, engine, tmp_path, sample_chunks):
        self._all_mocked(engine)
        paths = engine._run_pipeline(
            chunks=sample_chunks,
            domain_name="Test",
            output_dir=tmp_path,
            n_ir_cases=None,
            n_ood_cases=2,
            n_poisoned_cases=2,
            n_multiturn_cases=0,
            adversarial_only=True,
            source_info={"type": "local_dir"},
        )
        assert "ir" not in paths
        assert not (tmp_path / "ground_truth_ir.json").exists()

    def test_hf_cases_get_trusted_quality_flag(self, engine, tmp_path, sample_chunks, sample_ir_cases):
        self._all_mocked(engine)
        engine._run_pipeline(
            chunks=sample_chunks,
            domain_name="Test",
            output_dir=tmp_path,
            n_ir_cases=1,
            n_ood_cases=0,
            n_poisoned_cases=0,
            n_multiturn_cases=0,
            adversarial_only=False,
            source_info={"type": "hf"},
            hf_cases=sample_ir_cases,
        )
        saved = json.loads((tmp_path / "ground_truth_ir.json").read_text())
        assert saved[0]["quality_flag"] == "hf_trusted"

    def test_seed_ir_used_when_adversarial_only(self, engine, tmp_path, sample_chunks, sample_ir_cases):
        self._all_mocked(engine)
        engine._run_pipeline(
            chunks=sample_chunks,
            domain_name="Test",
            output_dir=tmp_path,
            n_ir_cases=None,
            n_ood_cases=0,
            n_poisoned_cases=0,
            n_multiturn_cases=1,
            adversarial_only=True,
            source_info={"type": "local_dir"},
            seed_ir_cases=sample_ir_cases,
        )
        engine._generate_multiturn_cases.assert_called_once_with(sample_ir_cases[:1])

    def test_empty_ir_cases_filter_rate_zero(self, engine, tmp_path, sample_chunks):
        self._all_mocked(engine, ir_return=[])
        engine._run_pipeline(
            chunks=sample_chunks,
            domain_name="Test",
            output_dir=tmp_path,
            n_ir_cases=0,
            n_ood_cases=0,
            n_poisoned_cases=0,
            n_multiturn_cases=0,
            adversarial_only=False,
            source_info={"type": "local_dir"},
        )
        meta = json.loads((tmp_path / "domain_metadata.json").read_text())
        assert meta["ir_quality_filter_rate"] == 0.0

    def test_doc_texts_provided_not_rebuilt(self, engine, tmp_path, sample_chunks, sample_ir_cases):
        self._all_mocked(engine, ir_return=sample_ir_cases)
        provided = {"custom-doc": "custom content"}
        engine._run_pipeline(
            chunks=sample_chunks,
            domain_name="Test",
            output_dir=tmp_path,
            n_ir_cases=1,
            n_ood_cases=0,
            n_poisoned_cases=0,
            n_multiturn_cases=0,
            adversarial_only=False,
            source_info={"type": "local_dir"},
            doc_texts=provided,
        )
        saved = json.loads((tmp_path / "source_passages.json").read_text())
        assert saved == provided

    def test_metadata_written_correctly(self, engine, tmp_path, sample_chunks):
        self._all_mocked(engine, ir_return=[{"id": "x", "expected_chunk_keywords": [], "ground_truth_context": ""}])
        engine._run_pipeline(
            chunks=sample_chunks,
            domain_name="TestDomain",
            output_dir=tmp_path,
            n_ir_cases=1,
            n_ood_cases=0,
            n_poisoned_cases=0,
            n_multiturn_cases=0,
            adversarial_only=False,
            source_info={"type": "test"},
        )
        meta = json.loads((tmp_path / "domain_metadata.json").read_text())
        assert meta["domain_name"] == "TestDomain"
        assert meta["generation_model"] == "claude-haiku-4-5"


# ── ingest (integration) ──────────────────────────────────────────────────────

class TestIngest:
    def test_ingest_calls_run_pipeline(self, engine, tmp_path):
        with patch.object(engine, "_load_and_chunk_dir", return_value=[]):
            with patch.object(engine, "_run_pipeline", return_value={}) as mock_pipeline:
                engine.ingest(docs_dir=tmp_path, domain_name="Legal", output_dir=tmp_path)
        mock_pipeline.assert_called_once()
        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["domain_name"] == "Legal"
        assert call_kwargs["adversarial_only"] is False

    def test_ingest_passes_seed_ir_cases(self, engine, tmp_path):
        seeds = [{"id": "x"}]
        with patch.object(engine, "_load_and_chunk_dir", return_value=[]):
            with patch.object(engine, "_run_pipeline", return_value={}) as mock_pipeline:
                engine.ingest(docs_dir=tmp_path, domain_name="Legal", output_dir=tmp_path, seed_ir_cases=seeds)
        assert mock_pipeline.call_args[1]["seed_ir_cases"] == seeds


# ── ingest_from_hf (integration) ──────────────────────────────────────────────

class TestGenerateMultiturnCases:
    def test_returns_successful_cases(self, engine, sample_ir_cases):
        with patch.object(engine, "_call_multiturn_generation", return_value={
            "id": "mt_001", "description": "...", "turns": [],
        }):
            result = engine._generate_multiturn_cases(sample_ir_cases)
        assert len(result) == 1
        assert result[0]["id"] == "mt_001"

    def test_none_results_skipped(self, engine, sample_ir_cases):
        with patch.object(engine, "_call_multiturn_generation", return_value=None):
            result = engine._generate_multiturn_cases(sample_ir_cases)
        assert result == []

    def test_empty_input_returns_empty(self, engine):
        result = engine._generate_multiturn_cases([])
        assert result == []


class TestIngestFromHf:
    def test_ingest_from_hf_calls_run_pipeline(self, engine, tmp_path):
        fake_chunks = [
            {"doc_id": "HF-0000", "chunk_id": "HF-0000_chunk_0000", "text": "text", "source_file": "x"},
            {"doc_id": "HF-0000", "chunk_id": "HF-0000_chunk_0001", "text": "text2", "source_file": "x"},
        ]
        fake_cases = [{"id": "hf_0000", "question": "Q?", "expected_documents": ["HF-0000"],
                       "ground_truth_context": "A.", "ideal_answer_points": "A.",
                       "intent": "document_content", "expected_chunk_keywords": None,
                       "relevance_context_for_ir": "A.", "_source": "hf",
                       "_source_doc_name": "ds row 0", "_doc_number": "HF-0000"}]
        with patch.object(engine, "_load_from_hf", return_value=(fake_chunks, fake_cases)):
            with patch.object(engine, "_run_pipeline", return_value={}) as mock_pipeline:
                engine.ingest_from_hf(
                    dataset_id="some/dataset",
                    domain_name="Bio",
                    output_dir=tmp_path,
                    n_cases=1,
                )
        mock_pipeline.assert_called_once()
        kwargs = mock_pipeline.call_args[1]
        assert kwargs["hf_cases"] == fake_cases
        assert "HF-0000" in kwargs["doc_texts"]
