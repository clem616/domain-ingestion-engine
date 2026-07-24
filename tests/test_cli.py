"""Tests for __main__.py — CLI entry point and _load_env."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from domain_ingestion_engine.__main__ import _load_env, main
from domain_ingestion_engine.engine import DomainIngestionEngine


# ── _load_env ─────────────────────────────────────────────────────────────────

class TestLoadEnv:
    def test_dotenv_installed_calls_load_dotenv(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("ANTHROPIC_API_KEY=from_dotenv\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        mock_dotenv = MagicMock()
        with patch.dict(sys.modules, {"dotenv": mock_dotenv}):
            mock_dotenv.load_dotenv = MagicMock()
            _load_env()
        mock_dotenv.load_dotenv.assert_called_once()

    def test_no_dotenv_parses_manually(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "# comment line\n"
            "ANTHROPIC_API_KEY=manual_key\n"
            "UNRELATED_VAR=should_be_ignored\n"
            "LLM_MODEL_CLAUDE_HAIKU=claude-haiku-4-5\n"
            "\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("LLM_MODEL_CLAUDE_HAIKU", raising=False)

        with patch.dict(sys.modules, {"dotenv": None}):
            _load_env()

        assert os.environ.get("ANTHROPIC_API_KEY") == "manual_key"
        assert os.environ.get("LLM_MODEL_CLAUDE_HAIKU") == "claude-haiku-4-5"
        # Unrelated vars not in the allowlist must not be set
        assert "UNRELATED_VAR" not in os.environ

    def test_no_env_file_does_nothing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        before = dict(os.environ)
        with patch.dict(sys.modules, {"dotenv": None}):
            _load_env()
        # No new keys added (except possibly others)
        for k in ["ANTHROPIC_API_KEY", "LLM_API_KEY_CLAUDE", "LLM_BASE_URL_CLAUDE"]:
            if k not in before:
                assert k not in os.environ

    def test_parent_dir_env_used_when_cwd_missing(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("ANTHROPIC_API_KEY=parent_key\n", encoding="utf-8")
        child = tmp_path / "child"
        child.mkdir()
        monkeypatch.chdir(child)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with patch.dict(sys.modules, {"dotenv": None}):
            _load_env()

        assert os.environ.get("ANTHROPIC_API_KEY") == "parent_key"

    def test_comment_lines_skipped(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "# this is a comment\n"
            "ANTHROPIC_API_KEY=real_key\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with patch.dict(sys.modules, {"dotenv": None}):
            _load_env()

        assert os.environ.get("ANTHROPIC_API_KEY") == "real_key"

    def test_line_without_equals_skipped(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("NOT_A_KV_LINE\nANTHROPIC_API_KEY=valid\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with patch.dict(sys.modules, {"dotenv": None}):
            _load_env()

        assert os.environ.get("ANTHROPIC_API_KEY") == "valid"

    def test_setdefault_does_not_override_existing(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("ANTHROPIC_API_KEY=from_file\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "already_set")

        with patch.dict(sys.modules, {"dotenv": None}):
            _load_env()

        assert os.environ.get("ANTHROPIC_API_KEY") == "already_set"

    def test_all_allowed_keys_are_parsed(self, tmp_path, monkeypatch):
        content = (
            "ANTHROPIC_API_KEY=k1\n"
            "LLM_API_KEY_CLAUDE=k2\n"
            "LLM_BASE_URL_CLAUDE=http://base\n"
            "LLM_MODEL_CLAUDE=model-x\n"
            "LLM_MODEL_CLAUDE_HAIKU=haiku-x\n"
        )
        env_file = tmp_path / ".env"
        env_file.write_text(content, encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        for k in ["ANTHROPIC_API_KEY", "LLM_API_KEY_CLAUDE", "LLM_BASE_URL_CLAUDE",
                   "LLM_MODEL_CLAUDE", "LLM_MODEL_CLAUDE_HAIKU"]:
            monkeypatch.delenv(k, raising=False)

        with patch.dict(sys.modules, {"dotenv": None}):
            _load_env()

        assert os.environ.get("LLM_API_KEY_CLAUDE") == "k2"
        assert os.environ.get("LLM_BASE_URL_CLAUDE") == "http://base"
        assert os.environ.get("LLM_MODEL_CLAUDE") == "model-x"


# ── main — docs-dir path ──────────────────────────────────────────────────────

class TestMainDocsDir:
    def test_docs_dir_calls_ingest(self, tmp_path, monkeypatch):
        (tmp_path / "doc.txt").write_text("content", encoding="utf-8")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        fake_paths = {"ir": str(out_dir / "ground_truth_ir.json")}
        with patch("domain_ingestion_engine.__main__._load_env"):
            with patch("domain_ingestion_engine.__main__.DomainIngestionEngine") as MockEngine:
                instance = MagicMock()
                instance.ingest.return_value = fake_paths
                MockEngine.return_value = instance

                monkeypatch.setattr(
                    sys, "argv",
                    ["prog", "--docs-dir", str(tmp_path),
                     "--domain-name", "Legal", "--output-dir", str(out_dir)]
                )
                main()

        instance.ingest.assert_called_once()
        call_kwargs = instance.ingest.call_args[1]
        assert call_kwargs["domain_name"] == "Legal"
        assert call_kwargs["adversarial_only"] is False
        assert call_kwargs["seed_ir_cases"] is None

    def test_no_api_key_exits(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("LLM_API_KEY_CLAUDE", raising=False)

        with patch("domain_ingestion_engine.__main__._load_env"):
            monkeypatch.setattr(
                sys, "argv",
                ["prog", "--docs-dir", str(tmp_path),
                 "--domain-name", "Test", "--output-dir", str(tmp_path)]
            )
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1

    def test_adversarial_only_with_source_ir_path(self, tmp_path, monkeypatch):
        ir_file = tmp_path / "ir.json"
        ir_data = [{"id": "gen_0000", "question": "Q?"}]
        ir_file.write_text(json.dumps(ir_data), encoding="utf-8")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        with patch("domain_ingestion_engine.__main__._load_env"):
            with patch("domain_ingestion_engine.__main__.DomainIngestionEngine") as MockEngine:
                instance = MagicMock()
                instance.ingest.return_value = {}
                MockEngine.return_value = instance

                monkeypatch.setattr(
                    sys, "argv",
                    ["prog", "--docs-dir", str(tmp_path),
                     "--domain-name", "Legal", "--output-dir", str(out_dir),
                     "--adversarial-only", "--source-ir-path", str(ir_file)]
                )
                main()

        call_kwargs = instance.ingest.call_args[1]
        assert call_kwargs["adversarial_only"] is True
        assert call_kwargs["seed_ir_cases"] == ir_data

    def test_adversarial_only_without_source_ir_path(self, tmp_path, monkeypatch):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        with patch("domain_ingestion_engine.__main__._load_env"):
            with patch("domain_ingestion_engine.__main__.DomainIngestionEngine") as MockEngine:
                instance = MagicMock()
                instance.ingest.return_value = {}
                MockEngine.return_value = instance

                monkeypatch.setattr(
                    sys, "argv",
                    ["prog", "--docs-dir", str(tmp_path),
                     "--domain-name", "Legal", "--output-dir", str(out_dir),
                     "--adversarial-only"]
                )
                main()

        call_kwargs = instance.ingest.call_args[1]
        assert call_kwargs["adversarial_only"] is True
        assert call_kwargs["seed_ir_cases"] is None


# ── main — hf-dataset path ────────────────────────────────────────────────────

class TestMainHfDataset:
    def test_hf_dataset_calls_ingest_from_hf(self, tmp_path, monkeypatch):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        with patch("domain_ingestion_engine.__main__._load_env"):
            with patch("domain_ingestion_engine.__main__.DomainIngestionEngine") as MockEngine:
                instance = MagicMock()
                instance.ingest_from_hf.return_value = {}
                MockEngine.return_value = instance

                monkeypatch.setattr(
                    sys, "argv",
                    ["prog", "--hf-dataset", "qiaojin/PubMedQA",
                     "--domain-name", "Bio", "--output-dir", str(out_dir),
                     "--hf-config", "pqa_labeled", "--hf-split", "validation"]
                )
                main()

        call_kwargs = instance.ingest_from_hf.call_args[1]
        assert call_kwargs["dataset_id"] == "qiaojin/PubMedQA"
        assert call_kwargs["hf_split"] == "validation"
        assert call_kwargs["hf_config"] == "pqa_labeled"
        assert call_kwargs["domain_name"] == "Bio"

    def test_hf_cache_dir_passed(self, tmp_path, monkeypatch):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        cache = tmp_path / "hf_cache"
        cache.mkdir()
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        with patch("domain_ingestion_engine.__main__._load_env"):
            with patch("domain_ingestion_engine.__main__.DomainIngestionEngine") as MockEngine:
                instance = MagicMock()
                instance.ingest_from_hf.return_value = {}
                MockEngine.return_value = instance

                monkeypatch.setattr(
                    sys, "argv",
                    ["prog", "--hf-dataset", "some/ds",
                     "--domain-name", "Legal", "--output-dir", str(out_dir),
                     "--hf-cache-dir", str(cache)]
                )
                main()

        call_kwargs = instance.ingest_from_hf.call_args[1]
        assert call_kwargs["cache_dir"] == cache


# ── main — model override ─────────────────────────────────────────────────────

class TestMainModelOverride:
    def test_env_model_overrides_default_when_arg_is_default(self, tmp_path, monkeypatch):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("LLM_MODEL_CLAUDE_HAIKU", "claude-haiku-custom")

        with patch("domain_ingestion_engine.__main__._load_env"):
            with patch("domain_ingestion_engine.__main__.DomainIngestionEngine") as MockEngine:
                instance = MagicMock()
                instance.ingest.return_value = {}
                MockEngine.return_value = instance

                monkeypatch.setattr(
                    sys, "argv",
                    ["prog", "--docs-dir", str(tmp_path),
                     "--domain-name", "X", "--output-dir", str(out_dir)]
                )
                main()

        _, kwargs = MockEngine.call_args
        assert kwargs["model"] == "claude-haiku-custom"

    def test_explicit_model_arg_not_overridden_by_env(self, tmp_path, monkeypatch):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("LLM_MODEL_CLAUDE_HAIKU", "should-not-use-this")

        with patch("domain_ingestion_engine.__main__._load_env"):
            with patch("domain_ingestion_engine.__main__.DomainIngestionEngine") as MockEngine:
                instance = MagicMock()
                instance.ingest.return_value = {}
                MockEngine.return_value = instance

                monkeypatch.setattr(
                    sys, "argv",
                    ["prog", "--docs-dir", str(tmp_path),
                     "--domain-name", "X", "--output-dir", str(out_dir),
                     "--model", "claude-sonnet-4-6"]
                )
                main()

        _, kwargs = MockEngine.call_args
        assert kwargs["model"] == "claude-sonnet-4-6"

    def test_llt_api_key_claude_also_accepted(self, tmp_path, monkeypatch):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("LLM_API_KEY_CLAUDE", "alt-key")

        with patch("domain_ingestion_engine.__main__._load_env"):
            with patch("domain_ingestion_engine.__main__.DomainIngestionEngine") as MockEngine:
                instance = MagicMock()
                instance.ingest.return_value = {}
                MockEngine.return_value = instance

                monkeypatch.setattr(
                    sys, "argv",
                    ["prog", "--docs-dir", str(tmp_path),
                     "--domain-name", "X", "--output-dir", str(out_dir)]
                )
                main()

        _, kwargs = MockEngine.call_args
        assert kwargs["anthropic_api_key"] == "alt-key"

    def test_api_base_url_from_env_passed_through(self, tmp_path, monkeypatch):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("LLM_BASE_URL_CLAUDE", "https://custom.azure.com/anthropic")

        with patch("domain_ingestion_engine.__main__._load_env"):
            with patch("domain_ingestion_engine.__main__.DomainIngestionEngine") as MockEngine:
                instance = MagicMock()
                instance.ingest.return_value = {}
                MockEngine.return_value = instance

                monkeypatch.setattr(
                    sys, "argv",
                    ["prog", "--docs-dir", str(tmp_path),
                     "--domain-name", "X", "--output-dir", str(out_dir)]
                )
                main()

        _, kwargs = MockEngine.call_args
        assert kwargs["api_base_url"] == "https://custom.azure.com/anthropic"
