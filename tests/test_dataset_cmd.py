"""Tests for evalytic dataset command -- all offline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from evalytic.cli.main import cli


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def text2img_dataset(tmp_path: Path) -> Path:
    data = {
        "name": "test-text2img",
        "pipeline": "text2img",
        "items": [
            {"prompt": "A red apple", "expected": {"visual_quality": 4.5, "overall": 4.0}},
            {"prompt": "A blue car", "metadata": {"category": "vehicle"}},
        ],
    }
    path = tmp_path / "text2img.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture()
def img2img_dataset(tmp_path: Path) -> Path:
    data = {
        "name": "test-img2img",
        "pipeline": "img2img",
        "items": [
            {
                "image_url": "https://example.com/input.jpg",
                "instruction": "Place in kitchen",
                "expected": {"input_fidelity": 4.0},
            },
        ],
    }
    path = tmp_path / "img2img.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture()
def sample_bench_report(tmp_path: Path) -> Path:
    data = {
        "name": "test-bench",
        "version": "1.0",
        "config": {"pipeline": "text2img"},
        "winner": "flux-pro",
        "items": [
            {
                "item_id": "0",
                "prompt": "A red apple",
                "results": {
                    "flux-schnell": {
                        "image_url": "https://example.com/schnell.jpg",
                        "overall_score": 3.8,
                        "scores": {
                            "visual_quality": {"score": 4.0},
                            "prompt_adherence": {"score": 3.6},
                        },
                    },
                    "flux-pro": {
                        "image_url": "https://example.com/pro.jpg",
                        "overall_score": 4.5,
                        "scores": {
                            "visual_quality": {"score": 4.8},
                            "prompt_adherence": {"score": 4.2},
                        },
                    },
                },
            },
            {
                "item_id": "1",
                "prompt": "A blue car",
                "results": {
                    "flux-schnell": {
                        "image_url": "https://example.com/schnell2.jpg",
                        "overall_score": 3.2,
                        "scores": {
                            "visual_quality": {"score": 3.5},
                            "prompt_adherence": {"score": 2.9},
                        },
                    },
                    "flux-pro": {
                        "image_url": "https://example.com/pro2.jpg",
                        "overall_score": 4.0,
                        "scores": {
                            "visual_quality": {"score": 4.2},
                            "prompt_adherence": {"score": 3.8},
                        },
                    },
                },
            },
        ],
        "summary": {
            "flux-schnell": {
                "overall_score": 3.5,
                "dimension_averages": {"visual_quality": 3.75, "prompt_adherence": 3.25},
            },
            "flux-pro": {
                "overall_score": 4.25,
                "dimension_averages": {"visual_quality": 4.5, "prompt_adherence": 4.0},
            },
        },
    }
    path = tmp_path / "bench-report.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture()
def legacy_prompts_file(tmp_path: Path) -> Path:
    data = {"prompts": ["A cat", "A dog", "A bird"]}
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture()
def legacy_inputs_file(tmp_path: Path) -> Path:
    data = {
        "inputs": [
            {"image_url": "https://example.com/a.jpg", "instruction": "Edit A"},
            {"image_url": "https://example.com/b.jpg", "instruction": "Edit B"},
        ]
    }
    path = tmp_path / "legacy_inputs.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture()
def plain_array_file(tmp_path: Path) -> Path:
    data = ["Prompt one", "Prompt two"]
    path = tmp_path / "plain.json"
    path.write_text(json.dumps(data))
    return path


# ---------------------------------------------------------------------------
# dataset create
# ---------------------------------------------------------------------------


class TestDatasetCreate:
    def test_creates_empty_dataset(self, runner: CliRunner, tmp_path: Path) -> None:
        out = tmp_path / "golden.json"
        result = runner.invoke(cli, ["dataset", "create", "--name", "golden-v1", "-o", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["name"] == "golden-v1"
        assert data["pipeline"] == "text2img"
        assert data["items"] == []
        assert "created_at" in data

    def test_file_exists_exits_2(self, runner: CliRunner, tmp_path: Path) -> None:
        existing = tmp_path / "exists.json"
        existing.write_text("{}")
        result = runner.invoke(cli, ["dataset", "create", "--name", "test", "-o", str(existing)])
        assert result.exit_code == 2
        assert "already exists" in result.output

    def test_pipeline_img2img(self, runner: CliRunner, tmp_path: Path) -> None:
        out = tmp_path / "img.json"
        result = runner.invoke(cli, [
            "dataset", "create", "--name", "img-test", "--pipeline", "img2img", "-o", str(out),
        ])
        assert result.exit_code == 0
        data = json.loads(out.read_text())
        assert data["pipeline"] == "img2img"

    def test_custom_description(self, runner: CliRunner, tmp_path: Path) -> None:
        out = tmp_path / "desc.json"
        result = runner.invoke(cli, [
            "dataset", "create", "--name", "with-desc", "-d", "My description", "-o", str(out),
        ])
        assert result.exit_code == 0
        data = json.loads(out.read_text())
        assert data["description"] == "My description"


# ---------------------------------------------------------------------------
# dataset show
# ---------------------------------------------------------------------------


class TestDatasetShow:
    def test_show_text2img(self, runner: CliRunner, text2img_dataset: Path) -> None:
        result = runner.invoke(cli, ["dataset", "show", str(text2img_dataset)])
        assert result.exit_code == 0
        assert "test-text2img" in result.output
        assert "A red apple" in result.output
        assert "2 items" in result.output

    def test_show_img2img(self, runner: CliRunner, img2img_dataset: Path) -> None:
        result = runner.invoke(cli, ["dataset", "show", str(img2img_dataset)])
        assert result.exit_code == 0
        assert "test-img2img" in result.output
        assert "Place in kitchen" in result.output

    def test_file_not_found_exits_2(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["dataset", "show", "/nonexistent/dataset.json"])
        assert result.exit_code == 2
        assert "not found" in result.output


# ---------------------------------------------------------------------------
# dataset add
# ---------------------------------------------------------------------------


class TestDatasetAdd:
    def test_add_prompt(self, runner: CliRunner, text2img_dataset: Path) -> None:
        result = runner.invoke(cli, [
            "dataset", "add", str(text2img_dataset), "--prompt", "A green tree",
        ])
        assert result.exit_code == 0
        assert "3 items" in result.output
        data = json.loads(text2img_dataset.read_text())
        assert data["items"][-1]["prompt"] == "A green tree"

    def test_add_image_instruction(self, runner: CliRunner, img2img_dataset: Path) -> None:
        result = runner.invoke(cli, [
            "dataset", "add", str(img2img_dataset),
            "--image", "https://example.com/new.jpg",
            "--instruction", "Place in office",
        ])
        assert result.exit_code == 0
        assert "2 items" in result.output
        data = json.loads(img2img_dataset.read_text())
        assert data["items"][-1]["image_url"] == "https://example.com/new.jpg"

    def test_add_with_metadata_and_expected(self, runner: CliRunner, text2img_dataset: Path) -> None:
        result = runner.invoke(cli, [
            "dataset", "add", str(text2img_dataset),
            "--prompt", "A cat",
            "-m", "category=animal",
            "-e", "visual_quality:4.5",
            "-e", "overall:4.0",
        ])
        assert result.exit_code == 0
        data = json.loads(text2img_dataset.read_text())
        last = data["items"][-1]
        assert last["metadata"] == {"category": "animal"}
        assert last["expected"]["visual_quality"] == 4.5
        assert last["expected"]["overall"] == 4.0

    def test_add_file_not_found_exits_2(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, [
            "dataset", "add", "/nonexistent.json", "--prompt", "test",
        ])
        assert result.exit_code == 2
        assert "not found" in result.output


# ---------------------------------------------------------------------------
# dataset from-bench
# ---------------------------------------------------------------------------


class TestDatasetFromBench:
    def test_creates_dataset_from_report(self, runner: CliRunner, sample_bench_report: Path, tmp_path: Path) -> None:
        out = tmp_path / "golden.json"
        result = runner.invoke(cli, [
            "dataset", "from-bench", str(sample_bench_report), "-o", str(out),
        ])
        assert result.exit_code == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert len(data["items"]) == 2
        assert data["items"][0]["prompt"] == "A red apple"
        assert "flux-pro" in result.output  # winner model

    def test_min_score_filter(self, runner: CliRunner, sample_bench_report: Path, tmp_path: Path) -> None:
        out = tmp_path / "filtered.json"
        result = runner.invoke(cli, [
            "dataset", "from-bench", str(sample_bench_report),
            "--min-score", "4.3", "-o", str(out),
        ])
        assert result.exit_code == 0
        data = json.loads(out.read_text())
        # Only item 0 (flux-pro overall 4.5) should pass
        assert len(data["items"]) == 1
        assert data["items"][0]["prompt"] == "A red apple"

    def test_model_filter(self, runner: CliRunner, sample_bench_report: Path, tmp_path: Path) -> None:
        out = tmp_path / "schnell.json"
        result = runner.invoke(cli, [
            "dataset", "from-bench", str(sample_bench_report),
            "--model", "flux-schnell", "-o", str(out),
        ])
        assert result.exit_code == 0
        data = json.loads(out.read_text())
        # Expected scores should come from flux-schnell
        expected = data["items"][0].get("expected", {})
        assert expected["visual_quality"] == 4.0  # flux-schnell score

    def test_expected_scores_written(self, runner: CliRunner, sample_bench_report: Path, tmp_path: Path) -> None:
        out = tmp_path / "expected.json"
        result = runner.invoke(cli, [
            "dataset", "from-bench", str(sample_bench_report), "-o", str(out),
        ])
        assert result.exit_code == 0
        data = json.loads(out.read_text())
        expected = data["items"][0]["expected"]
        assert "visual_quality" in expected
        assert "prompt_adherence" in expected
        assert "overall" in expected

    def test_zero_items_exits_2(self, runner: CliRunner, sample_bench_report: Path, tmp_path: Path) -> None:
        out = tmp_path / "empty.json"
        result = runner.invoke(cli, [
            "dataset", "from-bench", str(sample_bench_report),
            "--min-score", "5.0", "-o", str(out),
        ])
        assert result.exit_code == 2
        assert "No items pass" in result.output


# ---------------------------------------------------------------------------
# dataset validate
# ---------------------------------------------------------------------------


class TestDatasetValidate:
    def test_valid_dataset(self, runner: CliRunner, text2img_dataset: Path) -> None:
        result = runner.invoke(cli, ["dataset", "validate", str(text2img_dataset)])
        assert result.exit_code == 0
        assert "Valid" in result.output

    def test_warnings_shown(self, runner: CliRunner, tmp_path: Path) -> None:
        data = {
            "pipeline": "text2img",
            "items": [
                {"prompt": "test", "expected": {"visual_quality": 6.0}},
            ],
        }
        path = tmp_path / "warn.json"
        path.write_text(json.dumps(data))
        result = runner.invoke(cli, ["dataset", "validate", str(path)])
        assert result.exit_code == 0
        assert "outside 0-5" in result.output

    def test_invalid_json_exits_2(self, runner: CliRunner, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("not json{{{")
        result = runner.invoke(cli, ["dataset", "validate", str(bad)])
        assert result.exit_code == 2
        assert "Invalid JSON" in result.output


# ---------------------------------------------------------------------------
# dataset stats
# ---------------------------------------------------------------------------


class TestDatasetStats:
    def test_shows_expected_stats(self, runner: CliRunner, text2img_dataset: Path) -> None:
        result = runner.invoke(cli, ["dataset", "stats", str(text2img_dataset)])
        assert result.exit_code == 0
        assert "Items: 2" in result.output
        assert "Items with expected scores: 1" in result.output
        assert "visual_quality" in result.output

    def test_shows_metadata_keys(self, runner: CliRunner, text2img_dataset: Path) -> None:
        result = runner.invoke(cli, ["dataset", "stats", str(text2img_dataset)])
        assert result.exit_code == 0
        assert "category" in result.output

    def test_no_expected(self, runner: CliRunner, tmp_path: Path) -> None:
        data = {"items": [{"prompt": "A cat"}, {"prompt": "A dog"}]}
        path = tmp_path / "no_expected.json"
        path.write_text(json.dumps(data))
        result = runner.invoke(cli, ["dataset", "stats", str(path)])
        assert result.exit_code == 0
        assert "Items: 2" in result.output
        assert "Items with expected scores: 0" in result.output


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_legacy_prompts_format(self, runner: CliRunner, legacy_prompts_file: Path) -> None:
        result = runner.invoke(cli, ["dataset", "show", str(legacy_prompts_file)])
        assert result.exit_code == 0
        assert "3 items" in result.output
        assert "A cat" in result.output

    def test_legacy_inputs_format(self, runner: CliRunner, legacy_inputs_file: Path) -> None:
        result = runner.invoke(cli, ["dataset", "show", str(legacy_inputs_file)])
        assert result.exit_code == 0
        assert "2 items" in result.output
        assert "Edit A" in result.output

    def test_plain_array_format(self, runner: CliRunner, plain_array_file: Path) -> None:
        result = runner.invoke(cli, ["dataset", "show", str(plain_array_file)])
        assert result.exit_code == 0
        assert "2 items" in result.output
        assert "Prompt one" in result.output

    def test_add_preserves_prompts_key(self, runner: CliRunner, legacy_prompts_file: Path) -> None:
        """Adding to a legacy 'prompts' file should preserve that key."""
        result = runner.invoke(cli, [
            "dataset", "add", str(legacy_prompts_file), "--prompt", "A fish",
        ])
        assert result.exit_code == 0
        data = json.loads(legacy_prompts_file.read_text())
        assert "prompts" in data
        assert len(data["prompts"]) == 4

    def test_validate_legacy_prompts(self, runner: CliRunner, legacy_prompts_file: Path) -> None:
        result = runner.invoke(cli, ["dataset", "validate", str(legacy_prompts_file)])
        assert result.exit_code == 0

    def test_stats_legacy_inputs(self, runner: CliRunner, legacy_inputs_file: Path) -> None:
        result = runner.invoke(cli, ["dataset", "stats", str(legacy_inputs_file)])
        assert result.exit_code == 0
        assert "Items: 2" in result.output
