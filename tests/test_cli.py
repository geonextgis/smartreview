"""Smoke tests for the ``smartreview`` CLI entrypoint."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from smartreview.cli import build_parser, main


def test_parser_requires_input():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_top_k_and_percentile_mutually_exclusive():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["-i", "x.xlsx", "--interest", "y", "--top-k", "5", "--top-percentile", "80"]
        )


def test_main_invokes_rank_papers(tmp_path, sample_dataframe, mock_openai_client):
    input_path = tmp_path / "papers.xlsx"
    try:
        sample_dataframe.to_excel(input_path, index=False)
    except ImportError:
        pytest.skip("openpyxl not installed")

    with patch("smartreview.cli.rank_papers") as fake:
        fake.return_value = {"dataframe": sample_dataframe}
        rc = main([
            "--input", str(input_path),
            "--interest", "deep learning crop yield",
            "--output-dir", str(tmp_path / "out"),
            "--top-k", "2",
            "--no-export",
        ])
    assert rc == 0
    fake.assert_called_once()
    kwargs = fake.call_args.kwargs
    assert kwargs["top_k"] == 2
    assert kwargs["export"] is False


def test_main_reads_interest_file(tmp_path, sample_dataframe):
    interest_file = tmp_path / "interest.txt"
    interest_file.write_text("interest from file")
    input_path = tmp_path / "papers.xlsx"
    try:
        sample_dataframe.to_excel(input_path, index=False)
    except ImportError:
        pytest.skip("openpyxl not installed")

    with patch("smartreview.cli.rank_papers") as fake:
        fake.return_value = {"dataframe": sample_dataframe}
        rc = main([
            "--input", str(input_path),
            "--interest-file", str(interest_file),
            "--no-export",
        ])
    assert rc == 0
    assert fake.call_args.kwargs["interest_text"] == "interest from file"
