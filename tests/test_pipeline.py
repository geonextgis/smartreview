"""End-to-end test for the high-level ``rank_papers`` pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from smartreview import rank_papers


def test_rank_papers_end_to_end(tmp_path, sample_dataframe, mock_openai_client):
    input_path = tmp_path / "papers.xlsx"
    try:
        sample_dataframe.to_excel(input_path, index=False)
    except ImportError:
        pytest.skip("openpyxl not installed")

    out_dir = tmp_path / "out"
    result = rank_papers(
        input_path=input_path,
        interest_text="Crop yield prediction with deep learning and remote sensing.",
        output_dir=out_dir,
        top_k=2,
        client=mock_openai_client,
    )
    df = result["dataframe"]
    assert len(df) == 2
    assert list(df.columns)[:2] == ["Rank", "Similarity_Score"]
    assert (out_dir / "top_2_papers_complete.csv").exists()
    assert (out_dir / "top_2_papers.bib").exists()


def test_rank_papers_top_percentile(tmp_path, sample_dataframe, mock_openai_client):
    input_path = tmp_path / "papers.xlsx"
    try:
        sample_dataframe.to_excel(input_path, index=False)
    except ImportError:
        pytest.skip("openpyxl not installed")

    result = rank_papers(
        input_path=input_path,
        interest_text="any topic",
        output_dir=tmp_path / "out",
        top_percentile=50.0,
        client=mock_openai_client,
    )
    # Half (or close to half) of three papers
    assert 1 <= len(result["dataframe"]) <= 3


def test_rank_papers_rejects_both_selectors(tmp_path, sample_dataframe, mock_openai_client):
    input_path = tmp_path / "papers.xlsx"
    try:
        sample_dataframe.to_excel(input_path, index=False)
    except ImportError:
        pytest.skip("openpyxl not installed")

    with pytest.raises(ValueError):
        rank_papers(
            input_path=input_path,
            interest_text="x",
            output_dir=tmp_path / "out",
            top_k=2,
            top_percentile=80.0,
            client=mock_openai_client,
        )
