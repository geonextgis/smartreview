"""Tests for similarity, ranking, DataFrame, and BibTeX utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smartreview.smartreview import (
    calculate_cosine_similarity,
    create_bibtex_entry,
    create_top_k_dataframe,
    generate_bibtex_file,
    get_top_k_papers,
    load_embeddings,
    load_wos_export,
    save_embeddings,
    save_top_k_papers,
    top_k_by_percentile,
)


def test_calculate_cosine_similarity_orders_by_score():
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    papers = {
        0: np.array([1.0, 0.0, 0.0], dtype=np.float32),  # identical → 1.0
        1: np.array([0.0, 1.0, 0.0], dtype=np.float32),  # orthogonal → 0.0
        2: np.array([0.7, 0.7, 0.0], dtype=np.float32),  # ~0.707
    }
    sims = calculate_cosine_similarity(query, papers)
    assert [idx for idx, _ in sims] == [0, 2, 1]
    assert sims[0][1] == pytest.approx(1.0, abs=1e-6)
    assert sims[2][1] == pytest.approx(0.0, abs=1e-6)


def test_calculate_cosine_similarity_handles_zero_vector():
    query = np.array([1.0, 0.0], dtype=np.float32)
    papers = {0: np.array([0.0, 0.0], dtype=np.float32)}
    sims = calculate_cosine_similarity(query, papers)
    assert sims[0][1] == pytest.approx(0.0)


def test_calculate_cosine_similarity_empty_dict():
    assert calculate_cosine_similarity(np.array([1.0]), {}) == []


def test_get_top_k_papers_basic():
    sims = [(0, 0.9), (1, 0.8), (2, 0.7)]
    assert get_top_k_papers(sims, k=2) == [(0, 0.9), (1, 0.8)]
    assert get_top_k_papers(sims, k=10) == sims
    assert get_top_k_papers(sims, k=0) == []


def test_top_k_by_percentile():
    sims = [(i, i / 10) for i in range(10)]  # scores 0.0..0.9
    sims.sort(key=lambda x: x[1], reverse=True)
    top = top_k_by_percentile(sims, percentile=80.0)
    # Top 20%: scores >= 0.72 → indices with score in {0.8, 0.9}
    assert {pair[0] for pair in top} == {8, 9}


def test_top_k_by_percentile_validates_range():
    with pytest.raises(ValueError):
        top_k_by_percentile([(0, 0.5)], percentile=150.0)


def test_create_top_k_dataframe(sample_dataframe):
    sims = [(0, 0.9), (2, 0.7)]
    df = create_top_k_dataframe(sims, sample_dataframe)
    assert list(df["Rank"]) == [1, 2]
    assert df["Similarity_Score"].tolist() == [0.9, 0.7]
    assert df.iloc[0]["Article Title"] == "Deep learning for crop yield"
    assert df.iloc[1]["Article Title"] == "Random forests for soil moisture"


def test_save_top_k_papers_writes_csv_and_excel(tmp_path, sample_dataframe):
    sims = [(0, 0.9), (1, 0.8)]
    df = create_top_k_dataframe(sims, sample_dataframe)
    info = save_top_k_papers(df, output_dir=tmp_path, k=2)
    assert info["rows"] == 2
    assert (tmp_path / "top_2_papers_complete.csv").exists()
    # Excel may be skipped if openpyxl missing; either is acceptable
    if info["excel"]:
        assert (tmp_path / "top_2_papers_complete.xlsx").exists()


def test_create_bibtex_entry_includes_required_fields(sample_dataframe):
    row = sample_dataframe.iloc[0].to_dict()
    entry = create_bibtex_entry(rank=1, paper_data=row)
    assert entry.startswith("@article{")
    assert "Smith and Doe, J" in entry or "Smith, A and Doe, J" in entry
    assert "title = {Deep learning for crop yield}" in entry
    assert "year = {2024}" in entry
    assert "doi = {10.0/abc}" in entry
    assert entry.rstrip().endswith("}")


def test_create_bibtex_entry_handles_missing_year():
    row = {"Article Title": "Untitled", "Authors": "Anon"}
    entry = create_bibtex_entry(rank=1, paper_data=row)
    assert "year =" not in entry  # n.d. case → year is omitted
    assert "Anon_n.d._1" in entry


def test_generate_bibtex_file(tmp_path, sample_dataframe):
    sims = [(0, 0.9), (1, 0.85)]
    df = create_top_k_dataframe(sims, sample_dataframe)
    info = generate_bibtex_file(df, output_dir=tmp_path, k=2)
    text = (tmp_path / "top_2_papers.bib").read_text()
    assert info["entries"] == 2
    assert text.count("@article{") == 2


def test_save_and_load_embeddings_roundtrip(tmp_path):
    paper_emb = {0: np.array([1.0, 2.0]), 1: np.array([3.0, 4.0])}
    interest_emb = np.array([5.0, 6.0])
    text = "test interest"
    save_embeddings(paper_emb, interest_emb, text, output_dir=tmp_path)
    loaded_paper, loaded_interest, loaded_text = load_embeddings(tmp_path)
    assert set(loaded_paper.keys()) == {0, 1}
    np.testing.assert_array_equal(loaded_paper[0], paper_emb[0])
    np.testing.assert_array_equal(loaded_interest, interest_emb)
    assert loaded_text == text


def test_load_wos_export_roundtrip(tmp_path, sample_dataframe):
    path = tmp_path / "papers.xlsx"
    try:
        sample_dataframe.to_excel(path, index=False)
    except ImportError:
        pytest.skip("openpyxl not installed")
    df, summary = load_wos_export(path)
    assert len(df) == 3
    assert len(summary) == 3
    assert summary[0][0] == "Deep learning for crop yield"


def test_load_wos_export_missing_columns(tmp_path):
    path = tmp_path / "bad.xlsx"
    try:
        pd.DataFrame({"Foo": [1]}).to_excel(path, index=False)
    except ImportError:
        pytest.skip("openpyxl not installed")
    with pytest.raises(ValueError):
        load_wos_export(path)


def test_load_wos_export_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_wos_export(tmp_path / "missing.xlsx")
