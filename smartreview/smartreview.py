"""Core SmartReview utilities.

Similarity computation, top-K extraction, DataFrame export, BibTeX
generation, and a high-level :func:`rank_papers` pipeline.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .embeddings import (
    DEFAULT_MODEL,
    EmbeddingCache,
    create_openai_client,
    embed_papers,
    get_embedding,
)

PathLike = Union[str, Path]
SimilarityList = List[Tuple[int, float]]


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------


def calculate_cosine_similarity(
    query_embedding: np.ndarray,
    paper_embeddings_dict: Dict[int, np.ndarray],
) -> SimilarityList:
    """Compute cosine similarity between a query and many paper embeddings.

    Args:
        query_embedding: 1-D embedding for the query.
        paper_embeddings_dict: ``{paper_idx: embedding}`` mapping.

    Returns:
        List of ``(paper_idx, similarity)`` tuples sorted by descending
        similarity.
    """
    if not paper_embeddings_dict:
        return []

    indices = list(paper_embeddings_dict.keys())
    matrix = np.vstack(
        [np.asarray(paper_embeddings_dict[i], dtype=np.float32) for i in indices]
    )
    query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)

    matrix_norms = np.linalg.norm(matrix, axis=1)
    query_norm = np.linalg.norm(query)
    denom = matrix_norms * query_norm
    denom[denom == 0] = 1.0  # avoid division by zero
    scores = matrix @ query / denom

    pairs = list(zip(indices, scores.tolist()))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


def get_top_k_papers(similarities: SimilarityList, k: int = 100) -> SimilarityList:
    """Return the first ``k`` items from a sorted similarity list."""
    if k <= 0:
        return []
    return similarities[: min(k, len(similarities))]


def top_k_by_percentile(
    similarities: SimilarityList, percentile: float = 80.0
) -> SimilarityList:
    """Return all papers with a similarity at or above a given percentile.

    Args:
        similarities: Output of :func:`calculate_cosine_similarity`.
        percentile: Percentile threshold in ``[0, 100]``. ``80`` keeps the
            top 20%.
    """
    if not similarities:
        return []
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be in [0, 100]")
    threshold = float(np.percentile([s for _, s in similarities], percentile))
    return [pair for pair in similarities if pair[1] >= threshold]


# ---------------------------------------------------------------------------
# DataFrame & export
# ---------------------------------------------------------------------------


def create_top_k_dataframe(
    top_k_papers: SimilarityList,
    data: pd.DataFrame,
    summary: Optional[Dict[int, Tuple[str, str]]] = None,
) -> pd.DataFrame:
    """Build a ranked DataFrame from a top-K similarity list.

    Args:
        top_k_papers: ``[(paper_idx, similarity), ...]``.
        data: Source DataFrame whose row at ``paper_idx`` will be merged in.
        summary: Unused; kept for backwards compatibility.

    Returns:
        DataFrame ranked by similarity with ``Rank`` and ``Similarity_Score``
        as the first two columns.
    """
    del summary  # accepted for backwards compatibility, not needed
    rows = []
    for rank, (paper_idx, similarity) in enumerate(top_k_papers, start=1):
        row = {"Rank": rank, "Similarity_Score": float(similarity)}
        row.update(data.iloc[paper_idx].to_dict())
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by="Rank", ascending=True).reset_index(drop=True)
    return df


def save_top_k_papers(
    top_k_df: pd.DataFrame,
    output_dir: PathLike = "data",
    k: Optional[int] = None,
) -> Dict[str, Any]:
    """Write the top-K DataFrame to CSV and (optionally) Excel.

    Args:
        top_k_df: DataFrame produced by :func:`create_top_k_dataframe`.
        output_dir: Destination directory; created if missing.
        k: Number of papers, used in the filename. Defaults to ``len(df)``.

    Returns:
        Dict with ``csv``, ``excel``, ``rows``, and ``columns`` entries.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    k = k if k is not None else len(top_k_df)

    csv_file = out / f"top_{k}_papers_complete.csv"
    excel_file: Optional[Path] = out / f"top_{k}_papers_complete.xlsx"

    top_k_df.to_csv(csv_file, index=False, encoding="utf-8")
    try:
        top_k_df.to_excel(excel_file, index=False, engine="openpyxl")
    except ImportError:
        print("Warning: openpyxl not installed; skipping .xlsx export.")
        excel_file = None

    return {
        "csv": str(csv_file),
        "excel": str(excel_file) if excel_file else None,
        "rows": len(top_k_df),
        "columns": len(top_k_df.columns),
    }


def _safe(value: Any) -> str:
    return "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)


def create_bibtex_entry(rank: int, paper_data: Dict[str, Any]) -> str:
    """Render a BibTeX entry for one paper row.

    Args:
        rank: 1-based rank used to disambiguate citation keys.
        paper_data: Row dict (e.g. from ``DataFrame.to_dict()``) using the
            standard Web of Science column names.
    """
    authors = _safe(paper_data.get("Authors"))
    year = paper_data.get("Publication Year")
    title = _safe(paper_data.get("Article Title"))
    journal = _safe(paper_data.get("Source Title"))
    volume = _safe(paper_data.get("Volume"))
    issue = _safe(paper_data.get("Issue"))
    pages = _safe(paper_data.get("Article Number"))
    doi = _safe(paper_data.get("DOI"))
    abstract = _safe(paper_data.get("Abstract"))

    first_author = (
        authors.split(";")[0].split(",")[0].strip().replace(" ", "")
        if authors
        else "Unknown"
    )
    year_str = (
        str(int(year))
        if year is not None and not (isinstance(year, float) and pd.isna(year))
        else "n.d."
    )
    citation_key = f"{first_author}_{year_str}_{rank}"

    lines = [f"@article{{{citation_key},"]
    if authors:
        lines.append(f"  author = {{{authors.replace(';', ' and ')}}},")
    if title:
        lines.append(f"  title = {{{title}}},")
    if journal:
        lines.append(f"  journal = {{{journal}}},")
    if year_str != "n.d.":
        lines.append(f"  year = {{{year_str}}},")
    if volume:
        lines.append(f"  volume = {{{volume}}},")
    if issue:
        lines.append(f"  number = {{{issue}}},")
    if pages:
        lines.append(f"  pages = {{{pages}}},")
    if doi:
        lines.append(f"  doi = {{{doi}}},")
    if abstract:
        snippet = abstract[:200] + ("..." if len(abstract) > 200 else "")
        lines.append(f"  abstract = {{{snippet}}},")

    body = "\n".join(lines).rstrip(",")
    return body + "\n}\n\n"


def generate_bibtex_file(
    top_k_df: pd.DataFrame,
    output_dir: PathLike = "data",
    k: Optional[int] = None,
) -> Dict[str, Any]:
    """Write a ``.bib`` file containing one entry per row of ``top_k_df``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    k = k if k is not None else len(top_k_df)

    bibtex = "".join(
        create_bibtex_entry(int(row["Rank"]), row.to_dict())
        for _, row in top_k_df.iterrows()
    )
    bibtex_file = out / f"top_{k}_papers.bib"
    bibtex_file.write_text(bibtex, encoding="utf-8")
    return {
        "file": str(bibtex_file),
        "entries": len(top_k_df),
        "file_size": bibtex_file.stat().st_size,
    }


def print_top_k_summary(
    top_k_df: pd.DataFrame, k: Optional[int] = None, show_rows: int = 10
) -> None:
    """Pretty-print a brief summary of the top-K DataFrame."""
    k = k if k is not None else len(top_k_df)
    print(f"\nTop {k} papers summary:")
    print(f"  rows: {len(top_k_df)}, columns: {len(top_k_df.columns)}")
    cols = ["Rank", "Similarity_Score", "Article Title", "Publication Year"]
    available = [c for c in cols if c in top_k_df.columns]
    print("-" * 80)
    print(top_k_df[available].head(show_rows).to_string(index=False))


# ---------------------------------------------------------------------------
# Embedding persistence (legacy pickle layout)
# ---------------------------------------------------------------------------


def save_embeddings(
    paper_embeddings: Dict[int, np.ndarray],
    interest_embedding: np.ndarray,
    interest_text: str,
    output_dir: PathLike = "data/embeddings",
) -> Dict[str, str]:
    """Pickle paper embeddings, interest embedding, and the interest text."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paper_file = out / "paper_embeddings.pkl"
    interest_file = out / "interest_embedding.pkl"
    text_file = out / "interest_text.txt"

    with open(paper_file, "wb") as f:
        pickle.dump(paper_embeddings, f)
    with open(interest_file, "wb") as f:
        pickle.dump(interest_embedding, f)
    text_file.write_text(interest_text, encoding="utf-8")

    return {
        "paper_embeddings": str(paper_file),
        "interest_embedding": str(interest_file),
        "interest_text": str(text_file),
    }


def load_embeddings(
    output_dir: PathLike = "data/embeddings",
) -> Tuple[Dict[int, np.ndarray], np.ndarray, str]:
    """Load embeddings previously written by :func:`save_embeddings`."""
    out = Path(output_dir)
    with open(out / "paper_embeddings.pkl", "rb") as f:
        paper_embeddings = pickle.load(f)
    with open(out / "interest_embedding.pkl", "rb") as f:
        interest_embedding = pickle.load(f)
    interest_text = (out / "interest_text.txt").read_text(encoding="utf-8")
    return paper_embeddings, interest_embedding, interest_text


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_wos_export(
    path: PathLike,
    title_col: str = "Article Title",
    abstract_col: str = "Abstract",
) -> Tuple[pd.DataFrame, Dict[int, Tuple[str, str]]]:
    """Read a Web of Science ``.xls``/``.xlsx`` export.

    Args:
        path: Path to the export file.
        title_col: Column holding the paper title.
        abstract_col: Column holding the paper abstract.

    Returns:
        Tuple ``(dataframe, summary)`` where ``summary`` maps row index to
        ``(title, abstract)``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_excel(path)
    for col in (title_col, abstract_col):
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in {path}. "
                f"Available columns: {list(df.columns)}"
            )
    summary = {
        i: (str(row[title_col]), "" if pd.isna(row[abstract_col]) else str(row[abstract_col]))
        for i, row in df.iterrows()
    }
    return df, summary


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------


def rank_papers(
    input_path: PathLike,
    interest_text: str,
    output_dir: PathLike = "data",
    top_k: Optional[int] = None,
    top_percentile: Optional[float] = None,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    client=None,
    cache_dir: Optional[PathLike] = None,
    export: bool = True,
) -> Dict[str, Any]:
    """End-to-end ranking pipeline.

    Reads a Web of Science export, embeds the papers and the interest
    statement, and (by default) writes CSV, Excel, and BibTeX files for the
    top-ranked subset.

    Exactly one of ``top_k`` or ``top_percentile`` should be supplied; if
    neither is given, ``top_k=100`` is used.

    Args:
        input_path: Path to the ``.xls``/``.xlsx`` Web of Science export.
        interest_text: Free-text description of your research interests.
        output_dir: Where exports and the embedding cache live.
        top_k: Keep the top ``k`` papers.
        top_percentile: Alternatively, keep papers above this similarity
            percentile (e.g. ``80`` keeps the top 20%).
        model: OpenAI embedding model.
        api_key: Optional explicit API key. Falls back to ``OPENAI_API_KEY``.
        client: Pre-built OpenAI client. Mostly useful for testing.
        cache_dir: Embedding-cache directory. Defaults to
            ``<output_dir>/embeddings/cache``.
        export: When ``True`` (default), write CSV/Excel/BibTeX files.

    Returns:
        A dict with ``dataframe``, ``similarities``, ``top_k``, and (if
        ``export``) the file paths produced by :func:`save_top_k_papers`
        and :func:`generate_bibtex_file`.
    """
    if top_k is not None and top_percentile is not None:
        raise ValueError("Pass either top_k or top_percentile, not both.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = EmbeddingCache(cache_dir or output_dir / "embeddings" / "cache")

    if client is None:
        client = create_openai_client(api_key=api_key)

    data, summary = load_wos_export(input_path)
    print(f"Loaded {len(data)} papers from {input_path}")

    print("Embedding papers...")
    paper_embeddings = embed_papers(
        summary, client=client, model=model, cache=cache
    )
    print("Embedding interest statement...")
    interest_embedding = get_embedding(
        interest_text, client=client, model=model, cache=cache
    )

    similarities = calculate_cosine_similarity(interest_embedding, paper_embeddings)

    if top_percentile is not None:
        selected = top_k_by_percentile(similarities, percentile=top_percentile)
    else:
        selected = get_top_k_papers(similarities, k=top_k or 100)

    df = create_top_k_dataframe(selected, data)
    result: Dict[str, Any] = {
        "dataframe": df,
        "similarities": similarities,
        "top_k": selected,
    }

    if export:
        k_label = len(selected)
        result["files"] = save_top_k_papers(df, output_dir=output_dir, k=k_label)
        result["bibtex"] = generate_bibtex_file(df, output_dir=output_dir, k=k_label)
        save_embeddings(
            paper_embeddings,
            interest_embedding,
            interest_text,
            output_dir=output_dir / "embeddings",
        )
    return result
