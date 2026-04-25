# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

**smartreview** is a Python package that provides an intelligent literature review tool. It uses AI-powered sentence embeddings to rank and filter research papers (e.g., exported from Web of Science) by semantic similarity to a user-defined research interest.

- **Author:** Krishnagopal Halder (geonextgis@gmail.com)
- **License:** MIT
- **Python:** ≥ 3.8
- **PyPI name:** `smartreview`

---

## Repository Layout

```
smartreview/               ← repo root
├── smartreview/           ← importable Python package
│   ├── __init__.py        ← exposes public API; sets __version__
│   ├── common.py          ← shared utility helpers
│   └── smartreview.py     ← core logic (embeddings, similarity, export)
├── tests/
│   ├── __init__.py
│   └── test_smartreview.py
├── docs/
│   ├── examples/
│   │   └── intro.ipynb    ← example notebook shown in the documentation
│   └── *.md               ← MkDocs documentation pages
├── pyproject.toml         ← PEP 517/518 build config (setuptools backend)
├── requirements.txt       ← runtime dependencies (read by pyproject.toml dynamic)
├── requirements_dev.txt   ← dev / test dependencies
├── MANIFEST.in
└── mkdocs.yml
```

---

## Package Structure Rules

1. **All public functions/classes** must live under `smartreview/` and be re-exported from `smartreview/__init__.py` so users can do `from smartreview import <symbol>`.
2. **New sub-modules** (e.g., `smartreview/embeddings.py`) should be added inside `smartreview/` and imported in `__init__.py`.
3. **Do not** add business logic directly to `__init__.py` — keep it as a thin re-export layer.
4. **Type hints** should be added to every new function/method signature.
5. **Docstrings** must follow Google style (Args / Returns / Raises sections).

---

## Building the Package

```bash
# Install build tooling
pip install build twine

# Build source distribution + wheel
python -m build

# Check the built artifacts
twine check dist/*
```

Artifacts land in `dist/`. The version is controlled by `pyproject.toml` → `[project] version` **and** `smartreview/__init__.py` → `__version__`. Keep them in sync. Use `bump-my-version` (alias `bumpversion`) when bumping:

```bash
bump-my-version bump patch   # or minor / major
```

---

## Installing Locally (Editable Mode)

```bash
pip install -e ".[all]"
# or, for dev extras:
pip install -e ".[all]" -r requirements_dev.txt
```

---

## Running Tests

```bash
pytest tests/ -v
```

Tests live in `tests/test_smartreview.py`. Every new function should have at least one corresponding test.

---

## Adding New Dependencies

- **Runtime:** add to `requirements.txt` (picked up automatically by `pyproject.toml` via `dynamic = ["dependencies"]`).
- **Optional/extra:** add to `[project.optional-dependencies]` in `pyproject.toml`.
- **Dev/test only:** add to `requirements_dev.txt`.

---

## Core Module: `smartreview/smartreview.py`

Key public functions (all operate on pandas DataFrames or numpy arrays):

| Function | Purpose |
|---|---|
| `calculate_cosine_similarity(query_embedding, paper_embeddings_dict)` | Returns sorted list of `(idx, score)` tuples |
| `get_top_k_papers(similarities, k=100)` | Slices the top-k results |
| `create_top_k_dataframe(top_k_papers, data, summary)` | Builds ranked DataFrame |
| `save_top_k_papers(top_k_df, output_dir, k)` | Exports to CSV + Excel |
| `create_bibtex_entry(rank, paper_data)` | Formats a single BibTeX entry |
| `generate_bibtex_file(top_k_df, output_dir, k)` | Writes `.bib` file for all top-k papers |
| `print_top_k_summary(top_k_df, k, show_rows)` | Pretty-prints a ranked summary table |
| `save_embeddings(...)` / `load_embeddings(output_dir)` | Persist/restore embeddings to/from pickle |

When **adding a new function**:
1. Implement in `smartreview/smartreview.py` (or a new sub-module).
2. Export from `smartreview/__init__.py`.
3. Write a test in `tests/test_smartreview.py`.
4. Document it in `docs/smartreview.md`.

---

## Documentation (MkDocs)

```bash
pip install mkdocs mkdocs-material
mkdocs serve        # live preview at http://127.0.0.1:8000
mkdocs build        # static site → site/
mkdocs gh-deploy    # push to GitHub Pages
```

Example notebooks under `docs/examples/` are rendered via `mkdocs-jupyter` or `nbconvert`. Keep them runnable end-to-end.

---

## Code Style & Quality

- Formatter: **black** (`black smartreview/ tests/`)
- Linter: **flake8** (`flake8 smartreview/ tests/ --max-line-length 88`)
- Type checker: **mypy** (`mypy smartreview/`)
- All tools are listed in `requirements_dev.txt`.

---

## Common Pitfalls

- The `paper_embeddings_dict` passed to `calculate_cosine_similarity` must have integer keys matching DataFrame row indices.
- `create_top_k_dataframe` expects `data` to be a raw Web of Science export DataFrame with columns such as `Article Title`, `Authors`, `Abstract`, `DOI`, `Publication Year`, `Source Title`.
- Excel export (`save_top_k_papers`) silently skips the `.xlsx` file if `openpyxl` is not installed — handle gracefully.
- `save_embeddings` / `load_embeddings` use `pickle`; embeddings must be numpy arrays.
