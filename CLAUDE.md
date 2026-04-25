# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Purpose

`smartreview` is an AI-powered literature-review helper. Given a corpus of research papers
(exported from Web of Science as `.xls` or `.xlsx`) and a free-text description of the user's research interests, it:

1. Generates OpenAI text embeddings for every paper (title + abstract).
2. Generates an embedding for the user's research-interest statement.
3. Ranks all papers by cosine similarity to the interest embedding.
4. Exports the top-K papers as CSV, Excel, and BibTeX.

---

## Current Repository Layout

```
smartreview/                   ← installable Python package
    __init__.py                ← public API; re-exports everything users need
    smartreview.py             ← core utilities (similarity, I/O, BibTeX)
docs/
    examples/
        example.ipynb          ← end-to-end walkthrough notebook
        test_data.xls          ← test data
requirements.txt               ← runtime dependencies
requirements_dev.txt           ← dev / test dependencies
pyproject.toml                 ← PEP 517 build config (setuptools)
CLAUDE.md                      ← this file
README.md                      ← project overview for users / PyPI
```

---

## TODO

1. Ensure the package is intuitive and user-friendly, with a smooth onboarding experience.
2. Implement secure and efficient handling of the user’s OpenAI API key.
3. Add caching for embeddings: save them after the first run and reuse existing embeddings from the output directory to avoid redundant computation.
4. Enhance the documentation, including improving examples and refining the `examples/example.ipynb` notebook.
5. Develop a comprehensive `README.md` that clearly explains the package overview, features, setup instructions, and usage details.

---

## Coding Conventions

- **Python ≥ 3.8** – no walrus operator, no `match` statements in public API.
- **Type hints** – use them on all public functions.
- **Docstrings** – Google style.
- **No global state** – functions receive clients/configs as arguments with sensible defaults.
- **Dependencies** – keep `requirements.txt` up to date; do not import optional packages at
  the module level without a try/except guard.
- **Tests** – live in `tests/` (to be created). Use `pytest`. Do not test against the real
  OpenAI API; mock `openai.OpenAI`.

---

## Running the Example Notebook

```bash
cd docs/examples
jupyter lab example.ipynb
```

Place your Web of Science `.xls` export in `docs/examples/data/` before running.

---

## Versioning

Version is kept in `pyproject.toml` (`version = "…"`) and mirrored in
`smartreview/__init__.py` (`__version__`). Use `bump2version` to update both at once:

```bash
bump2version patch   # 0.0.1 → 0.0.2
```

---

## Common Tasks

| Task | Command |
|---|---|
| Install in editable mode | `pip install -e .` |
| Run tests | `pytest` |
| Build wheel | `python -m build` |
| Lint | `ruff check smartreview/` |
| Format | `ruff format smartreview/` |
