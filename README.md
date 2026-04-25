# SmartReview

[![PyPI version](https://img.shields.io/pypi/v/smartreview.svg)](https://pypi.python.org/pypi/smartreview)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**SmartReview** is an AI-powered literature review tool that uses OpenAI text embeddings to
rank a large corpus of research papers by how closely they match a free-text description of
your research interests.

---

## Features

- 🔍 **Semantic ranking** – embed every paper (title + abstract) and your interest statement,
  then rank by cosine similarity.
- 📊 **Flexible top-K selection** – choose a fixed K or derive it automatically (e.g. top 20 %
  by similarity score).
- 💾 **Multiple export formats** – CSV, Excel (`.xlsx`), and BibTeX (`.bib`).
- 🗄️ **Embedding cache** – save / reload embeddings with pickle so you don't re-call the API
  on every run.
- 🔑 **Safe API-key handling** – reads `OPENAI_API_KEY` from the environment (or a `.env`
  file) and raises a clear error if it is missing.

---

## Installation

```bash
pip install smartreview
```

For development / editable installs:

```bash
git clone https://github.com/geonextgis/smartreview.git
cd smartreview
pip install -e .
```

---

## Quick Start

### 1 – Set your OpenAI API key

```bash
# Option A: environment variable
export OPENAI_API_KEY="sk-..."

# Option B: .env file (recommended)
echo 'OPENAI_API_KEY=sk-...' > .env
```

### 2 – Generate embeddings and find top papers

```python
from dotenv import load_dotenv
import pandas as pd
from smartreview import (
    create_openai_client, get_embedding,
    calculate_cosine_similarity, get_top_k_papers,
    create_top_k_dataframe, save_top_k_papers,
    generate_bibtex_file, save_embeddings, load_embeddings,
)

load_dotenv()  # reads OPENAI_API_KEY from .env

# 1. Load your Web of Science export
data = pd.read_excel("data/papers.xls")
summary = {i: (row["Article Title"], row["Abstract"]) for i, row in data.iterrows()}

# 2. Create OpenAI client
client = create_openai_client()  # raises ValueError if key is missing

# 3. Embed all papers
paper_embeddings = {}
for idx, (title, abstract) in summary.items():
    text = title + " " + (str(abstract) if pd.notna(abstract) else "")
    paper_embeddings[idx] = get_embedding(text, client=client)

# 4. Embed your research interest
interest_text = "Machine learning for crop yield prediction using remote sensing data."
interest_embedding = get_embedding(interest_text, client=client)

# 5. Save embeddings (avoids re-calling the API next time)
save_embeddings(paper_embeddings, interest_embedding, interest_text)

# 6. Rank papers
similarities = calculate_cosine_similarity(interest_embedding, paper_embeddings)
top_k = get_top_k_papers(similarities, k=100)

# 7. Export
df = create_top_k_dataframe(top_k, data, summary)
save_top_k_papers(df, output_dir="data", k=100)
generate_bibtex_file(df, output_dir="data", k=100)
print("Done! Check the data/ folder for your results.")
```

### 3 – Re-use cached embeddings

```python
from dotenv import load_dotenv
from smartreview import load_embeddings, calculate_cosine_similarity, get_top_k_papers

load_dotenv()
paper_embeddings, interest_embedding, interest_text = load_embeddings()
similarities = calculate_cosine_similarity(interest_embedding, paper_embeddings)
top_k = get_top_k_papers(similarities, k=50)
```

---

## API Reference

### OpenAI helpers (`smartreview.embeddings`)

| Function | Description |
|---|---|
| `create_openai_client(api_key=None)` | Return an `openai.OpenAI` client; reads `OPENAI_API_KEY` from env if `api_key` is omitted. |
| `get_embedding(text, client=None, model="text-embedding-3-large")` | Embed a single string and return a NumPy array. |
| `get_embeddings_batch(texts, client=None, ...)` | Embed a list of strings with optional progress logging. |

### Similarity (`smartreview.smartreview`)

| Function | Description |
|---|---|
| `calculate_cosine_similarity(query_emb, paper_emb_dict)` | Return a list of `(idx, score)` tuples sorted by descending similarity. |
| `get_top_k_papers(similarities, k=100)` | Slice the top-K entries from a similarity list. |

### DataFrame & Export

| Function | Description |
|---|---|
| `create_top_k_dataframe(top_k, data, summary)` | Build a ranked `pd.DataFrame` from top-K results. |
| `save_top_k_papers(df, output_dir, k)` | Write CSV + Excel files; returns a dict of file paths. |
| `print_top_k_summary(df, k, show_rows)` | Pretty-print a summary table. |
| `generate_bibtex_file(df, output_dir, k)` | Write a `.bib` file; returns a dict with path and entry count. |

### Embedding Persistence

| Function | Description |
|---|---|
| `save_embeddings(paper_emb, interest_emb, interest_text, output_dir)` | Pickle embeddings to `output_dir`. |
| `load_embeddings(output_dir)` | Load and return `(paper_emb, interest_emb, interest_text)`. |

---

## Example Notebook

An end-to-end walkthrough is provided in
[`docs/examples/example.ipynb`](docs/examples/example.ipynb).  
Place your Web of Science `.xls` export in `docs/examples/data/` before running.

---

## Requirements

| Package | Purpose |
|---|---|
| `openai` | Text embeddings via the OpenAI API |
| `numpy` | Numerical arrays |
| `pandas` | DataFrame I/O |
| `scikit-learn` | Cosine similarity |
| `tiktoken` | Token counting |
| `openpyxl` | Excel export |
| `python-dotenv` | `.env` file support |

---

## License

[MIT](LICENSE) © Krishnagopal Halder
