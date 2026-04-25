"""SmartReview: AI-powered literature-review helper.

Public API re-exports. Import everything you need straight from
``smartreview``::

    from smartreview import rank_papers, create_openai_client
"""

__author__ = "Krishnagopal Halder"
__email__ = "geonextgis@gmail.com"
__version__ = "0.0.1"

from .embeddings import (
    DEFAULT_MODEL,
    EmbeddingCache,
    MissingAPIKeyError,
    create_openai_client,
    embed_papers,
    get_embedding,
    get_embeddings_batch,
)
from .smartreview import (
    calculate_cosine_similarity,
    create_bibtex_entry,
    create_top_k_dataframe,
    generate_bibtex_file,
    get_top_k_papers,
    load_embeddings,
    load_wos_export,
    print_top_k_summary,
    rank_papers,
    save_embeddings,
    save_top_k_papers,
    top_k_by_percentile,
)

__all__ = [
    "__version__",
    # embeddings
    "DEFAULT_MODEL",
    "EmbeddingCache",
    "MissingAPIKeyError",
    "create_openai_client",
    "embed_papers",
    "get_embedding",
    "get_embeddings_batch",
    # similarity & ranking
    "calculate_cosine_similarity",
    "get_top_k_papers",
    "top_k_by_percentile",
    # I/O
    "load_wos_export",
    "create_top_k_dataframe",
    "save_top_k_papers",
    "create_bibtex_entry",
    "generate_bibtex_file",
    "print_top_k_summary",
    "save_embeddings",
    "load_embeddings",
    # high-level pipeline
    "rank_papers",
]
