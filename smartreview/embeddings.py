"""OpenAI embedding helpers.

Thin wrapper around ``openai.OpenAI`` with safe API-key handling, retries,
and an on-disk cache so repeated runs do not re-call the API.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

DEFAULT_MODEL = "text-embedding-3-large"
DEFAULT_CACHE_DIR = "data/embeddings/cache"


class MissingAPIKeyError(RuntimeError):
    """Raised when no OpenAI API key can be found."""


def create_openai_client(api_key: Optional[str] = None):
    """Create an authenticated ``openai.OpenAI`` client.

    The key is read from the ``api_key`` argument if provided, otherwise from
    the ``OPENAI_API_KEY`` environment variable. A clear, actionable error is
    raised if neither is set.

    Args:
        api_key: Explicit OpenAI API key. If ``None``, falls back to the
            ``OPENAI_API_KEY`` environment variable.

    Returns:
        An ``openai.OpenAI`` client instance.

    Raises:
        MissingAPIKeyError: When no API key is available.
        ImportError: When the ``openai`` package is not installed.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "The 'openai' package is required. Install it with `pip install openai`."
        ) from exc

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise MissingAPIKeyError(
            "No OpenAI API key found. Set the OPENAI_API_KEY environment "
            "variable, place it in a .env file, or pass api_key=... explicitly."
        )
    return OpenAI(api_key=key)


def _hash_text(text: str, model: str) -> str:
    """Return a stable SHA-256 hex digest for a (model, text) pair."""
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\0")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


class EmbeddingCache:
    """Tiny content-addressed cache for embeddings on the local filesystem.

    Each ``(model, text)`` pair is stored as a single ``.npy`` file named by
    the SHA-256 of its contents, so repeated runs across different corpora
    transparently reuse already-paid-for embeddings.
    """

    def __init__(self, cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, text: str, model: str) -> Path:
        return self.cache_dir / f"{_hash_text(text, model)}.npy"

    def get(self, text: str, model: str) -> Optional[np.ndarray]:
        path = self._path_for(text, model)
        if path.exists():
            return np.load(path)
        return None

    def put(self, text: str, model: str, embedding: np.ndarray) -> None:
        np.save(self._path_for(text, model), embedding)


def get_embedding(
    text: str,
    client=None,
    model: str = DEFAULT_MODEL,
    cache: Optional[EmbeddingCache] = None,
    max_retries: int = 3,
) -> np.ndarray:
    """Embed a single string and return a NumPy array.

    Args:
        text: The text to embed. Newlines are replaced with spaces, matching
            OpenAI's recommendation for embedding inputs.
        client: An ``openai.OpenAI`` client. Created lazily if omitted.
        model: Embedding model name.
        cache: Optional :class:`EmbeddingCache`. When provided, results are
            read/written transparently.
        max_retries: Number of retries on transient API errors with
            exponential backoff.

    Returns:
        A 1-D ``numpy.ndarray`` of shape ``(embedding_dim,)``.
    """
    text = text.replace("\n", " ").strip()
    if not text:
        raise ValueError("Cannot embed an empty string.")

    if cache is not None:
        cached = cache.get(text, model)
        if cached is not None:
            return cached

    if client is None:
        client = create_openai_client()

    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(input=[text], model=model)
            embedding = np.asarray(response.data[0].embedding, dtype=np.float32)
            if cache is not None:
                cache.put(text, model, embedding)
            return embedding
        except Exception as exc:  # noqa: BLE001 - retry-any-error is intentional
            last_exc = exc
            if attempt == max_retries - 1:
                break
            time.sleep(2 ** attempt)
    raise RuntimeError(
        f"Failed to fetch embedding after {max_retries} attempts: {last_exc}"
    ) from last_exc


def get_embeddings_batch(
    texts: Sequence[str],
    client=None,
    model: str = DEFAULT_MODEL,
    cache: Optional[EmbeddingCache] = None,
    show_progress: bool = True,
    progress_every: int = 50,
) -> List[np.ndarray]:
    """Embed a sequence of strings, reusing cached values where possible.

    Args:
        texts: Iterable of strings to embed.
        client: OpenAI client. Created lazily if omitted (only when at least
            one text is not in the cache).
        model: Embedding model name.
        cache: Optional :class:`EmbeddingCache` for persistence.
        show_progress: Print a progress line every ``progress_every`` items.
        progress_every: How often to print progress (in items).

    Returns:
        A list of 1-D ``numpy.ndarray`` embeddings in the same order as
        ``texts``.
    """
    embeddings: List[np.ndarray] = []
    n = len(texts)
    api_calls = 0
    cache_hits = 0
    for idx, text in enumerate(texts):
        if cache is not None:
            cached = cache.get(text.replace("\n", " ").strip(), model)
            if cached is not None:
                embeddings.append(cached)
                cache_hits += 1
                if show_progress and (idx + 1) % progress_every == 0:
                    print(
                        f"  [{idx + 1}/{n}] processed "
                        f"({cache_hits} cached, {api_calls} api)"
                    )
                continue
        if client is None:
            client = create_openai_client()
        embeddings.append(
            get_embedding(text, client=client, model=model, cache=cache)
        )
        api_calls += 1
        if show_progress and (idx + 1) % progress_every == 0:
            print(
                f"  [{idx + 1}/{n}] processed "
                f"({cache_hits} cached, {api_calls} api)"
            )
    if show_progress:
        print(
            f"  done: {n} items "
            f"({cache_hits} cached, {api_calls} new API calls)"
        )
    return embeddings


def embed_papers(
    summary: Dict[int, Iterable],
    client=None,
    model: str = DEFAULT_MODEL,
    cache: Optional[EmbeddingCache] = None,
    show_progress: bool = True,
) -> Dict[int, np.ndarray]:
    """Embed a ``{idx: (title, abstract)}`` mapping into ``{idx: embedding}``.

    Missing or non-string abstracts are silently coerced to an empty string.

    Args:
        summary: Mapping of paper index to ``(title, abstract)``.
        client: OpenAI client; created lazily if omitted.
        model: Embedding model name.
        cache: Optional embedding cache.
        show_progress: Whether to print progress.

    Returns:
        Dictionary ``{paper_idx: embedding}``.
    """
    indices = list(summary.keys())
    texts = []
    for idx in indices:
        title, abstract = summary[idx]
        title = "" if title is None else str(title)
        abstract = "" if abstract is None or abstract != abstract else str(abstract)
        texts.append((title + " " + abstract).strip())
    embeddings = get_embeddings_batch(
        texts,
        client=client,
        model=model,
        cache=cache,
        show_progress=show_progress,
    )
    return dict(zip(indices, embeddings))
