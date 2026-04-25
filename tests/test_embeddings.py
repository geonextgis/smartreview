"""Tests for ``smartreview.embeddings``."""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest

from smartreview.embeddings import (
    DEFAULT_MODEL,
    EmbeddingCache,
    MissingAPIKeyError,
    create_openai_client,
    embed_papers,
    get_embedding,
    get_embeddings_batch,
)


def test_create_openai_client_raises_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        create_openai_client()


def test_create_openai_client_uses_explicit_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Should not raise — API construction is lazy
    client = create_openai_client(api_key="sk-test-key")
    assert client is not None


def test_get_embedding_returns_numpy_array(mock_openai_client):
    emb = get_embedding("hello world", client=mock_openai_client)
    assert isinstance(emb, np.ndarray)
    assert emb.ndim == 1
    assert emb.dtype == np.float32


def test_get_embedding_rejects_empty_text(mock_openai_client):
    with pytest.raises(ValueError):
        get_embedding("   ", client=mock_openai_client)


def test_get_embedding_caches_results(tmp_path, mock_openai_client):
    cache = EmbeddingCache(tmp_path / "cache")
    emb1 = get_embedding("same text", client=mock_openai_client, cache=cache)
    emb2 = get_embedding("same text", client=mock_openai_client, cache=cache)
    np.testing.assert_array_equal(emb1, emb2)
    # Only one API call: second hit served from cache
    assert mock_openai_client.embeddings.create.call_count == 1


def test_get_embeddings_batch_preserves_order(mock_openai_client):
    texts = ["alpha", "beta", "gamma"]
    embs = get_embeddings_batch(
        texts, client=mock_openai_client, show_progress=False
    )
    assert len(embs) == 3
    # Each input text must map to a distinct embedding
    norms = [np.linalg.norm(e) for e in embs]
    assert all(n > 0 for n in norms)


def test_get_embeddings_batch_uses_cache(tmp_path, mock_openai_client):
    cache = EmbeddingCache(tmp_path / "cache")
    texts = ["a", "b", "c"]
    get_embeddings_batch(texts, client=mock_openai_client, cache=cache, show_progress=False)
    first_call_count = mock_openai_client.embeddings.create.call_count
    assert first_call_count == 3

    # Second pass — all three should be cached
    get_embeddings_batch(texts, client=mock_openai_client, cache=cache, show_progress=False)
    assert mock_openai_client.embeddings.create.call_count == first_call_count


def test_embed_papers_handles_nan_abstract(mock_openai_client):
    summary = {
        0: ("title one", "abstract one"),
        1: ("title two", float("nan")),  # NaN abstract
        2: ("title three", None),
    }
    embeddings = embed_papers(
        summary, client=mock_openai_client, show_progress=False
    )
    assert set(embeddings.keys()) == {0, 1, 2}
    for emb in embeddings.values():
        assert isinstance(emb, np.ndarray)


def test_embedding_cache_hash_distinguishes_models(tmp_path, mock_openai_client):
    cache = EmbeddingCache(tmp_path / "cache")
    a = get_embedding("hello", client=mock_openai_client, cache=cache, model="m1")
    b = get_embedding("hello", client=mock_openai_client, cache=cache, model="m2")
    # Different models hash differently → both served from API
    assert mock_openai_client.embeddings.create.call_count == 2
    # And both retrieved without further API calls on a second pass
    a2 = get_embedding("hello", client=mock_openai_client, cache=cache, model="m1")
    b2 = get_embedding("hello", client=mock_openai_client, cache=cache, model="m2")
    assert mock_openai_client.embeddings.create.call_count == 2
    np.testing.assert_array_equal(a, a2)
    np.testing.assert_array_equal(b, b2)
