"""Shared pytest fixtures.

These tests never call the real OpenAI API — every fixture or test that
needs an embedding either uses a deterministic stub or a ``MagicMock`` that
mimics ``openai.OpenAI``.
"""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


EMBED_DIM = 16


def _deterministic_embedding(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """Produce a deterministic, normalised vector from a text input.

    Identical text → identical vector. Different text → different vector.
    Useful for similarity tests without needing a real model.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / (np.linalg.norm(vec) or 1.0)


@pytest.fixture
def mock_openai_client():
    """A ``MagicMock`` that mimics ``openai.OpenAI`` for embeddings."""
    client = MagicMock()

    def _create(input, model):  # noqa: A002 - mirrors OpenAI signature
        if isinstance(input, str):
            input = [input]
        data = []
        for text in input:
            emb = _deterministic_embedding(text).tolist()
            item = MagicMock()
            item.embedding = emb
            data.append(item)
        response = MagicMock()
        response.data = data
        return response

    client.embeddings.create.side_effect = _create
    return client


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Article Title": "Deep learning for crop yield",
                "Abstract": "We use CNNs to predict maize yield from Sentinel-2.",
                "Authors": "Smith, A; Doe, J",
                "Publication Year": 2024,
                "Source Title": "Remote Sens.",
                "Volume": 16,
                "Issue": 3,
                "Article Number": "1234",
                "DOI": "10.0/abc",
            },
            {
                "Article Title": "Quantum chromodynamics review",
                "Abstract": "A review of QCD calculations on the lattice.",
                "Authors": "Lee, B",
                "Publication Year": 2023,
                "Source Title": "Phys. Rev. D",
                "Volume": 108,
                "Issue": 1,
                "Article Number": "111",
                "DOI": "10.0/qcd",
            },
            {
                "Article Title": "Random forests for soil moisture",
                "Abstract": "Random-forest regression of SMAP soil moisture.",
                "Authors": "Patel, R; Ng, S",
                "Publication Year": 2025,
                "Source Title": "Hydrol. Process.",
                "Volume": 39,
                "Issue": 2,
                "Article Number": "5678",
                "DOI": "10.0/sm",
            },
        ]
    )


@pytest.fixture
def sample_summary(sample_dataframe):
    return {
        i: (row["Article Title"], row["Abstract"])
        for i, row in sample_dataframe.iterrows()
    }


@pytest.fixture
def deterministic_embedding():
    return _deterministic_embedding
