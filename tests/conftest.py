from __future__ import annotations

from pathlib import Path

import pytest

from bidding_qa.config import PROJECT_ROOT, Settings
from bidding_qa.corpus import load_corpus
from bidding_qa.index_store import HybridIndex

DATA_DIR = PROJECT_ROOT / "data"


@pytest.fixture(scope="session")
def corpus():
    return load_corpus(DATA_DIR)


@pytest.fixture(scope="session")
def index(corpus):
    return HybridIndex(corpus)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        api_key="",
        base_url="https://api.moonshot.cn/v1",
        model="kimi-k3.1-pro-highspeed",
        reasoning_effort="low",
        max_completion_tokens=1024,
        embedding_backend="local",
        data_dir=DATA_DIR,
        index_dir=tmp_path / "index",
        top_k=5,
        history_turns=6,
    )
