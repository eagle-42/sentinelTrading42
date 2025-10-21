"""
🧪 Configuration des tests Sentinel42
Fixtures et configuration commune pour tous les tests
"""

import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator

import numpy as np
import pandas as pd
import pytest

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent.parent))
from constants import CONSTANTS

# Configuration des variables d'environnement pour les tests
os.environ.setdefault("FINBERT_MODE", "stub")
os.environ.setdefault("NEWSAPI_ENABLED", "false")
os.environ.setdefault("POLYGON_API_KEY", "test_key")
os.environ.setdefault("NEWSAPI_KEY", "test_key")
os.environ.setdefault("TICKERS", "SPY:S&P 500 ETF,NVDA:NVIDIA Corporation")
os.environ.setdefault("NEWS_FEEDS", "https://example.com/feed1.rss,https://example.com/feed2.rss")
os.environ.setdefault("PRICE_INTERVAL", "1min")
os.environ.setdefault("PRICE_PERIOD", "1d")
os.environ.setdefault("FUSION_MODE", "adaptive")


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Répertoire temporaire pour les données de test"""
    return Path(tempfile.mkdtemp(prefix="sentinel42_test_"))


@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """Répertoire temporaire pour chaque test"""
    temp_path = Path(tempfile.mkdtemp(prefix="sentinel42_test_"))
    yield temp_path
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture(scope="function")
def sample_price_data() -> pd.DataFrame:
    """Données de prix de test"""
    dates = pd.date_range("2024-01-01", periods=100, freq="1min")
    return pd.DataFrame(
        {
            "ts_utc": dates,
            "open": np.random.randn(100) * 100 + 100,
            "high": np.random.randn(100) * 100 + 105,
            "low": np.random.randn(100) * 100 + 95,
            "close": np.random.randn(100) * 100 + 100,
            "volume": np.random.randint(1000, 10000, 100),
            "ticker": "SPY",
        }
    )


@pytest.fixture(scope="function")
def sample_news_data() -> pd.DataFrame:
    """Données de news de test"""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1H"),
            "title": [f"News {i}" for i in range(10)],
            "content": [f"Content {i}" for i in range(10)],
            "source": ["Reuters", "Bloomberg", "CNBC"] * 3 + ["Reuters"],
            "ticker": ["SPY", "NVDA"] * 5,
        }
    )


@pytest.fixture(scope="function")
def sample_sentiment_data() -> pd.DataFrame:
    """Données de sentiment de test"""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="1H"),
            "sentiment": np.random.randn(20),
            "confidence": np.random.rand(20),
            "ticker": "SPY",
        }
    )


@pytest.fixture(scope="function")
def sample_features_data() -> pd.DataFrame:
    """Données de features pour les tests LSTM"""
    from src.constants import CONSTANTS

    data = {}
    for feature in CONSTANTS.get_feature_columns():
        data[feature] = np.random.randn(100)

    data["DATE"] = pd.date_range("2024-01-01", periods=100, freq="1min")
    return pd.DataFrame(data)


@pytest.fixture(scope="function")
def mock_yahoo_data() -> pd.DataFrame:
    """Données mockées de Yahoo Finance"""
    dates = pd.date_range("2024-01-01", periods=50, freq="1min")
    return pd.DataFrame(
        {
            "Open": np.random.randn(50) * 100 + 100,
            "High": np.random.randn(50) * 100 + 105,
            "Low": np.random.randn(50) * 100 + 95,
            "Close": np.random.randn(50) * 100 + 100,
            "Volume": np.random.randint(1000, 10000, 50),
        },
        index=dates,
    )


@pytest.fixture(scope="function")
def mock_news_items() -> list:
    """Articles de news mockés"""
    return [
        {
            "title": "NVIDIA stock rises on strong earnings",
            "summary": "NVIDIA reported better than expected earnings",
            "content": "Full article content about NVIDIA earnings...",
            "link": "https://example.com/nvidia-news-1",
            "published": "2024-01-01T10:00:00Z",
            "source": "Reuters",
        },
        {
            "title": "S&P 500 shows strong performance",
            "summary": "The S&P 500 index continues to rise",
            "content": "Full article content about S&P 500...",
            "link": "https://example.com/spy-news-1",
            "published": "2024-01-01T11:00:00Z",
            "source": "Bloomberg",
        },
    ]


@pytest.fixture(scope="function")
def mock_polygon_response() -> dict:
    """Réponse mockée de l'API Polygon"""
    return {
        "status": "OK",
        "results": [
            {
                "t": 1704067200000,  # timestamp
                "o": 100.0,  # open
                "h": 102.0,  # high
                "l": 99.0,  # low
                "c": 101.0,  # close
                "v": 1000,  # volume
            },
            {"t": 1704067260000, "o": 101.0, "h": 103.0, "l": 100.0, "c": 102.0, "v": 1200},
        ],
    }


@pytest.fixture(scope="function")
def mock_newsapi_response() -> dict:
    """Réponse mockée de NewsAPI"""
    return {
        "status": "ok",
        "articles": [
            {
                "title": "NVIDIA earnings beat expectations",
                "description": "Strong AI demand drives growth",
                "content": "Full article content...",
                "url": "https://example.com/news1",
                "publishedAt": "2024-01-01T10:00:00Z",
                "source": {"name": "Reuters"},
            }
        ],
    }


@pytest.fixture(scope="function")
def mock_rss_feed():
    """Feed RSS mocké"""

    class MockEntry:
        def __init__(self, title, summary, link, published):
            self.title = title
            self.summary = summary
            self.link = link
            self.published = published

    class MockFeed:
        def __init__(self):
            self.bozo = False
            self.entries = [
                MockEntry(
                    "NVIDIA stock rises", "Strong earnings report", "https://example.com/news1", "2024-01-01T10:00:00Z"
                ),
                MockEntry(
                    "Market analysis", "S&P 500 shows growth", "https://example.com/news2", "2024-01-01T11:00:00Z"
                ),
            ]

    return MockFeed()


@pytest.fixture(scope="function")
def mock_trading_config() -> Dict[str, Any]:
    """Configuration de trading mockée"""
    return {"buy_threshold": 0.3, "sell_threshold": -0.3, "hold_confidence": 0.3, "success_threshold": 0.02}


@pytest.fixture(scope="function")
def mock_lstm_config() -> Dict[str, Any]:
    """Configuration LSTM mockée (utilise CONSTANTS)"""
    return {
        "sequence_length": CONSTANTS.LSTM_SEQUENCE_LENGTH,
        "top_features": CONSTANTS.LSTM_TOP_FEATURES,
        "prediction_horizon": CONSTANTS.LSTM_PREDICTION_HORIZON,
        "hidden_sizes": CONSTANTS.LSTM_HIDDEN_SIZES,
        "dropout_rate": CONSTANTS.LSTM_DROPOUT_RATE,
        "epochs": CONSTANTS.LSTM_EPOCHS,
        "batch_size": CONSTANTS.LSTM_BATCH_SIZE,
        "patience": CONSTANTS.LSTM_PATIENCE,
        "learning_rate": CONSTANTS.LSTM_LEARNING_RATE,
    }


@pytest.fixture(scope="function")
def mock_fusion_config() -> Dict[str, Any]:
    """Configuration de fusion mockée"""
    return {
        "mode": "adaptive",
        "base_price_weight": 0.6,
        "base_sentiment_weight": 0.4,
        "max_weight_change": 0.1,
        "regularization_factor": 0.1,
    }


# Marqueurs personnalisés pour les tests
pytestmark = [
    pytest.mark.unit,
]
