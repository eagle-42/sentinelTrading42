"""
Tests critiques Sentiment - Analyse FinBERT
3 tests essentiels
"""

import pytest
from src.core.sentiment import SentimentAnalyzer


class TestSentiment:
    """Tests analyse sentiment"""

    @pytest.fixture
    def analyzer(self):
        """Fixture analyseur"""
        return SentimentAnalyzer()

    def test_sentiment_analyzer_init(self, analyzer):
        """Test: Analyseur initialisé correctement"""
        assert analyzer is not None
        assert hasattr(analyzer, 'finbert')

    def test_sentiment_window(self, analyzer):
        """Test: Fenêtre d'agrégation configurée"""
        assert hasattr(analyzer, 'window_minutes')
        assert analyzer.window_minutes > 0

    def test_window_aggregation(self, analyzer):
        """Test: Fenêtre par défaut = 12min"""
        assert analyzer.window_minutes == 12
