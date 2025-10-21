"""
Tests critiques Fusion - Fusion adaptative des signaux
3 tests essentiels
"""

import pytest
from src.core.fusion import AdaptiveFusion


class TestFusion:
    """Tests fusion adaptative"""

    @pytest.fixture
    def fusion(self):
        """Fixture fusion"""
        return AdaptiveFusion()

    def test_add_signal(self, fusion):
        """Test: Ajout de signaux et fusion (3 signaux: prix, sentiment, LSTM)"""
        result = fusion.add_signal(
            price_signal=0.5,
            sentiment_signal=-0.3,
            lstm_signal=0.2,
            price_volatility=0.02,
            volume_ratio=1.0
        )
        assert "fused_signal" in result
        assert "lstm_signal" in result
        assert -1 <= result["fused_signal"] <= 1

    def test_fusion_weights(self, fusion):
        """Test: Poids de fusion (somme = 1) - 3 signaux"""
        fusion.add_signal(0.5, 0.2, 0.02, 1.0, 0.1)
        weights = fusion.current_weights
        assert "price" in weights
        assert "sentiment" in weights
        assert "lstm" in weights
        assert abs(sum(weights.values()) - 1.0) < 0.01  # Somme = 1

    def test_fusion_initialization(self, fusion):
        """Test: Fusion initialisée avec poids corrects (3 signaux)"""
        assert fusion is not None
        assert hasattr(fusion, 'current_weights')
        assert "price" in fusion.current_weights
        assert "sentiment" in fusion.current_weights
        assert "lstm" in fusion.current_weights
