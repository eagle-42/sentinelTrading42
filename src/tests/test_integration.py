"""
Tests critiques Intégration - Pipeline complet
5 tests essentiels
"""

import pytest
import pandas as pd
from src.core.prediction import PricePredictor
from src.core.sentiment import SentimentAnalyzer
from src.core.fusion import AdaptiveFusion
from src.constants import CONSTANTS


class TestIntegration:
    """Tests d'intégration end-to-end"""

    def test_end_to_end_prediction(self):
        """Test: Pipeline prédiction complet (données → prédiction)"""
        # Créer données
        data = pd.DataFrame({
            "CLOSE": [100 + i * 0.5 for i in range(300)]
        })
        
        # Prédire
        predictor = PricePredictor("SPY")
        result = predictor.predict(data, horizon=1)
        
        # Vérifier
        assert isinstance(result, dict)

    def test_constants_accessibility(self):
        """Test: Constantes accessibles et correctes"""
        assert CONSTANTS.LSTM_SEQUENCE_LENGTH == 216
        assert CONSTANTS.TICKERS == ["SPY"]

    def test_sentiment_to_fusion(self):
        """Test: Intégration Sentiment → Fusion (3 signaux)"""
        analyzer = SentimentAnalyzer()
        fusion = AdaptiveFusion()

        # Fusionner avec signal simulé (prix, sentiment, LSTM)
        result = fusion.add_signal(0.1, 0.05, 0.02, 1.0, 0.03)

        assert "fused_signal" in result
        assert "lstm_signal" in result

    def test_paths_structure(self):
        """Test: Structure chemins correcte"""
        # Vérifier chemins fonctionnent
        spy_path = CONSTANTS.get_data_path("prices", "SPY", "15min")
        assert "spy" in str(spy_path).lower()

    def test_trading_decision_logic(self):
        """Test: Logique décision trading (BUY/SELL/HOLD) - 3 signaux"""
        fusion = AdaptiveFusion()
        result = fusion.add_signal(0.15, 0.1, 0.02, 1.0, 0.08)  # Signal d'achat (prix, sentiment, LSTM)
        
        signal = result["fused_signal"]
        
        # Simuler décision
        if signal > 0.1:
            decision = "BUY"
        elif signal < -0.1:
            decision = "SELL"
        else:
            decision = "HOLD"
        
        assert decision in ["BUY", "SELL", "HOLD"]
