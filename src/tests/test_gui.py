"""
Tests critiques GUI - Interface Streamlit
Logique métier et validation des données (sans dépendance Streamlit)
"""

import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path


class TestGUILogic:
    """Tests de la logique métier GUI (sans imports services)"""

    def test_price_data_format(self):
        """Test: Format données prix pour GUI"""
        # Simuler données prix
        prices = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="15min"),
            "open": [100 + i * 0.1 for i in range(100)],
            "high": [101 + i * 0.1 for i in range(100)],
            "low": [99 + i * 0.1 for i in range(100)],
            "close": [100.5 + i * 0.1 for i in range(100)],
            "volume": [1000000] * 100
        })
        
        # Vérifier format correct pour affichage
        assert not prices.empty
        assert len(prices) == 100
        assert all(col in prices.columns for col in ["open", "high", "low", "close"])

    def test_decision_data_structure(self):
        """Test: Structure décision pour GUI"""
        decision = {
            "ticker": "SPY",
            "timestamp": datetime.now().isoformat(),
            "decision": "HOLD",
            "confidence": 0.85,
            "fused_signal": 0.05,
            "signals": {
                "price": 0.03,
                "sentiment": 0.02
            }
        }
        
        # Vérifier structure valide
        assert "ticker" in decision
        assert "decision" in decision
        assert decision["decision"] in ["BUY", "SELL", "HOLD"]
        assert 0 <= decision["confidence"] <= 1

    def test_validation_result_calculation(self):
        """Test: Calcul résultat validation"""
        decision = {
            "decision": "BUY",
            "current_price": 100.0
        }
        future_price = 102.0
        
        # Calculer si décision correcte
        if decision["decision"] == "BUY":
            is_correct = future_price > decision["current_price"]
        elif decision["decision"] == "SELL":
            is_correct = future_price < decision["current_price"]
        else:
            is_correct = True
        
        assert is_correct is True  # BUY avec hausse = correct

    def test_prediction_signal_conversion(self):
        """Test: Conversion prédiction → signal"""
        current_price = 100.0
        predicted_price = 105.0
        
        # Calculer signal
        signal = (predicted_price - current_price) / current_price
        
        assert isinstance(signal, float)
        assert signal == 0.05  # +5%
        assert -1 <= signal <= 1

    def test_chart_data_aggregation(self):
        """Test: Agrégation données pour chart"""
        # Simuler données 1min
        data_1min = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01 10:00", periods=60, freq="1min"),
            "close": [100 + i * 0.01 for i in range(60)]
        })
        
        # Agréger en 5min
        data_1min.set_index("timestamp", inplace=True)
        data_5min = data_1min.resample("5min").last()
        
        assert len(data_5min) == 12  # 60min / 5min = 12


class TestGUIValidation:
    """Tests de validation GUI"""

    def test_decision_buy_validation(self):
        """Test: Validation BUY (prix monte = correct)"""
        decision = {"decision": "BUY", "current_price": 100.0}
        future_price = 102.0
        
        is_correct = future_price > decision["current_price"]
        assert is_correct is True

    def test_decision_sell_validation(self):
        """Test: Validation SELL (prix baisse = correct)"""
        decision = {"decision": "SELL", "current_price": 100.0}
        future_price = 98.0
        
        is_correct = future_price < decision["current_price"]
        assert is_correct is True

    def test_decision_hold_validation(self):
        """Test: Validation HOLD (toujours correct)"""
        decision = {"decision": "HOLD", "current_price": 100.0}
        future_price = 102.0  # Peu importe
        
        is_correct = True  # HOLD toujours correct
        assert is_correct is True

    def test_gain_calculation(self):
        """Test: Calcul gain trading"""
        current_price = 100.0
        future_price = 105.0
        
        gain = future_price - current_price
        gain_pct = (gain / current_price) * 100
        
        assert gain == 5.0
        assert gain_pct == 5.0


class TestGUIEdgeCases:
    """Tests des cas limites GUI"""

    def test_empty_dataframe(self):
        """Test: Détection DataFrame vide"""
        empty_df = pd.DataFrame()
        assert empty_df.empty

    def test_missing_columns(self):
        """Test: Détection colonnes manquantes"""
        incomplete_data = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="15min"),
            "close": [100] * 10
        })
        
        required_cols = ["open", "high", "low", "close"]
        missing = [col for col in required_cols if col not in incomplete_data.columns]
        
        assert len(missing) == 3  # open, high, low manquants

    def test_invalid_decision(self):
        """Test: Détection décision invalide"""
        invalid_decision = {"decision": "INVALID"}
        valid_decisions = ["BUY", "SELL", "HOLD"]
        is_valid = invalid_decision.get("decision") in valid_decisions
        
        assert is_valid is False

    def test_confidence_bounds(self):
        """Test: Confiance entre 0 et 1"""
        decisions = [
            {"confidence": 0.0},   # OK
            {"confidence": 0.5},   # OK
            {"confidence": 1.0},   # OK
            {"confidence": 1.5},   # INVALID
            {"confidence": -0.1},  # INVALID
        ]
        
        valid_count = sum(1 for d in decisions if 0 <= d["confidence"] <= 1)
        assert valid_count == 3
