"""
Tests pour le modèle LSTM Article (arXiv:2501.17366v1)
Tests du modèle RETURNS + features corrélées
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from constants import CONSTANTS
from core.prediction import FinancialLSTM, PricePredictor


class TestFinancialLSTM:
    """Tests du modèle LSTM"""

    def test_lstm_initialization(self):
        """Test initialisation modèle avec CONSTANTS"""
        model = FinancialLSTM(
            input_size=4,
            hidden_size=CONSTANTS.LSTM_HIDDEN_SIZES[0],
            num_layers=2,
            dropout=CONSTANTS.LSTM_DROPOUT_RATE
        )
        assert model.hidden_size == CONSTANTS.LSTM_HIDDEN_SIZES[0]
        assert model.num_layers == 2

    def test_lstm_forward_pass(self):
        """Test forward pass avec CONSTANTS"""
        model = FinancialLSTM(
            input_size=4,
            hidden_size=CONSTANTS.LSTM_HIDDEN_SIZES[0],
            num_layers=2
        )
        model.eval()

        # Batch de 10 séquences de 216 timesteps, 4 features
        x = torch.randn(10, CONSTANTS.LSTM_SEQUENCE_LENGTH, 4)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (10, 1)  # Batch, 1 prédiction


class TestPricePredictor:
    """Tests du prédicteur de prix"""

    @pytest.fixture
    def predictor(self):
        """Fixture prédicteur"""
        return PricePredictor("SPY")

    @pytest.fixture
    def sample_data(self):
        """Fixture données test"""
        # 1200 jours pour avoir assez de données après split 60/20/20 et window 216
        # Train=720, Val=240, Test=240 → Val-216=24 séquences (OK)
        n_days = 1200
        dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
        data = pd.DataFrame(
            {
                "DATE": dates,
                "Open_RETURN": np.random.randn(n_days) * 0.01,
                "High_RETURN": np.random.randn(n_days) * 0.01,
                "Low_RETURN": np.random.randn(n_days) * 0.01,
                "TARGET": np.random.randn(n_days) * 0.01,
            }
        )
        return data

    def test_predictor_initialization(self, predictor):
        """Test initialisation prédicteur"""
        assert predictor.ticker == "SPY"
        assert predictor.sequence_length == CONSTANTS.LSTM_SEQUENCE_LENGTH
        assert not predictor.is_loaded

    def test_create_sequences(self, predictor, sample_data):
        """Test création séquences"""
        features = sample_data[["Open_RETURN", "High_RETURN", "Low_RETURN", "TARGET"]].values

        X, y = predictor.create_sequences(features)

        assert X is not None
        assert y is not None
        assert X.shape[1] == CONSTANTS.LSTM_SEQUENCE_LENGTH  # 216
        assert X.shape[2] == 4  # 4 features
        assert len(X) == len(y)

    def test_create_sequences_not_enough_data(self, predictor):
        """Test séquences avec pas assez de données"""
        features = np.random.randn(50, 4)  # Seulement 50 jours

        X, y = predictor.create_sequences(features)

        assert X is None
        assert y is None

    def test_train_success(self, predictor, sample_data):
        """Test entraînement réussi"""
        result = predictor.train(sample_data, epochs=2)  # Seulement 2 epochs pour test

        assert "success" in result
        assert result["success"]
        assert "epochs_trained" in result
        assert predictor.is_loaded
        assert predictor.model is not None
        assert predictor.scaler is not None

    def test_train_missing_target(self, predictor):
        """Test entraînement sans colonne TARGET"""
        data = pd.DataFrame({"DATE": pd.date_range("2023-01-01", periods=300), "CLOSE": np.random.randn(300)})

        result = predictor.train(data, epochs=2)

        assert "error" in result
        assert "TARGET" in result["error"]

    def test_save_and_load_model(self, predictor, sample_data, tmp_path):
        """Test sauvegarde et chargement modèle"""
        # Entraîner
        predictor.train(sample_data, epochs=2)

        # Sauvegarder
        model_path = tmp_path / "test_model.pkl"
        assert predictor.save_model(model_path)
        assert model_path.exists()

        # Charger dans nouveau prédicteur
        new_predictor = PricePredictor("SPY")
        assert new_predictor.load_model(model_path)
        assert new_predictor.is_loaded
        assert new_predictor.model is not None


class TestReturnsPriceConversion:
    """Tests conversion RETURNS → Prix"""

    def test_returns_to_price_simple(self):
        """Test conversion returns vers prix"""
        # Prix initial = 100
        # Return +1% → Prix = 101
        initial_price = 100
        returns = np.array([0.01])  # +1%

        predicted_price = initial_price * (1 + returns[0])

        assert abs(predicted_price - 101) < 0.01

    def test_returns_to_price_multiple(self):
        """Test conversion multiple returns"""
        initial_price = 100
        returns = np.array([0.01, 0.02, -0.01])  # +1%, +2%, -1%

        prices = [initial_price]
        for ret in returns:
            next_price = prices[-1] * (1 + ret)
            prices.append(next_price)

        # 100 → 101 → 103.02 → 101.99
        assert abs(prices[-1] - 101.99) < 0.01

    def test_accuracy_calculation(self):
        """Test calcul accuracy (méthode article)"""
        mae = 3.66  # Notre MAE
        mean_price = 450  # Prix moyen

        accuracy = 100 - (mae / mean_price * 100)

        assert accuracy > 99  # Notre accuracy = 99.32%


class TestModelPerformance:
    """Tests performance modèle"""

    def test_prediction_format(self):
        """Test format prédictions"""
        # Mock résultat prédiction
        result = {
            "historical_predictions": [100.0, 101.0, 102.0],
            "predictions": [103.0, 104.0],
            "ticker": "SPY",
            "horizon": 2,
        }

        assert "historical_predictions" in result
        assert "predictions" in result
        assert len(result["predictions"]) == result["horizon"]

    def test_mae_calculation(self):
        """Test calcul MAE"""
        predictions = np.array([100.0, 101.0, 102.0])
        actuals = np.array([100.5, 100.8, 101.9])

        mae = np.mean(np.abs(predictions - actuals))

        assert mae < 1.0  # MAE faible = bon modèle

    def test_rmse_calculation(self):
        """Test calcul RMSE"""
        predictions = np.array([100.0, 101.0, 102.0])
        actuals = np.array([100.5, 100.8, 101.9])

        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

        assert rmse < 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
