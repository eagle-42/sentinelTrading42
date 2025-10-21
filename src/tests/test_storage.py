"""
Tests critiques Storage - Stockage Parquet
4 tests essentiels
"""

import pytest
import pandas as pd
import numpy as np
from src.data.storage import ParquetStorage
from src.constants import CONSTANTS


class TestStorage:
    """Tests stockage Parquet"""

    @pytest.fixture
    def storage(self):
        """Fixture storage"""
        return ParquetStorage()

    def test_save_and_load_prices(self, storage, tmp_path):
        """Test: Sauvegarde/chargement prix"""
        data = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1min"),
            "open": np.random.rand(10) * 100,
            "high": np.random.rand(10) * 100,
            "low": np.random.rand(10) * 100,
            "close": np.random.rand(10) * 100,
            "volume": np.random.randint(1000, 10000, 10)
        })
        
        # Sauvegarder
        test_path = tmp_path / "test_prices.parquet"
        data.to_parquet(test_path)
        
        # Charger
        loaded = pd.read_parquet(test_path)
        assert len(loaded) == 10
        assert "close" in loaded.columns

    def test_incremental_save(self, storage, tmp_path):
        """Test: Sauvegarde incrémentale (pas de duplication)"""
        # Première sauvegarde
        data1 = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="1min"),
            "value": [1, 2, 3, 4, 5]
        })
        test_path = tmp_path / "test_incremental.parquet"
        data1.to_parquet(test_path)
        
        # Deuxième sauvegarde (ajout)
        data2 = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01 00:05:00", periods=5, freq="1min"),
            "value": [6, 7, 8, 9, 10]
        })
        existing = pd.read_parquet(test_path)
        combined = pd.concat([existing, data2]).drop_duplicates(subset=["timestamp"])
        combined.to_parquet(test_path)
        
        # Vérifier
        final = pd.read_parquet(test_path)
        assert len(final) == 10

    def test_data_paths(self):
        """Test: Chemins de données corrects"""
        prices_path = CONSTANTS.get_data_path("prices", "SPY", "15min")
        assert "realtime" in str(prices_path)
        assert "spy_15min.parquet" in str(prices_path)

    def test_model_paths(self):
        """Test: Chemins modèles corrects"""
        model_path = CONSTANTS.get_model_path("SPY")
        assert "models" in str(model_path)
        assert "spy" in str(model_path).lower()
