"""
Tests unitaires pour DataService
Conforme aux bonnes pratiques de test Streamlit
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ajouter le path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from gui.services.data_service import DataService


class TestDataService:
    """Tests unitaires pour DataService"""

    def test_init(self):
        """Test d'initialisation du service"""
        service = DataService()
        assert service.data_path is not None
        assert isinstance(service.cache, dict)

    def test_load_data_empty_file(self, tmp_path):
        """Test avec fichier vide"""
        service = DataService()

        # Créer un fichier parquet vide
        empty_file = tmp_path / "empty.parquet"
        pd.DataFrame().to_parquet(empty_file)

        # Modifier temporairement le path
        original_path = service.data_path
        service.data_path = tmp_path

        try:
            result = service.load_data("empty")
            assert result.empty
        finally:
            service.data_path = original_path

    def test_filter_period_7_days(self):
        """Test du filtrage sur 7 jours"""
        service = DataService()

        # Créer des données de test
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame({"DATE": dates, "CLOSE": np.random.randn(10) * 100 + 500})

        result = service._filter_period(df, "7 derniers jours")

        # Vérifier que seules les 7 dernières dates sont conservées
        assert len(result) == 7
        assert result["DATE"].min() == dates[-7]
        assert result["DATE"].max() == dates[-1]

    def test_filter_period_1_month(self):
        """Test du filtrage sur 1 mois"""
        service = DataService()

        # Créer des données de test sur 2 mois
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        df = pd.DataFrame({"DATE": dates, "CLOSE": np.random.randn(60) * 100 + 500})

        result = service._filter_period(df, "1 mois")

        # Vérifier que seules les 30 dernières dates sont conservées
        assert len(result) <= 30  # Approximativement 1 mois
        assert result["DATE"].min() >= dates[-30]

    def test_calculate_moving_averages(self):
        """Test du calcul des moyennes mobiles"""
        service = DataService()

        # Créer des données de test
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        prices = np.random.randn(100) * 10 + 500
        df = pd.DataFrame({"DATE": dates, "CLOSE": prices})

        result = service._calculate_moving_averages(df)

        # Vérifier que les colonnes MA sont présentes
        assert "MA_20" in result.columns
        assert "MA_50" in result.columns
        assert "MA_100" in result.columns

        # Vérifier que les MA sont calculées correctement
        assert not result["MA_20"].isna().all()
        assert not result["MA_50"].isna().all()
        assert not result["MA_100"].isna().all()

    def test_calculate_volatility(self):
        """Test du calcul de la volatilité"""
        service = DataService()

        # Créer des données de test
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        volume = np.random.randint(1000, 10000, 50)
        df = pd.DataFrame({"DATE": dates, "VOLUME": volume})

        result = service._calculate_volatility(df)

        # Vérifier que la colonne VOLATILITY est présente
        assert "VOLATILITY" in result.columns

        # Vérifier que la volatilité est calculée
        assert not result["VOLATILITY"].isna().all()
        assert result["VOLATILITY"].min() >= 0  # La volatilité ne peut pas être négative


if __name__ == "__main__":
    pytest.main([__file__])
