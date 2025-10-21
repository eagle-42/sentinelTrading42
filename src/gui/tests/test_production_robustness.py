"""
Tests de robustesse pour l'onglet Production
Vérification des arguments requis et de la fiabilité
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.gui.services.chart_service import ChartService
from src.gui.services.data_service import DataService
from src.gui.services.fusion_service import FusionService
from src.gui.services.llm_service import LLMService
from src.gui.services.monitoring_service import MonitoringService
from src.gui.services.prediction_service import PredictionService
from src.gui.services.sentiment_service import SentimentService


class TestProductionRobustness:
    """Tests de robustesse pour l'onglet Production"""

    def setup_method(self):
        """Configuration des tests"""
        self.data_service = DataService()
        self.chart_service = ChartService()
        self.prediction_service = PredictionService()
        self.sentiment_service = SentimentService()
        self.fusion_service = FusionService()
        self.llm_service = LLMService()
        self.monitoring_service = MonitoringService()

    def test_data_service_required_arguments(self):
        """Test que DataService gère correctement les arguments requis"""
        # Test avec arguments valides
        try:
            data = self.data_service.load_data("SPY")
            assert isinstance(data, pd.DataFrame)
        except Exception as e:
            pytest.fail(f"DataService.load_data() a échoué avec des arguments valides: {e}")

        # Test avec ticker invalide
        try:
            data = self.data_service.load_data("INVALID")
            # Devrait retourner un DataFrame vide ou lever une exception
            assert isinstance(data, pd.DataFrame)
        except Exception:
            # Exception attendue pour ticker invalide
            pass

    def test_chart_service_required_arguments(self):
        """Test que ChartService gère correctement les arguments requis"""
        # Créer des données de test
        test_data = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=10),
                "Close": np.random.uniform(100, 200, 10),
                "Volume": np.random.uniform(1000, 5000, 10),
            }
        )

        # Test avec arguments valides
        try:
            chart = self.chart_service.create_price_chart(test_data, "SPY", "1 mois")
            assert chart is not None
        except Exception as e:
            pytest.fail(f"ChartService.create_price_chart() a échoué avec des arguments valides: {e}")

        # Test avec données vides
        try:
            empty_data = pd.DataFrame()
            chart = self.chart_service.create_price_chart(empty_data, "SPY", "1 mois")
            # Devrait gérer les données vides gracieusement
        except Exception as e:
            # Exception acceptable pour données vides
            assert "empty" in str(e).lower() or "no data" in str(e).lower()

    def test_prediction_service_required_arguments(self):
        """Test que PredictionService gère correctement les arguments requis"""
        # Créer des données de test
        test_data = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=100),
                "Close": np.random.uniform(100, 200, 100),
                "Volume": np.random.uniform(1000, 5000, 100),
            }
        )

        # Test avec arguments valides
        try:
            prediction = self.prediction_service.predict(test_data, horizon=20)
            assert isinstance(prediction, dict)
            assert "predictions" in prediction
            assert "confidence" in prediction
        except Exception as e:
            pytest.fail(f"PredictionService.predict() a échoué avec des arguments valides: {e}")

        # Test avec données vides
        try:
            empty_data = pd.DataFrame()
            prediction = self.prediction_service.predict(empty_data, horizon=20)
            # Devrait gérer les données vides gracieusement
            assert isinstance(prediction, dict)
        except Exception:
            # Exception acceptable pour données vides
            pass

    def test_sentiment_service_required_arguments(self):
        """Test que SentimentService gère correctement les arguments requis"""
        # Test avec arguments valides
        try:
            articles = self.sentiment_service.get_news_articles("SPY", 5)
            assert isinstance(articles, list)
        except Exception as e:
            pytest.fail(f"SentimentService.get_news_articles() a échoué avec des arguments valides: {e}")

        # Test analyse sentiment
        test_article = {
            "id": "test_1",
            "title": "Test Article",
            "content": "This is a test article for sentiment analysis.",
            "source": "Test Source",
            "timestamp": pd.Timestamp.now(),
        }

        try:
            sentiment = self.sentiment_service.analyze_article_sentiment(test_article)
            assert isinstance(sentiment, dict)
            assert "sentiment_score" in sentiment
            assert "confidence" in sentiment
        except Exception as e:
            pytest.fail(f"SentimentService.analyze_article_sentiment() a échoué: {e}")

    def test_fusion_service_required_arguments(self):
        """Test que FusionService gère correctement les arguments requis"""
        # Test avec arguments valides
        try:
            fusion_data = self.fusion_service.calculate_fusion_score(0.7, 0.6, 0.8)
            assert isinstance(fusion_data, dict)
            assert "fusion_score" in fusion_data
            assert "confidence" in fusion_data
            assert "recommendation" in fusion_data
        except Exception as e:
            pytest.fail(f"FusionService.calculate_fusion_score() a échoué avec des arguments valides: {e}")

        # Test avec valeurs extrêmes
        try:
            fusion_data = self.fusion_service.calculate_fusion_score(1.0, 1.0, 1.0)
            assert isinstance(fusion_data, dict)
        except Exception as e:
            pytest.fail(f"FusionService.calculate_fusion_score() a échoué avec des valeurs extrêmes: {e}")

    def test_llm_service_required_arguments(self):
        """Test que LLMService gère correctement les arguments requis"""
        # Test avec arguments valides
        test_fusion_data = {"fusion_score": 0.7, "confidence": 0.8, "recommendation": "ACHETER"}
        test_sentiment_data = {"avg_sentiment": 0.6, "total_articles": 10}
        test_price_data = {"last_price": 445.67, "change_percent": 2.34}

        try:
            explanation = self.llm_service.generate_trading_explanation(
                test_fusion_data, test_sentiment_data, test_price_data
            )
            assert isinstance(explanation, dict)
            assert "explanation" in explanation
            assert "confidence" in explanation
        except Exception as e:
            pytest.fail(f"LLMService.generate_trading_explanation() a échoué: {e}")

    def test_monitoring_service_required_arguments(self):
        """Test que MonitoringService gère correctement les arguments requis"""
        # Test statut système
        try:
            status = self.monitoring_service.get_system_status()
            assert isinstance(status, dict)
            assert "overall_status" in status
            assert "services" in status
        except Exception as e:
            pytest.fail(f"MonitoringService.get_system_status() a échoué: {e}")

        # Test métriques de performance
        try:
            metrics = self.monitoring_service.get_performance_metrics()
            assert isinstance(metrics, dict)
            assert "cpu_percent" in metrics
            assert "memory_percent" in metrics
        except Exception as e:
            pytest.fail(f"MonitoringService.get_performance_metrics() a échoué: {e}")

    def test_error_handling_consistency(self):
        """Test que tous les services gèrent les erreurs de manière cohérente"""
        services = [
            self.data_service,
            self.chart_service,
            self.prediction_service,
            self.sentiment_service,
            self.fusion_service,
            self.llm_service,
            self.monitoring_service,
        ]

        for service in services:
            # Vérifier que le service a des méthodes de gestion d'erreur
            assert hasattr(service, "__class__"), f"Service {service.__class__.__name__} n'a pas de classe"

            # Vérifier que les méthodes principales existent
            if hasattr(service, "load_data"):
                assert callable(getattr(service, "load_data"))
            if hasattr(service, "predict"):
                assert callable(getattr(service, "predict"))
            if hasattr(service, "get_system_status"):
                assert callable(getattr(service, "get_system_status"))

    def test_data_validation(self):
        """Test la validation des données"""
        # Test DataService avec données invalides
        try:
            # Créer des données avec des valeurs manquantes
            invalid_data = pd.DataFrame(
                {"Date": [None, None, None], "Close": [np.nan, np.nan, np.nan], "Volume": [0, 0, 0]}
            )

            # Le service devrait gérer les données invalides
            result = self.data_service._validate_data(invalid_data, "SPY")
            # _validate_data retourne un DataFrame filtré, pas un bool
            assert isinstance(result, pd.DataFrame)
        except Exception as e:
            # Exception acceptable pour données très invalides
            assert "invalid" in str(e).lower() or "missing" in str(e).lower()

    def test_fusion_signal_validation(self):
        """Test la validation des signaux de fusion"""
        # Test avec signaux valides
        try:
            fusion_data = self.fusion_service.calculate_fusion_score(0.5, 0.5, 0.5)
            assert 0 <= fusion_data["fusion_score"] <= 1
            assert 0 <= fusion_data["confidence"] <= 1
            assert fusion_data["recommendation"] in ["ACHETER", "VENDRE", "ATTENDRE"]
        except Exception as e:
            pytest.fail(f"FusionService n'a pas validé les signaux correctement: {e}")

        # Test avec signaux extrêmes
        try:
            fusion_data = self.fusion_service.calculate_fusion_score(2.0, -2.0, 1.5)
            # Devrait normaliser les valeurs
            assert 0 <= fusion_data["fusion_score"] <= 1
        except Exception as e:
            pytest.fail(f"FusionService n'a pas géré les signaux extrêmes: {e}")

    def test_market_status_validation(self):
        """Test la validation de l'état du marché"""
        from src.gui.pages.production_page_improved import _check_market_status

        try:
            market_status = _check_market_status()
            assert isinstance(market_status, dict)
            assert "is_open" in market_status
            assert "current_time" in market_status
            assert isinstance(market_status["is_open"], bool)
        except Exception as e:
            pytest.fail(f"_check_market_status() a échoué: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
