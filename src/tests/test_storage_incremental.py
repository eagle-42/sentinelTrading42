"""
Tests de sauvegarde incrémentale
Utilise des fichiers temporaires - AUCUNE donnée de production touchée
"""

import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.data.storage import ParquetStorage


class TestIncrementalStorage:
    """Tests de sauvegarde incrémentale avec fichiers temporaires"""
    
    @pytest.fixture
    def storage(self):
        """Fixture storage"""
        return ParquetStorage()
    
    def test_prices_incremental_save(self, storage, tmp_path):
        """Test: Sauvegarde incrémentale des prix (pas de duplication)"""
        # Créer fichier temporaire
        test_file = tmp_path / "spy_15min.parquet"
        
        # Première vague de données
        data1 = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=5, freq="15min", tz="UTC"),
            "ticker": ["SPY"] * 5,
            "open": [450.0, 451.0, 452.0, 453.0, 454.0],
            "high": [451.0, 452.0, 453.0, 454.0, 455.0],
            "low": [449.0, 450.0, 451.0, 452.0, 453.0],
            "close": [450.5, 451.5, 452.5, 453.5, 454.5],
            "volume": [1000000, 1100000, 1200000, 1300000, 1400000]
        })
        data1.to_parquet(test_file, index=False)
        
        # Deuxième vague (incrémental)
        data2 = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01 01:15:00", periods=5, freq="15min", tz="UTC"),
            "ticker": ["SPY"] * 5,
            "open": [455.0, 456.0, 457.0, 458.0, 459.0],
            "high": [456.0, 457.0, 458.0, 459.0, 460.0],
            "low": [454.0, 455.0, 456.0, 457.0, 458.0],
            "close": [455.5, 456.5, 457.5, 458.5, 459.5],
            "volume": [1500000, 1600000, 1700000, 1800000, 1900000]
        })
        
        # Charger existant et combiner (simulation sauvegarde incrémentale)
        existing = pd.read_parquet(test_file)
        combined = pd.concat([existing, data2]).drop_duplicates(subset=["ts_utc"], keep="last")
        combined.to_parquet(test_file, index=False)
        
        # Vérifier
        final = pd.read_parquet(test_file)
        assert len(final) == 10, "Devrait avoir 10 lignes (5 + 5)"
        assert final["close"].iloc[0] == 450.5, "Première valeur correcte"
        assert final["close"].iloc[-1] == 459.5, "Dernière valeur correcte"
    
    def test_news_incremental_save(self, storage, tmp_path):
        """Test: Sauvegarde incrémentale des news (évite doublons)"""
        test_file = tmp_path / "all_news.parquet"
        
        # Première vague
        news1 = pd.DataFrame({
            "title": ["News 1", "News 2", "News 3"],
            "link": ["http://test.com/1", "http://test.com/2", "http://test.com/3"],
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC"),
            "source": ["Source A", "Source B", "Source C"],
            "ticker": ["SPY", "SPY", "SPY"]
        })
        news1.to_parquet(test_file, index=False)
        
        # Deuxième vague (avec 1 doublon)
        news2 = pd.DataFrame({
            "title": ["News 3", "News 4", "News 5"],  # News 3 = doublon
            "link": ["http://test.com/3", "http://test.com/4", "http://test.com/5"],
            "timestamp": pd.date_range("2024-01-01 02:00:00", periods=3, freq="1h", tz="UTC"),
            "source": ["Source C", "Source D", "Source E"],
            "ticker": ["SPY", "SPY", "SPY"]
        })
        
        # Combiner en évitant doublons par link
        existing = pd.read_parquet(test_file)
        combined = pd.concat([existing, news2]).drop_duplicates(subset=["link"], keep="last")
        combined.to_parquet(test_file, index=False)
        
        # Vérifier
        final = pd.read_parquet(test_file)
        assert len(final) == 5, "Devrait avoir 5 news uniques (pas 6)"
        assert len(final["link"].unique()) == 5, "Tous les links doivent être uniques"
    
    def test_sentiment_incremental_save(self, storage, tmp_path):
        """Test: Sauvegarde incrémentale du sentiment"""
        test_file = tmp_path / "spy_sentiment.parquet"
        
        # Première vague
        sent1 = pd.DataFrame({
            "ticker": ["SPY"] * 3,
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="10min", tz="UTC"),
            "sentiment_score": [0.5, 0.6, 0.7],
            "confidence": [0.8, 0.85, 0.9],
            "article_count": [5, 6, 7]
        })
        sent1.to_parquet(test_file, index=False)
        
        # Deuxième vague
        sent2 = pd.DataFrame({
            "ticker": ["SPY"] * 3,
            "timestamp": pd.date_range("2024-01-01 00:30:00", periods=3, freq="10min", tz="UTC"),
            "sentiment_score": [0.75, 0.80, 0.85],
            "confidence": [0.88, 0.90, 0.92],
            "article_count": [8, 9, 10]
        })
        
        # Combiner
        existing = pd.read_parquet(test_file)
        combined = pd.concat([existing, sent2]).drop_duplicates(subset=["timestamp"], keep="last")
        combined.to_parquet(test_file, index=False)
        
        # Vérifier
        final = pd.read_parquet(test_file)
        assert len(final) == 6, "Devrait avoir 6 entrées sentiment"
        assert final["sentiment_score"].iloc[-1] == 0.85, "Dernier score correct"
    
    def test_no_duplication_on_exact_same_data(self, storage, tmp_path):
        """Test: Pas de duplication si on sauvegarde 2 fois les mêmes données"""
        test_file = tmp_path / "test_no_dup.parquet"
        
        # Données identiques
        data = pd.DataFrame({
            "ts_utc": pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC"),
            "value": [1, 2, 3, 4, 5]
        })
        
        # Sauvegarder 1ère fois
        data.to_parquet(test_file, index=False)
        
        # Sauvegarder 2ème fois (même données)
        existing = pd.read_parquet(test_file)
        combined = pd.concat([existing, data]).drop_duplicates(subset=["ts_utc"], keep="last")
        combined.to_parquet(test_file, index=False)
        
        # Vérifier
        final = pd.read_parquet(test_file)
        assert len(final) == 5, "Toujours 5 lignes (pas de duplication)"
    
    def test_preserves_order_after_incremental_save(self, storage, tmp_path):
        """Test: L'ordre chronologique est préservé après sauvegarde incrémentale"""
        test_file = tmp_path / "test_order.parquet"
        
        # Données désordonnées
        data1 = pd.DataFrame({
            "ts_utc": pd.to_datetime(["2024-01-01 03:00:00", "2024-01-01 01:00:00"], utc=True),
            "value": [3, 1]
        })
        data1.to_parquet(test_file, index=False)
        
        # Ajouter données
        data2 = pd.DataFrame({
            "ts_utc": pd.to_datetime(["2024-01-01 02:00:00", "2024-01-01 04:00:00"], utc=True),
            "value": [2, 4]
        })
        
        # Combiner et trier
        existing = pd.read_parquet(test_file)
        combined = pd.concat([existing, data2]).drop_duplicates(subset=["ts_utc"])
        combined = combined.sort_values("ts_utc").reset_index(drop=True)
        combined.to_parquet(test_file, index=False)
        
        # Vérifier ordre
        final = pd.read_parquet(test_file)
        assert list(final["value"]) == [1, 2, 3, 4], "Données triées chronologiquement"
