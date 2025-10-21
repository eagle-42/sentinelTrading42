"""
Service de données pour Streamlit
Chargement et filtrage des données historiques
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger


class DataService:
    """Service de données optimisé pour Streamlit"""

    def __init__(self):
        from src.constants import CONSTANTS
        self.historical_path = CONSTANTS.YFINANCE_DIR
        self.realtime_path = CONSTANTS.PRICES_DIR
        self.features_dir = CONSTANTS.FEATURES_DIR
        self.cache = {}
        logger.info("📊 Service de données initialisé")

    def load_data(self, ticker: str, use_historical: bool = False, use_features: bool = False) -> pd.DataFrame:
        """Charge les données pour un ticker (features en priorité pour LSTM)"""
        try:
            cache_key = f"{ticker}_{'features' if use_features else 'historical' if use_historical else 'combined'}"
            if cache_key in self.cache:
                return self.cache[cache_key]

            if use_features:
                # Pour le modèle LSTM, charger les features techniques
                features_file = self.features_dir / f"{ticker.lower()}_features.parquet"
                if features_file.exists():
                    df = self._load_features_data(features_file, ticker)
                    if not df.empty:
                        self.cache[cache_key] = df
                        logger.info(f"✅ Features chargées pour {ticker}: {len(df)} lignes")
                        return df

            if use_historical:
                # Pour les analyses long terme, combiner historique + temps réel
                df_combined = self._load_combined_data(ticker)
                if not df_combined.empty:
                    self.cache[cache_key] = df_combined
                    logger.info(f"✅ Données combinées chargées pour {ticker}: {len(df_combined)} lignes")
                    return df_combined
            else:
                # Essayer d'abord les données temps réel
                realtime_file = self.realtime_path / f"{ticker.lower()}_15min.parquet"
                if realtime_file.exists():
                    df = self._load_realtime_data(realtime_file, ticker)
                    if not df.empty:
                        self.cache[cache_key] = df
                        logger.info(f"✅ Données temps réel chargées pour {ticker}: {len(df)} lignes")
                        return df

                # Fallback vers les données combinées
                df_combined = self._load_combined_data(ticker)
                if not df_combined.empty:
                    self.cache[cache_key] = df_combined
                    logger.info(f"✅ Données combinées chargées pour {ticker}: {len(df_combined)} lignes")
                    return df_combined

            logger.error(f"❌ Aucune donnée trouvée pour {ticker}")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"❌ Erreur chargement {ticker}: {e}")
            return pd.DataFrame()

    def _load_combined_data(self, ticker: str) -> pd.DataFrame:
        """Combine les données historiques et temps réel pour avoir les données les plus récentes"""
        try:
            # Charger les données historiques
            historical_file = self.historical_path / f"{ticker}_1999_2025.parquet"
            df_hist = pd.DataFrame()
            if historical_file.exists():
                df_hist = self._load_historical_data(historical_file, ticker)

            # Charger les données temps réel
            realtime_file = self.realtime_path / f"{ticker.lower()}_15min.parquet"
            df_realtime = pd.DataFrame()
            if realtime_file.exists():
                df_realtime = self._load_realtime_data(realtime_file, ticker)

            if df_hist.empty and df_realtime.empty:
                return pd.DataFrame()

            if df_hist.empty:
                return df_realtime

            if df_realtime.empty:
                return df_hist

            # Combiner les données en évitant les doublons
            # Utiliser les données temps réel pour les dates récentes
            last_hist_date = df_hist.index.max()
            first_realtime_date = df_realtime.index.min()

            # Garder les données historiques jusqu'à la veille des données temps réel
            df_hist_filtered = df_hist[df_hist.index < first_realtime_date]

            # Combiner
            df_combined = pd.concat([df_hist_filtered, df_realtime])
            df_combined = df_combined.sort_index()

            # Supprimer les doublons potentiels
            df_combined = df_combined[~df_combined.index.duplicated(keep="last")]

            logger.info(
                f"✅ Données combinées: {len(df_hist_filtered)} historiques + {len(df_realtime)} temps réel = {len(df_combined)} total"
            )
            return df_combined

        except Exception as e:
            logger.error(f"❌ Erreur combinaison données {ticker}: {e}")
            return pd.DataFrame()

    def _load_realtime_data(self, file_path: Path, ticker: str) -> pd.DataFrame:
        """Charge les données temps réel"""
        df = pd.read_parquet(file_path)

        # Convertir ts_utc en DATE et l'utiliser comme index
        df["DATE"] = pd.to_datetime(df["ts_utc"], utc=True)
        df = df.set_index("DATE")

        # Normalisation des colonnes (majuscules)
        df.columns = df.columns.str.upper()

        # Garder seulement les colonnes nécessaires
        keep_cols = ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
        df = df[[col for col in keep_cols if col in df.columns]]

        # Tri par date
        df = df.sort_index()

        return self._validate_data(df, ticker)

    def _load_features_data(self, file_path: Path, ticker: str) -> pd.DataFrame:
        """Charge les features techniques pour le modèle LSTM"""
        df = pd.read_parquet(file_path)

        # Filtrer par ticker
        if "TICKER" in df.columns:
            df = df[df["TICKER"] == ticker].copy()
            df = df.drop("TICKER", axis=1)

        # Utiliser DATE comme index
        if "DATE" in df.columns:
            df["DATE"] = pd.to_datetime(df["DATE"], utc=True)
            df = df.set_index("DATE")

        # Normalisation des colonnes (majuscules)
        df.columns = df.columns.str.upper()

        # Garder toutes les features techniques nécessaires pour LSTM (en majuscules)
        feature_cols = [
            "RSI_14",
            "MACD",
            "BB_POSITION",
            "STOCH_K",
            "WILLIAMS_R",
            "EMA_RATIO",
            "ATR_NORMALIZED",
            "VOLUME_RATIO",
            "PRICE_POSITION",
            "ROC_10",
            "RETURNS",
            "RETURNS_MA_5",
            "RETURNS_MA_10",
            "RETURNS_MA_20",
            "RETURNS_MA_50",
            "MOMENTUM_5",
            "MOMENTUM_10",
            "MOMENTUM_20",
            "VOLUME_PRICE_TREND",
            "PRICE_VELOCITY",
            "PRICE_POSITION_20",
        ]

        # Garder les colonnes disponibles
        available_cols = [col for col in feature_cols if col in df.columns]
        if available_cols:
            df = df[available_cols]
        else:
            # Fallback vers les colonnes OHLCV si pas de features
            fallback_cols = ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
            df = df[[col for col in fallback_cols if col in df.columns]]

        # Tri par date
        df = df.sort_index()

        # Nettoyer les NaN
        df = df.ffill().bfill()

        # Normaliser les colonnes en majuscules pour correspondre aux features attendues
        df.columns = df.columns.str.upper()

        logger.info(f"✅ Features chargées pour {ticker}: {len(df)} lignes, {len(df.columns)} colonnes")
        return df

    def _load_historical_data(self, file_path: Path, ticker: str) -> pd.DataFrame:
        """Charge les données historiques"""
        df = pd.read_parquet(file_path)

        # Normalisation des colonnes (majuscules)
        df.columns = df.columns.str.upper()

        # Filtrer par ticker si nécessaire
        if "TICKER" in df.columns:
            df = df[df["TICKER"] == ticker].copy()
            df = df.drop("TICKER", axis=1)

        # Conversion des dates en UTC
        if "DATE" in df.columns:
            df["DATE"] = pd.to_datetime(df["DATE"], utc=True)
            df = df.set_index("DATE")

        # Tri par date
        df = df.sort_index()

        return self._validate_data(df, ticker)

    def _validate_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Valide et nettoie les données"""
        if df.empty:
            return df

        # Vérifier les colonnes requises (CLOSE et VOLUME, DATE est l'index)
        required_cols = ["CLOSE", "VOLUME"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.warning(f"⚠️ Colonnes manquantes pour {ticker}: {missing_cols}")
            return pd.DataFrame()

        # Nettoyer les valeurs NaN/Inf
        df = df.replace([np.inf, -np.inf], np.nan)

        # Supprimer les lignes avec des prix invalides
        df = df.dropna(subset=["CLOSE"])

        # Vérifier que les prix sont positifs
        df = df[df["CLOSE"] > 0]

        if df.empty:
            logger.error(f"❌ Aucune donnée valide pour {ticker}")
            return pd.DataFrame()

        logger.info(f"✅ Données validées pour {ticker}: {len(df)} lignes valides")
        return df

    def filter_by_period(self, df: pd.DataFrame, period: str) -> pd.DataFrame:
        """Filtre les données par période"""
        if df.empty:
            return df

        periods = {
            "7 derniers jours": 7,
            "1 mois": 30,
            "3 mois": 90,
            "6 derniers mois": 180,
            "1 an": 365,
            "3 ans": 1095,
            "5 ans": 1825,
            "10 ans": 3650,
            "Total (toutes les données)": None,
        }

        if period not in periods:
            logger.warning(f"⚠️ Période inconnue: {period}")
            return df

        days = periods[period]
        if days is None:
            return df

        # Utiliser la dernière date des données comme référence
        last_date = df["DATE"].max()
        start_date = last_date - pd.Timedelta(days=days)

        # Filtrer et trier
        filtered_df = df[df["DATE"] >= start_date].copy()
        filtered_df = filtered_df.sort_values("DATE").reset_index(drop=True)

        logger.info(f"✅ Filtrage {period}: {len(filtered_df)} lignes")
        return filtered_df

    def get_available_tickers(self) -> list:
        """Retourne la liste des tickers disponibles"""
        if not self.historical_path.exists():
            return []

        tickers = []
        for file_path in self.historical_path.glob("*.parquet"):
            ticker = file_path.stem.replace("_1999_2025", "")
            tickers.append(ticker)

        return sorted(tickers)
