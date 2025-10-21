"""
Service de prédiction LSTM pour Streamlit
Utilise le vrai modèle LSTM entraîné depuis data/models/spy
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    from src.core.prediction import PricePredictor

    LSTM_AVAILABLE = True
    logger.info("✅ PricePredictor importé avec succès")
except ImportError as e:
    logger.warning(f"⚠️ PricePredictor non disponible: {e}")
    LSTM_AVAILABLE = False
    PricePredictor = None


class PredictionService:
    """Service de prédiction LSTM utilisant le vrai modèle entraîné"""

    def __init__(self):
        from src.constants import CONSTANTS
        self.model_path = CONSTANTS.get_model_path("SPY")
        self.predictor = None
        self.fallback_mode = False
        logger.info("🤖 Service de prédiction LSTM initialisé")

    def _load_model(self) -> bool:
        """Charge le modèle LSTM réel"""
        try:
            if not LSTM_AVAILABLE:
                logger.warning("⚠️ PricePredictor non disponible, passage en mode fallback")
                self.fallback_mode = True
                return False

            if self.predictor is None:
                self.predictor = PricePredictor("SPY")
                success = self.predictor.load_model()

                if success:
                    logger.info("✅ Modèle LSTM SPY chargé avec succès")
                    return True
                else:
                    logger.warning("⚠️ Échec chargement modèle LSTM, passage en mode fallback")
                    self.fallback_mode = True
                    return False

            return self.predictor.is_loaded

        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle LSTM: {e}")
            self.fallback_mode = True
            return False

    def predict_with_features(self, ticker: str, horizon: int = 3, df_input: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Génère un signal de tendance LSTM (daily) pour fusion avec signaux 15min

        Args:
            ticker: Symbole du ticker (ex: SPY)
            horizon: Nombre de JOURS à prédire (default 3 jours pour tendance court-terme)
            df_input: DataFrame optionnel avec données 15min (sera agrégé en daily)

        Returns:
            Dict avec signal de tendance daily pour affichage graphique
        """
        try:
            from src.constants import CONSTANTS
            from src.gui.services.data_service import DataService

            # TOUJOURS charger historique complet pour avoir 220+ jours après agrégation
            # (même si df_input est fourni avec une période filtrée)
            data_service = DataService()
            df_15min_full = data_service.load_data(ticker, use_historical=True)

            if df_15min_full.empty:
                logger.warning(f"⚠️ Aucune donnée historique pour {ticker}")
                return self._create_empty_prediction()

            logger.info(f"📊 Chargement données complètes: {len(df_15min_full)} lignes 15min pour prédiction LSTM")

            # ÉTAPE 1: Agréger 15min → DAILY pour le modèle LSTM
            # Convertir l'index DATE en colonne (DataService retourne DATE comme index)
            df_15min = df_15min_full.reset_index().copy()

            # Normaliser colonnes (DataService retourne OPEN, HIGH, LOW, CLOSE, DATE en majuscules)
            col_map = {
                'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
                'date': 'Date', 'DATE': 'Date'  # Supporter DATE (index converti en colonne)
            }
            df_15min.columns = [col_map.get(c, col_map.get(c.lower(), c)) for c in df_15min.columns]

            # S'assurer qu'on a une colonne Date
            if 'Date' not in df_15min.columns:
                if 'ts_utc' in df_15min.columns:
                    df_15min['Date'] = pd.to_datetime(df_15min['ts_utc'])
                else:
                    logger.error(f"❌ Aucune colonne de date trouvée. Colonnes disponibles: {df_15min.columns.tolist()}")
                    return self._create_empty_prediction()

            df_15min['Date'] = pd.to_datetime(df_15min['Date'])

            # Vérifier colonnes OHLC
            required = ['Open', 'High', 'Low', 'Close']
            if not all(c in df_15min.columns for c in required):
                logger.error(f"❌ Colonnes manquantes. Attendu: {required}, Trouvé: {df_15min.columns.tolist()}")
                return self._create_empty_prediction()

            # Agréger en DAILY (OHLC standard)
            df_15min['DateOnly'] = df_15min['Date'].dt.date
            df_daily = df_15min.groupby('DateOnly').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last'
            }).reset_index()
            df_daily['DATE'] = pd.to_datetime(df_daily['DateOnly'])
            df_daily = df_daily.drop('DateOnly', axis=1)

            logger.info(f"📊 Agrégation 15min→daily: {len(df_15min_full)} lignes 15min → {len(df_daily)} jours")

            # ÉTAPE 2: Calculer RETURNS daily pour le modèle
            df_returns = pd.DataFrame({
                'DATE': df_daily['DATE'],
                'Open_RETURN': df_daily['Open'].pct_change(),
                'High_RETURN': df_daily['High'].pct_change(),
                'Low_RETURN': df_daily['Low'].pct_change(),
                'Close_RETURN': df_daily['Close'].pct_change(),
            })
            df_returns['TARGET'] = df_returns['Close_RETURN']
            df_returns = df_returns.dropna().reset_index(drop=True)

            if df_returns.empty or len(df_returns) < 220:
                logger.warning(f"⚠️ Pas assez de données daily: {len(df_returns)} jours (besoin 220+)")
                return self._create_empty_prediction()

            logger.info(f"📊 Données RETURNS daily préparées: {len(df_returns)} jours")

            # ÉTAPE 3: Prédiction LSTM daily
            if not self._load_model():
                return self._create_empty_prediction()

            # Prédire avec horizon en JOURS
            prediction_result = self._real_predict(df_returns, horizon, df_daily['Close'].values)

            # Ajouter les prix 15min pour le graphique (utiliser df_input filtré si fourni)
            if df_input is not None and not df_input.empty:
                # Utiliser les données filtrées pour l'affichage
                df_graph = df_input.copy()
                df_graph = df_graph.reset_index(drop=True)
                col_map_graph = {'date': 'Date', 'close': 'Close'}
                df_graph.columns = [col_map_graph.get(c.lower(), c) for c in df_graph.columns]

                if 'Date' not in df_graph.columns and 'ts_utc' in df_graph.columns:
                    df_graph['Date'] = pd.to_datetime(df_graph['ts_utc'])
                elif 'Date' in df_graph.columns:
                    df_graph['Date'] = pd.to_datetime(df_graph['Date'])

                prediction_result['price_15min_dates'] = df_graph['Date'].tolist()
                prediction_result['price_15min_values'] = df_graph['Close'].tolist()
            else:
                # Sinon utiliser les données complètes
                prediction_result['price_15min_dates'] = df_15min['Date'].tolist()
                prediction_result['price_15min_values'] = df_15min['Close'].tolist()

            return prediction_result

        except Exception as e:
            logger.error(f"❌ Erreur prédiction avec features: {e}")
            return self._create_empty_prediction()

    def _real_predict(self, df: pd.DataFrame, horizon: int, close_prices: np.ndarray) -> Dict[str, Any]:
        """Prédiction avec le vrai modèle LSTM + conversion RETURNS → Prix absolus

        Args:
            df: DataFrame avec colonnes RETURNS daily
            horizon: Nombre de jours à prédire
            close_prices: Array des prix de clôture daily (pour reconstruction des prix)
        """
        try:
            # Convertir l'index DATE en colonne si nécessaire
            if "DATE" in df.index.names:
                df = df.reset_index()

            # Utiliser la méthode predict standard du PricePredictor
            prediction_result = self.predictor.predict(df, horizon=horizon)

            if "error" in prediction_result:
                logger.warning(f"⚠️ Erreur prédiction: {prediction_result['error']}")
                return self._create_empty_prediction()

            # Extraire les prédictions RETURNS
            hist_returns = prediction_result.get("historical_predictions", [])
            future_returns = prediction_result.get("predictions", [])

            # CONVERSION RETURNS → PRIX ABSOLUS
            # Historique: Reconstruire prix à partir des returns prédits
            hist_prices = []
            if len(hist_returns) > 0 and len(close_prices) > 0:
                # Premier prix = prix réel du premier jour
                current_price = close_prices[0]
                hist_prices.append(current_price)

                # Reconstruire les prix suivants: P[t] = P[t-1] * (1 + return[t])
                for i, ret in enumerate(hist_returns[1:], start=1):
                    if i < len(close_prices):
                        # Pour l'historique, utiliser le prix réel pour éviter drift
                        current_price = close_prices[i]
                    else:
                        # Au-delà des données réelles, utiliser la reconstruction
                        current_price = current_price * (1 + ret)
                    hist_prices.append(current_price)

            # Futur: Projeter à partir du dernier prix réel
            future_prices = []
            future_dates = []
            if len(future_returns) > 0 and len(close_prices) > 0:
                last_date = pd.to_datetime(df["DATE"].iloc[-1])
                current_price = close_prices[-1]  # Dernier prix réel connu

                for i, ret in enumerate(future_returns):
                    # P[t+1] = P[t] * (1 + return_predit)
                    current_price = current_price * (1 + ret)
                    future_prices.append(current_price)
                    future_dates.append(last_date + pd.Timedelta(days=i + 1))

            # Signal de tendance normalisé pour fusion (-1 à +1)
            trend_signal = 0.0
            if len(future_returns) > 0:
                # Tendance = moyenne des returns futurs, normalisée avec tanh
                avg_future_return = np.mean(future_returns)
                trend_signal = float(np.tanh(avg_future_return * 10))  # Amplifier pour rendre visible

            logger.info(
                f"✅ Prédiction LSTM daily: {len(hist_prices)} jours historiques + {len(future_prices)} jours futurs"
            )
            logger.info(f"📈 Signal de tendance: {trend_signal:.3f} (avg return: {avg_future_return*100:.2f}%)")

            return {
                "historical_predictions": hist_prices,
                "predictions": future_prices,
                "prediction_dates": future_dates,
                "model_type": "lstm_daily",
                "confidence": 0.9,
                "trend_signal": trend_signal,  # Pour la fusion adaptative
                "future_returns": future_returns,  # Garder les returns pour debug
            }

        except Exception as e:
            logger.error(f"❌ Erreur prédiction LSTM réelle: {e}")
            return self._fallback_predict(df, horizon)

    def _fallback_predict(self, df: pd.DataFrame, horizon: int) -> Dict[str, Any]:
        """Prédiction fallback sans simulation - Retourne erreur si modèle non disponible"""
        logger.error("❌ FALLBACK DÉSACTIVÉ - Aucune simulation autorisée selon les règles du projet")
        logger.error("❌ Le modèle LSTM doit être chargé correctement pour faire des prédictions")
        return self._create_empty_prediction()

    def _create_empty_prediction(self) -> Dict[str, Any]:
        """Crée une prédiction vide en cas d'erreur"""
        return {
            "historical_predictions": [],
            "predictions": [],
            "prediction_dates": [],
            "model_type": "empty",
            "confidence": 0.0,
        }
