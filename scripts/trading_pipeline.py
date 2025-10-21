#!/usr/bin/env python3
"""
🤖 Pipeline de trading complet
Exécute le pipeline de trading avec fusion des signaux et prise de décision
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.constants import CONSTANTS
from src.core.fusion import AdaptiveFusion
from src.core.prediction import PricePredictor
from src.core.sentiment import SentimentAnalyzer
from src.data.storage import DataStorage
from src.gui.services.decision_validation_service import DecisionValidationService


class TradingPipeline:
    """Pipeline de trading complet avec fusion des signaux"""

    def __init__(self):
        self.storage = DataStorage()
        self.tickers = CONSTANTS.TICKERS
        self.fusion = AdaptiveFusion()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.prediction_engine = PricePredictor()
        self.decision_validator = DecisionValidationService()

        # Configuration de trading
        self.buy_threshold = CONSTANTS.BASE_BUY_THRESHOLD
        self.sell_threshold = CONSTANTS.BASE_SELL_THRESHOLD
        self.hold_confidence = CONSTANTS.HOLD_CONFIDENCE

        # État de trading
        self.trading_state = self._load_trading_state()

    def _load_trading_state(self) -> Dict[str, Any]:
        """Charge l'état de trading depuis le fichier"""
        state_path = CONSTANTS.DATA_ROOT / "trading" / "decisions_log" / "trading_state.json"

        if state_path.exists():
            try:
                with open(state_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"⚠️ Erreur chargement état trading: {e}")

        # État par défaut
        return {
            "last_update": None,
            "positions": {},
            "decisions": [],
            "performance": {"total_decisions": 0, "correct_decisions": 0, "accuracy": 0.0},
        }

    def _save_trading_state(self):
        """Sauvegarde l'état de trading"""
        state_path = CONSTANTS.DATA_ROOT / "trading" / "decisions_log" / "trading_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(state_path, "w") as f:
                json.dump(self.trading_state, f, indent=2)
            logger.debug(f"💾 État de trading sauvegardé: {state_path}")
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde état trading: {e}")

    def get_latest_prices(self, ticker: str) -> Optional[pd.DataFrame]:
        """Récupère les dernières données de prix (yfinance pour LSTM)"""
        try:
            # Charger données historiques yfinance (suffisant pour LSTM)
            yfinance_path = CONSTANTS.DATA_ROOT / "historical" / "yfinance" / f"{ticker.upper()}_1999_2025.parquet"

            if yfinance_path.exists():
                data = pd.read_parquet(yfinance_path)

                # Renommer 'date' en 'ts_utc' pour uniformiser
                if 'date' in data.columns:
                    data = data.rename(columns={'date': 'ts_utc'})

                data['ts_utc'] = pd.to_datetime(data['ts_utc'])

                # Trier et prendre les 500 dernières lignes (largement assez pour LSTM 216)
                data = data.sort_values("ts_utc").tail(500)

                # Renommer colonnes pour compatibilité
                data = data.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close'
                })

                logger.debug(f"📊 Données yfinance {ticker}: {len(data)} lignes ({data['ts_utc'].min()} → {data['ts_utc'].max()})")
                return data

            # Fallback: données temps réel 15min (si yfinance absent)
            file_path = CONSTANTS.get_data_path("prices", ticker, "15min")
            if file_path.exists():
                data = pd.read_parquet(file_path)
                if not data.empty:
                    data = data.sort_values("ts_utc").tail(500)
                    data = data.rename(columns={
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close'
                    })
                    logger.debug(f"📊 Données temps réel {ticker}: {len(data)} lignes")
                    return data

            logger.warning(f"⚠️ Aucune donnée de prix pour {ticker}")
            return None

        except Exception as e:
            logger.error(f"❌ Erreur récupération prix {ticker}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def get_latest_sentiment(self, ticker: str) -> float:
        """Récupère le dernier sentiment pour un ticker"""
        try:
            # Chercher le fichier de sentiment le plus récent
            sentiment_dir = CONSTANTS.SENTIMENT_DIR
            sentiment_files = list(sentiment_dir.glob("sentiment_*.parquet"))

            if not sentiment_files:
                logger.warning(f"⚠️ Aucun fichier de sentiment pour {ticker}")
                return 0.0

            # Prendre le fichier le plus récent
            latest_file = max(sentiment_files, key=lambda x: x.stat().st_mtime)
            data = pd.read_parquet(latest_file)

            # Filtrer pour le ticker et prendre la dernière valeur
            ticker_data = data[data["ticker"] == ticker]
            if ticker_data.empty:
                logger.warning(f"⚠️ Aucun sentiment pour {ticker}")
                return 0.0

            latest_sentiment = ticker_data.iloc[-1]["sentiment_score"]
            logger.debug(f"📊 Sentiment {ticker}: {latest_sentiment:.3f}")
            return float(latest_sentiment)

        except Exception as e:
            logger.error(f"❌ Erreur récupération sentiment {ticker}: {e}")
            return 0.0

    def calculate_price_signal(self, prices: pd.DataFrame) -> float:
        """Calcule le signal de prix basé sur les données récentes"""
        if prices.empty or len(prices) < 2:
            return 0.0

        try:
            # Calculer le retour sur la période
            close_col = 'Close' if 'Close' in prices.columns else 'close'
            current_price = prices[close_col].iloc[-1]
            previous_price = prices[close_col].iloc[-2]

            # Retour simple
            price_return = (current_price - previous_price) / previous_price

            # Normaliser le signal entre -1 et 1
            price_signal = np.tanh(price_return * 10)  # Amplifier les petits changements

            logger.debug(f"📈 Signal prix: {price_return:.4f} -> {price_signal:.3f}")
            return float(price_signal)

        except Exception as e:
            logger.error(f"❌ Erreur calcul signal prix: {e}")
            return 0.0

    def get_lstm_prediction(self, ticker: str, prices: pd.DataFrame) -> Optional[float]:
        """Récupère la prédiction LSTM pour un ticker (modèle 4 RETURNS)"""
        try:
            # Préparer les données comme le modèle attend : 4 RETURNS
            # Open_RETURN, High_RETURN, Low_RETURN, TARGET (Close_RETURN)
            
            # Vérifier colonnes nécessaires
            required_cols = ['Open', 'High', 'Low', 'Close']
            prices_cols = [col for col in required_cols if col in prices.columns or col.lower() in prices.columns or col.upper() in prices.columns]
            
            if len(prices_cols) < 4:
                logger.warning(f"⚠️ Colonnes OHLC manquantes pour {ticker}")
                return None
            
            # Normaliser noms colonnes
            df = prices.copy()
            df.columns = df.columns.str.title()  # Open, High, Low, Close
            
            logger.debug(f"📊 Avant pct_change: {len(df)} lignes, colonnes: {df.columns.tolist()}")
            logger.debug(f"📊 Colonnes OHLC présentes: Open={('Open' in df.columns)}, High={('High' in df.columns)}, Low={('Low' in df.columns)}, Close={('Close' in df.columns)}")
            
            # Calculer les RETURNS
            df['Open_RETURN'] = df['Open'].pct_change()
            df['High_RETURN'] = df['High'].pct_change()
            df['Low_RETURN'] = df['Low'].pct_change()
            df['TARGET'] = df['Close'].pct_change()
            
            # Supprimer NaN SEULEMENT sur les colonnes RETURNS (pas tout le DataFrame)
            df = df.dropna(subset=['Open_RETURN', 'High_RETURN', 'Low_RETURN', 'TARGET'])
            
            logger.debug(f"📊 Après dropna: {len(df)} lignes")
            
            if len(df) < 220:  # Besoin de 216 + marge
                logger.warning(f"⚠️ Pas assez de données pour {ticker}: {len(df)} < 220")
                return None
            
            # Sélectionner seulement les colonnes RETURNS
            features_df = df[['Open_RETURN', 'High_RETURN', 'Low_RETURN', 'TARGET']].copy()
            
            # Normaliser noms colonnes en MAJUSCULES
            features_df.columns = features_df.columns.str.upper()
            
            # Initialiser le prédicteur
            predictor = PricePredictor(ticker)

            # Charger le modèle
            if not predictor.load_model():
                logger.warning(f"⚠️ Impossible de charger le modèle pour {ticker}")
                return None

            # Utiliser la méthode predict_with_technical_features (qui attend RETURNS)
            # Créer séquences manuellement
            from sklearn.preprocessing import MinMaxScaler
            scaler = predictor.scaler
            
            features_scaled = scaler.transform(features_df.values)
            X, y = predictor.create_sequences(features_scaled)
            
            if X is None or len(X) == 0:
                logger.warning(f"⚠️ Impossible de créer séquences pour {ticker}")
                return None
            
            # Prédiction
            import torch
            with torch.no_grad():
                sequence = torch.FloatTensor(X[-1:]).to(predictor.device)
                pred_scaled = predictor.model(sequence).cpu().numpy()[0, 0]
            
            # Dénormaliser
            dummy = [[pred_scaled, 0, 0, 0]]
            pred_return = scaler.inverse_transform(dummy)[0, 0]
            
            # Le signal = return prédit (déjà en %)
            signal = pred_return
            
            current_price = df['Close'].iloc[-1]
            predicted_price = current_price * (1 + pred_return)
            
            logger.debug(f"🔮 Prédiction LSTM {ticker}: ${current_price:.2f} -> ${predicted_price:.2f} (return: {pred_return:.3%})")
            return signal

        except Exception as e:
            logger.error(f"❌ Erreur prédiction LSTM {ticker}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def make_trading_decision(
        self, ticker: str, price_signal: float, sentiment_signal: float, prediction_signal: Optional[float] = None, prices: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Prend une décision de trading basée sur les signaux"""

        # Préparer les signaux pour la fusion (3 composantes)
        signals = {"price": price_signal, "sentiment": sentiment_signal}

        if prediction_signal is not None:
            signals["prediction"] = prediction_signal  # ← Signal LSTM

        # Fusionner les signaux
        fused_signal = self.fusion.fuse_signals(signals)

        # Prendre la décision
        decision = "HOLD"
        confidence = 0.0

        if fused_signal > self.buy_threshold:
            decision = "BUY"
            confidence = min(fused_signal, 1.0)
        elif fused_signal < self.sell_threshold:
            decision = "SELL"
            confidence = min(abs(fused_signal), 1.0)
        else:
            decision = "HOLD"
            confidence = 1.0 - abs(fused_signal)

        # Créer la décision
        # Récupérer le prix actuel
        if prices is not None and not prices.empty:
            close_col = 'Close' if 'Close' in prices.columns else 'close'
            current_price = float(prices[close_col].iloc[-1])
        else:
            # Fallback: récupérer depuis le fichier
            price_df = self.get_latest_prices(ticker)
            if price_df is not None and not price_df.empty:
                close_col = 'Close' if 'Close' in price_df.columns else 'close'
                current_price = float(price_df[close_col].iloc[-1])
            else:
                current_price = 0.0
        
        decision_data = {
            "ticker": ticker,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision": decision,
            "confidence": confidence,
            "fused_signal": fused_signal,
            "current_price": current_price,
            "signals": signals,
            "thresholds": {"buy": self.buy_threshold, "sell": self.sell_threshold, "hold": self.hold_confidence},
        }

        logger.info(f"🤖 Décision {ticker}: {decision} (confiance: {confidence:.3f}, signal: {fused_signal:.3f}, prix: ${current_price:.2f})")
        
        # Sauvegarder comme décision en attente de validation (15 minutes) - TOUTES les décisions
        validation_result = self.decision_validator.validate_decision(
            ticker=ticker,
            decision=decision,
            fusion_score=fused_signal,
            current_price=current_price,
            timestamp=datetime.now(timezone.utc)
        )
        if decision != "HOLD":
            logger.info(f"⏳ Décision en attente de validation: {validation_result.get('message', 'N/A')}")

        return decision_data

    def process_ticker(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Traite un ticker et génère une décision de trading"""
        logger.info(f"\n📊 Traitement {ticker}")

        # Récupérer les données
        prices = self.get_latest_prices(ticker)
        if prices is None:
            logger.warning(f"⚠️ Impossible de traiter {ticker}: pas de données de prix")
            return None

        # Calculer les signaux
        price_signal = self.calculate_price_signal(prices)
        sentiment_signal = self.get_latest_sentiment(ticker)
        prediction_signal = self.get_lstm_prediction(ticker, prices)

        # Prendre la décision
        decision = self.make_trading_decision(ticker, price_signal, sentiment_signal, prediction_signal, prices)

        return decision

    def _is_decision_window(self) -> bool:
        """Vérifie si on est dans une fenêtre de décision valide (15 minutes)"""
        try:
            import pytz

            # Timezone US Eastern (gère automatiquement EST/EDT)
            us_eastern = pytz.timezone("US/Eastern")
            now_est = datetime.now(us_eastern)

            # Heures de marché US (9:30-16:00)
            market_open = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now_est.replace(hour=16, minute=0, second=0, microsecond=0)

            # Vérifier si c'est un jour de semaine
            is_weekday = now_est.weekday() < 5  # 0-4 = lundi-vendredi

            if not is_weekday:
                return False

            if not (market_open <= now_est <= market_close):
                return False

            # Vérifier si on est dans une fenêtre de 15 minutes
            current_minute = now_est.minute
            return current_minute in [30, 45, 0]  # 9:30, 9:45, 10:00, 10:15, etc.

        except ImportError:
            # Fallback si pytz n'est pas disponible
            from datetime import timedelta, timezone

            edt = timezone(timedelta(hours=-4))
            now_est = datetime.now(edt)

            # Heures de marché US (9:30-16:00 EDT)
            market_open = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now_est.replace(hour=16, minute=0, second=0, microsecond=0)

            # Vérifier si c'est un jour de semaine
            is_weekday = now_est.weekday() < 5  # 0-4 = lundi-vendredi

            if not is_weekday:
                return False

            if not (market_open <= now_est <= market_close):
                return False

            # Vérifier si on est dans une fenêtre de 15 minutes
            current_minute = now_est.minute
            return current_minute in [30, 45, 0]  # 9:30, 9:45, 10:00, 10:15, etc.

    def run_trading_pipeline(self, force: bool = False) -> Dict[str, Any]:
        """Exécute le pipeline de trading complet"""
        logger.info("🤖 === PIPELINE DE TRADING ===")
        start_time = datetime.now()

        # Valider les décisions en attente AVANT de générer de nouvelles décisions
        logger.info("🔍 Vérification des décisions en attente de validation...")
        validated_count = self.decision_validator.process_pending_validations()
        if validated_count > 0:
            logger.info(f"✅ {validated_count} décision(s) validée(s)")

        # Vérifier si on est dans une fenêtre de décision valide (15 minutes)
        if not force and not self._is_decision_window():
            logger.info("⏰ Pas dans une fenêtre de décision (15min) - Attente")
            return {
                "success": True,
                "decisions": [],
                "tickers_processed": 0,
                "duration": (datetime.now() - start_time).total_seconds(),
            }
        
        if force:
            logger.info("🔧 Mode FORCE activé - Génération décision immédiate")

        decisions = []
        successful_tickers = 0

        for ticker in self.tickers:
            try:
                decision = self.process_ticker(ticker)
                if decision:
                    decisions.append(decision)
                    successful_tickers += 1
            except Exception as e:
                logger.error(f"❌ Erreur traitement {ticker}: {e}")
                continue

        # Mettre à jour l'état de trading
        self.trading_state["last_update"] = datetime.now().isoformat()
        self.trading_state["decisions"].extend(decisions)

        # Garder seulement les 100 dernières décisions
        if len(self.trading_state["decisions"]) > 100:
            self.trading_state["decisions"] = self.trading_state["decisions"][-100:]

        # Sauvegarder l'état
        self._save_trading_state()

        # Sauvegarder les décisions dans un fichier séparé
        self._save_decisions_log(decisions)

        # Calculer les métriques
        duration = (datetime.now() - start_time).total_seconds()

        result = {
            "timestamp": datetime.now().isoformat(),
            "tickers_processed": successful_tickers,
            "total_tickers": len(self.tickers),
            "decisions": decisions,
            "duration_seconds": duration,
            "status": "success" if successful_tickers > 0 else "no_data",
        }

        logger.info(f"\n📊 Résumé du pipeline:")
        logger.info(f"   Tickers traités: {successful_tickers}/{len(self.tickers)}")
        logger.info(f"   Décisions générées: {len(decisions)}")
        logger.info(f"   Durée: {duration:.1f}s")

        return result

    def _save_decisions_log(self, decisions: List[Dict[str, Any]]):
        """Sauvegarde le log des décisions dans un fichier unifié"""
        if not decisions:
            return

        # Fichier unifié pour toutes les décisions
        log_path = CONSTANTS.DATA_ROOT / "trading" / "decisions_log" / "trading_decisions.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Charger les décisions existantes
        existing_decisions = []
        if log_path.exists():
            try:
                with open(log_path, "r") as f:
                    existing_decisions = json.load(f)
                logger.debug(f"📊 Décisions existantes: {len(existing_decisions)}")
            except Exception as e:
                logger.warning(f"⚠️ Erreur lecture décisions existantes: {e}")

        # Fusionner les nouvelles décisions
        all_decisions = existing_decisions + decisions

        # Garder seulement les 1000 dernières décisions pour éviter la surcharge
        if len(all_decisions) > 1000:
            all_decisions = all_decisions[-1000:]
            logger.info(
                f"📊 Décisions limitées à 1000 (supprimé {len(existing_decisions) + len(decisions) - 1000} anciennes)"
            )

        try:
            with open(log_path, "w") as f:
                json.dump(all_decisions, f, indent=2)
            logger.debug(f"💾 Log des décisions sauvegardé: {log_path} ({len(all_decisions)} total)")
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde log décisions: {e}")


def main():
    """Fonction principale"""
    import sys
    force = "--force" in sys.argv
    
    logger.info("🚀 Démarrage du pipeline de trading")
    if force:
        logger.info("🔧 Mode FORCE détecté")

    try:
        pipeline = TradingPipeline()
        result = pipeline.run_trading_pipeline(force=force)

        if result.get("success", False):
            logger.info("✅ Pipeline de trading terminé avec succès")
            logger.info(f"   Décisions: {len(result.get('decisions', []))}")
            return 0
        else:
            logger.warning("⚠️ Pipeline de trading terminé sans données")
            return 1

    except Exception as e:
        logger.error(f"❌ Erreur lors du pipeline de trading: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
