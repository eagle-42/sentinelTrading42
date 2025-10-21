"""
Service de fusion pour Streamlit
Combine prédictions prix et sentiment
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    from src.core.fusion import AdaptiveFusion

    FUSION_AVAILABLE = True
    logger.info("✅ AdaptiveFusion importé avec succès")
except ImportError as e:
    logger.warning(f"⚠️ AdaptiveFusion non disponible: {e}")
    FUSION_AVAILABLE = False
    AdaptiveFusion = None

try:
    from src.gui.services.decision_validation_service import DecisionValidationService

    VALIDATION_AVAILABLE = True
    logger.info("✅ DecisionValidationService importé avec succès")
except ImportError as e:
    logger.warning(f"⚠️ DecisionValidationService non disponible: {e}")
    VALIDATION_AVAILABLE = False
    DecisionValidationService = None


class FusionService:
    """Service de fusion adaptative pour l'onglet Production"""

    def __init__(self):
        self.fusion_engine = None
        self.fusion_history = []
        self.validation_service = None

        if FUSION_AVAILABLE:
            try:
                self.fusion_engine = AdaptiveFusion()
                logger.info("✅ Moteur de fusion initialisé")
            except Exception as e:
                logger.warning(f"⚠️ Erreur initialisation fusion: {e}")

        if VALIDATION_AVAILABLE:
            try:
                self.validation_service = DecisionValidationService()
                logger.info("✅ Service de validation initialisé")
            except Exception as e:
                logger.warning(f"⚠️ Erreur initialisation validation: {e}")

    def calculate_fusion_score(
        self, price_signal: float, sentiment_signal: float, prediction_signal: float, market_regime: str = "normal"
    ) -> Dict[str, Any]:
        """Calcule le score de fusion final avec seuils adaptatifs"""
        try:
            if self.fusion_engine:
                # Utiliser le vrai moteur de fusion
                signals = {"price": price_signal, "sentiment": sentiment_signal, "prediction": prediction_signal}
                fusion_score = self.fusion_engine.fuse_signals(signals)
                confidence = 0.8  # Valeur par défaut
                weights = self.fusion_engine.current_weights
                thresholds = self.fusion_engine.get_current_thresholds()
            else:
                # Fallback simulation
                fusion_score = self._simulate_fusion_score(price_signal, sentiment_signal, prediction_signal)
                confidence = 0.75
                weights = {"price": 0.4, "sentiment": 0.3, "prediction": 0.3}
                thresholds = {"buy": 0.05, "sell": -0.05}

            # Déterminer la recommandation avec seuils adaptatifs
            recommendation = self._get_adaptive_recommendation(fusion_score, thresholds)

            # Sauvegarder dans l'historique
            fusion_data = {
                "timestamp": datetime.now(),
                "fusion_score": fusion_score,
                "confidence": confidence,
                "recommendation": recommendation,
                "price_signal": price_signal,
                "sentiment_signal": sentiment_signal,
                "prediction_signal": prediction_signal,
                "weights": weights,
                "thresholds": thresholds,
            }
            self.fusion_history.append(fusion_data)

            # Garder seulement les 100 dernières entrées
            if len(self.fusion_history) > 100:
                self.fusion_history = self.fusion_history[-100:]

            return {
                "fusion_score": fusion_score,
                "confidence": confidence,
                "recommendation": recommendation,
                "weights": weights,
                "color": self._get_score_color(fusion_score),
                "label": self._get_score_label(fusion_score),
            }
        except Exception as e:
            logger.error(f"❌ Erreur calcul fusion: {e}")
            return {
                "fusion_score": 0.5,
                "confidence": 0.5,
                "recommendation": "ATTENDRE",
                "weights": {"price": 0.33, "sentiment": 0.33, "prediction": 0.34},
                "color": "gray",
                "label": "Neutre",
            }

    def get_multi_signal_chart_data(self, ticker: str = "SPY") -> Dict[str, Any]:
        """Récupère les données pour le graphique multi-signaux"""
        try:
            # Simulation de données (en production, récupérer depuis la base)
            dates = pd.date_range(end=datetime.now(), periods=30, freq="D")

            # Signaux simulés
            price_signals = np.random.uniform(0.3, 0.8, 30)
            sentiment_signals = np.random.uniform(0.2, 0.9, 30)
            prediction_signals = np.random.uniform(0.1, 0.7, 30)

            # Calculer les scores de fusion
            fusion_signals = []
            for i in range(30):
                fusion_data = self.calculate_fusion_score(price_signals[i], sentiment_signals[i], prediction_signals[i])
                fusion_signals.append(fusion_data["fusion_score"])

            return {
                "dates": dates.tolist(),
                "price_signals": price_signals.tolist(),
                "sentiment_signals": sentiment_signals.tolist(),
                "prediction_signals": prediction_signals.tolist(),
                "fusion_signals": fusion_signals,
            }
        except Exception as e:
            logger.error(f"❌ Erreur données graphique: {e}")
            return {
                "dates": [],
                "price_signals": [],
                "sentiment_signals": [],
                "prediction_signals": [],
                "fusion_signals": [],
            }

    def get_fusion_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Récupère l'historique des fusions"""
        try:
            return self.fusion_history[-limit:] if self.fusion_history else []
        except Exception as e:
            logger.error(f"❌ Erreur historique fusion: {e}")
            return []

    def _get_adaptive_recommendation(self, fusion_score: float, thresholds: Dict[str, float]) -> str:
        """Détermine la recommandation avec seuils adaptatifs"""
        # Seuils plus sensibles pour générer plus de BUY/SELL
        buy_threshold = thresholds.get("buy", 0.05)  # Réduit de 0.1 à 0.05
        sell_threshold = thresholds.get("sell", -0.05)  # Réduit de -0.1 à -0.05

        if fusion_score > buy_threshold:
            return "BUY"
        elif fusion_score < sell_threshold:
            return "SELL"
        else:
            return "HOLD"

    def get_fusion_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques de fusion"""
        try:
            if not self.fusion_history:
                return {
                    "total_signals": 0,
                    "avg_score": 0.5,
                    "last_recommendation": "ATTENDRE",
                    "current_weights": {"price": 0.33, "sentiment": 0.33, "prediction": 0.34},
                    "current_thresholds": {"buy": 0.05, "sell": -0.05},
                }

            recent_data = self.fusion_history[-10:]  # 10 dernières entrées
            avg_score = np.mean([d["fusion_score"] for d in recent_data])
            last_recommendation = recent_data[-1]["recommendation"] if recent_data else "ATTENDRE"
            current_thresholds = recent_data[-1].get("thresholds", {"buy": 0.05, "sell": -0.05})
            current_weights = (
                recent_data[-1]["weights"] if recent_data else {"price": 0.33, "sentiment": 0.33, "prediction": 0.34}
            )

            return {
                "total_signals": len(self.fusion_history),
                "avg_score": avg_score,
                "last_recommendation": last_recommendation,
                "current_weights": current_weights,
                "current_thresholds": current_thresholds,
            }
        except Exception as e:
            logger.error(f"❌ Erreur stats fusion: {e}")
            return {
                "total_signals": 0,
                "avg_score": 0.5,
                "last_recommendation": "ATTENDRE",
                "current_weights": {"price": 0.33, "sentiment": 0.33, "prediction": 0.34},
            }

    def _simulate_fusion_score(self, price: float, sentiment: float, prediction: float) -> float:
        """FONCTION DÉSACTIVÉE - Simulation interdite selon les règles du projet"""
        logger.error("❌ _simulate_fusion_score() DÉSACTIVÉE - Simulations interdites")
        # Retourner une valeur neutre pour éviter les erreurs
        return 0.5

    def _get_recommendation(self, score: float, confidence: float) -> str:
        """Détermine la recommandation basée sur le score et la confiance"""
        if confidence < 0.6:
            return "ATTENDRE"
        elif score > 0.7:
            return "ACHETER"
        elif score < 0.3:
            return "VENDRE"
        else:
            return "ATTENDRE"

    def _get_score_color(self, score: float) -> str:
        """Détermine la couleur basée sur le score"""
        if score > 0.7:
            return "green"
        elif score > 0.5:
            return "blue"
        elif score > 0.3:
            return "orange"
        else:
            return "red"

    def validate_decision(
        self, ticker: str, decision: str, fusion_score: float, current_price: float
    ) -> Dict[str, Any]:
        """
        Valide une décision de trading en temps réel

        Args:
            ticker: Symbole de l'action
            decision: Décision prise (BUY/SELL/HOLD)
            fusion_score: Score de fusion utilisé
            current_price: Prix actuel

        Returns:
            Dict contenant les résultats de validation
        """
        try:
            if not self.validation_service:
                return {
                    "status": "validation_unavailable",
                    "message": "Service de validation non disponible",
                    "accuracy": None,
                    "price_change": None,
                    "validation_time": None,
                    "is_correct": None,
                }

            # Valider la décision
            validation_result = self.validation_service.validate_decision(
                ticker=ticker,
                decision=decision,
                fusion_score=fusion_score,
                current_price=current_price,
                timestamp=datetime.now(),
            )

            return validation_result

        except Exception as e:
            logger.error(f"❌ Erreur validation décision: {e}")
            return {
                "status": "error",
                "message": f"Erreur validation: {str(e)}",
                "accuracy": None,
                "price_change": None,
                "validation_time": None,
                "is_correct": None,
            }

    def get_validation_stats(self, ticker: str, days: int = 7) -> Dict[str, Any]:
        """
        Récupère les statistiques de validation pour un ticker

        Args:
            ticker: Symbole de l'action
            days: Nombre de jours à analyser

        Returns:
            Dict contenant les statistiques de validation
        """
        try:
            if not self.validation_service:
                return {
                    "total_decisions": 0,
                    "total_validations": 0,
                    "correct_decisions": 0,
                    "accuracy_rate": 0.0,
                    "average_accuracy": 0.0,
                    "buy_decisions": 0,
                    "sell_decisions": 0,
                    "hold_decisions": 0,
                    "buy_accuracy": 0.0,
                    "sell_accuracy": 0.0,
                }

            return self.validation_service.get_validation_stats(ticker, days)

        except Exception as e:
            logger.error(f"❌ Erreur récupération stats validation: {e}")
            return {
                "total_decisions": 0,
                "total_validations": 0,
                "correct_decisions": 0,
                "accuracy_rate": 0.0,
                "average_accuracy": 0.0,
                "buy_decisions": 0,
                "sell_decisions": 0,
                "hold_decisions": 0,
                "buy_accuracy": 0.0,
                "sell_accuracy": 0.0,
            }

    def get_adaptive_threshold_performance(self, ticker: str, days: int = 7) -> Dict[str, Any]:
        """
        Analyse la performance des seuils adaptatifs

        Args:
            ticker: Symbole de l'action
            days: Nombre de jours à analyser

        Returns:
            Dict contenant l'analyse de performance des seuils
        """
        try:
            if not self.validation_service:
                return {
                    "threshold_analysis": "Service de validation non disponible",
                    "recommended_adjustments": [],
                    "performance_score": 0.0,
                }

            return self.validation_service.get_adaptive_threshold_performance(ticker, days)

        except Exception as e:
            logger.error(f"❌ Erreur analyse performance seuils: {e}")
            return {"threshold_analysis": f"Erreur: {str(e)}", "recommended_adjustments": [], "performance_score": 0.0}

    def _get_score_label(self, score: float) -> str:
        """Convertit le score en label"""
        if score > 0.8:
            return "Très Fort"
        elif score > 0.6:
            return "Fort"
        elif score > 0.4:
            return "Modéré"
        elif score > 0.2:
            return "Faible"
        else:
            return "Très Faible"
