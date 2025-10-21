"""
Fusion Adaptative Prix/Sentiment
Fusion intelligente basée sur les régimes de marché
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from src.constants import CONSTANTS


@dataclass
class MarketRegime:
    """Régime de marché détecté"""

    volatility_level: str  # "low", "medium", "high"
    trend_strength: float  # Force de la tendance (-1 à 1)
    market_stress: float  # Niveau de stress du marché (0 à 1)


@dataclass
class FusionConfig:
    """Configuration de la fusion adaptative"""

    base_price_weight: float = CONSTANTS.BASE_PRICE_WEIGHT
    base_sentiment_weight: float = CONSTANTS.BASE_SENTIMENT_WEIGHT
    base_lstm_weight: float = CONSTANTS.BASE_LSTM_WEIGHT
    max_weight_change: float = CONSTANTS.MAX_WEIGHT_CHANGE
    regularization_factor: float = CONSTANTS.REGULARIZATION_FACTOR
    window_size: int = 20
    volatility_thresholds: tuple = (CONSTANTS.VOLATILITY_LOW_THRESHOLD, CONSTANTS.VOLATILITY_HIGH_THRESHOLD)
    adaptive_thresholds: bool = True  # Activer les seuils adaptatifs


class AdaptiveFusion:
    """
    Système de fusion adaptative de 3 signaux : prix, sentiment et LSTM
    
    Les poids sont adaptés dynamiquement selon le régime de marché :
    - Régime CALME : Privilégie le prix (plus stable et fiable)
    - Régime AGITÉ : Privilégie le sentiment (capte mieux les chocs)
    - Régime NORMAL : Privilégie le LSTM (conditions optimales pour ML)
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        """Initialise la fusion adaptative"""
        self.config = config or FusionConfig()
        self.history: List[Dict[str, Any]] = []
        self.current_weights = {
            "price": self.config.base_price_weight,
            "sentiment": self.config.base_sentiment_weight,
            "lstm": self.config.base_lstm_weight,
        }
        self.current_thresholds = CONSTANTS.NORMAL_VOLATILITY_THRESHOLDS.copy()
        self.rolling_stats = {
            "price_mean": [],
            "price_std": [],
            "sentiment_mean": [],
            "sentiment_std": [],
            "lstm_mean": [],
            "lstm_std": [],
            "correlation": [],
            "volatility": [],
            "volume_ratio": [],
        }

        logger.info(
            f"🔄 Fusion adaptative initialisée (3 signaux) | "
            f"Poids base: prix={self.config.base_price_weight:.2f}, "
            f"sentiment={self.config.base_sentiment_weight:.2f}, lstm={self.config.base_lstm_weight:.2f}"
        )

    def add_signal(
        self, price_signal: float, sentiment_signal: float, price_volatility: float, volume_ratio: float, lstm_signal: float = 0.0
    ) -> Dict[str, Any]:
        """
        Ajoute un nouveau signal et retourne la fusion adaptative

        Args:
            price_signal: Signal de prix normalisé (z-score)
            sentiment_signal: Signal de sentiment normalisé (z-score)
            price_volatility: Volatilité du prix (écart-type)
            volume_ratio: Ratio de volume (volume_actuel / volume_moyen)
            lstm_signal: Signal de prédiction LSTM normalisé (z-score)

        Returns:
            Dict avec la fusion et les métadonnées
        """
        # Détecter le régime de marché
        regime = self._detect_market_regime(price_volatility, volume_ratio)

        # Mettre à jour les seuils adaptatifs
        self._update_adaptive_thresholds(price_volatility, volume_ratio)

        # Calculer les statistiques glissantes
        self._update_rolling_stats(price_signal, sentiment_signal, lstm_signal, price_volatility, volume_ratio)

        # Adapter les poids
        self._adapt_weights(regime)

        # Calculer la fusion
        fused_signal = self._calculate_fusion(price_signal, sentiment_signal, lstm_signal)

        # Créer le résultat
        result = {
            "fused_signal": fused_signal,
            "weights": self.current_weights.copy(),
            "thresholds": self.current_thresholds.copy(),
            "regime": regime,
            "price_signal": price_signal,
            "sentiment_signal": sentiment_signal,
            "lstm_signal": lstm_signal,
            "volatility": price_volatility,
            "volume_ratio": volume_ratio,
            "stats": self._get_current_stats(),
        }

        # Ajouter à l'historique
        self.history.append(result)

        # Garder seulement les dernières entrées
        if len(self.history) > self.config.window_size:
            self.history = self.history[-self.config.window_size :]

        logger.debug(
            f"Signal fusionné: {fused_signal:.4f} | Poids: prix={self.current_weights['price']:.2f}, "
            f"sentiment={self.current_weights['sentiment']:.2f}, lstm={self.current_weights['lstm']:.2f}"
        )

        return result

    def _detect_market_regime(self, volatility: float, volume_ratio: float) -> MarketRegime:
        """Détecte le régime de marché basé sur la volatilité et le volume"""
        # Classification de la volatilité
        if volatility < self.config.volatility_thresholds[0]:
            vol_level = "low"
        elif volatility < self.config.volatility_thresholds[1]:
            vol_level = "medium"
        else:
            vol_level = "high"

        # Calcul de la force de tendance (basé sur les corrélations historiques)
        if len(self.rolling_stats["correlation"]) > 5:
            avg_correlation = np.mean(self.rolling_stats["correlation"][-5:])
            trend_strength = avg_correlation
        else:
            trend_strength = 0.0

        # Calcul du stress du marché (combinaison volatilité + volume)
        market_stress = min(1.0, (volatility * 2 + (volume_ratio - 1) * 0.5))

        return MarketRegime(volatility_level=vol_level, trend_strength=trend_strength, market_stress=market_stress)

    def _update_adaptive_thresholds(self, volatility: float, volume_ratio: float):
        """Met à jour les seuils adaptatifs selon les conditions de marché"""
        if self.config.adaptive_thresholds:
            self.current_thresholds = CONSTANTS.get_adaptive_thresholds(volatility, volume_ratio)
            logger.debug(f"🔧 Seuils adaptatifs mis à jour: {self.current_thresholds}")

    def get_current_thresholds(self) -> Dict[str, float]:
        """Retourne les seuils actuels"""
        return self.current_thresholds.copy()

    def _update_rolling_stats(
        self, price_signal: float, sentiment_signal: float, lstm_signal: float, volatility: float, volume_ratio: float
    ):
        """Met à jour les statistiques glissantes"""
        self.rolling_stats["price_mean"].append(price_signal)
        self.rolling_stats["price_std"].append(abs(price_signal))
        self.rolling_stats["sentiment_mean"].append(sentiment_signal)
        self.rolling_stats["sentiment_std"].append(abs(sentiment_signal))
        self.rolling_stats["lstm_mean"].append(lstm_signal)
        self.rolling_stats["lstm_std"].append(abs(lstm_signal))
        self.rolling_stats["volatility"].append(volatility)
        self.rolling_stats["volume_ratio"].append(volume_ratio)

        # Calculer la corrélation si on a assez de données
        if len(self.rolling_stats["price_mean"]) > 5:
            correlation = np.corrcoef(
                self.rolling_stats["price_mean"][-10:], self.rolling_stats["sentiment_mean"][-10:]
            )[0, 1]
            self.rolling_stats["correlation"].append(correlation if not np.isnan(correlation) else 0.0)
        else:
            self.rolling_stats["correlation"].append(0.0)

        # Garder seulement les dernières valeurs
        for key in self.rolling_stats:
            if len(self.rolling_stats[key]) > self.config.window_size:
                self.rolling_stats[key] = self.rolling_stats[key][-self.config.window_size :]

    def _adapt_weights(self, regime: MarketRegime):
        """Adapte les poids basés sur le régime de marché (3 signaux: price, sentiment, lstm)"""
        old_price_weight = self.current_weights["price"]
        old_sentiment_weight = self.current_weights["sentiment"]
        old_lstm_weight = self.current_weights["lstm"]

        # Poids de base
        price_weight = self.config.base_price_weight
        sentiment_weight = self.config.base_sentiment_weight
        lstm_weight = self.config.base_lstm_weight

        # Adaptation basée sur le régime de volatilité
        if regime.volatility_level == "high":
            # En haute volatilité, privilégier le sentiment (plus réactif aux chocs)
            sentiment_weight += 0.15
            price_weight -= 0.08
            lstm_weight -= 0.07
        elif regime.volatility_level == "low":
            # En basse volatilité, privilégier les prix (plus stables)
            price_weight += 0.15
            sentiment_weight -= 0.08
            lstm_weight -= 0.07
        else:
            # Volatilité normale, privilégier LSTM (conditions optimales pour ML)
            lstm_weight += 0.15
            price_weight -= 0.08
            sentiment_weight -= 0.07

        # Adaptation basée sur la force de tendance
        if abs(regime.trend_strength) > 0.3:
            # Tendance forte, renforcer prix et LSTM
            adjustment = regime.trend_strength * 0.05
            price_weight += adjustment
            lstm_weight += adjustment
            sentiment_weight -= adjustment * 2

        # Adaptation basée sur le stress du marché
        if regime.market_stress > 0.7:
            # Stress élevé, équilibrer tous les signaux
            balance_factor = 0.05
            price_weight += balance_factor
            sentiment_weight += balance_factor
            lstm_weight += balance_factor

        # Limiter les changements
        price_change = price_weight - old_price_weight
        sentiment_change = sentiment_weight - old_sentiment_weight
        lstm_change = lstm_weight - old_lstm_weight

        if abs(price_change) > self.config.max_weight_change:
            price_weight = old_price_weight + np.sign(price_change) * self.config.max_weight_change

        if abs(sentiment_change) > self.config.max_weight_change:
            sentiment_weight = old_sentiment_weight + np.sign(sentiment_change) * self.config.max_weight_change

        if abs(lstm_change) > self.config.max_weight_change:
            lstm_weight = old_lstm_weight + np.sign(lstm_change) * self.config.max_weight_change

        # Normaliser les poids (total = 1.0)
        total_weight = price_weight + sentiment_weight + lstm_weight
        price_weight /= total_weight
        sentiment_weight /= total_weight
        lstm_weight /= total_weight

        # Appliquer la régularisation
        price_weight = (
            1 - self.config.regularization_factor
        ) * price_weight + self.config.regularization_factor * self.config.base_price_weight
        sentiment_weight = (
            1 - self.config.regularization_factor
        ) * sentiment_weight + self.config.regularization_factor * self.config.base_sentiment_weight
        lstm_weight = (
            1 - self.config.regularization_factor
        ) * lstm_weight + self.config.regularization_factor * self.config.base_lstm_weight

        # Normaliser à nouveau
        total_weight = price_weight + sentiment_weight + lstm_weight
        self.current_weights["price"] = price_weight / total_weight
        self.current_weights["sentiment"] = sentiment_weight / total_weight
        self.current_weights["lstm"] = lstm_weight / total_weight

    def _calculate_fusion(self, price_signal: float, sentiment_signal: float, lstm_signal: float) -> float:
        """Calcule la fusion des 3 signaux avec leurs poids adaptatifs"""
        return (
            self.current_weights["price"] * price_signal
            + self.current_weights["sentiment"] * sentiment_signal
            + self.current_weights["lstm"] * lstm_signal
        )

    def _get_current_stats(self) -> Dict[str, float]:
        """Retourne les statistiques actuelles"""
        stats = {}
        for key, values in self.rolling_stats.items():
            if values:
                stats[f"{key}_mean"] = np.mean(values)
                stats[f"{key}_std"] = np.std(values)
                stats[f"{key}_latest"] = values[-1]
            else:
                stats[f"{key}_mean"] = 0.0
                stats[f"{key}_std"] = 0.0
                stats[f"{key}_latest"] = 0.0
        return stats

    def fuse_signals(self, signals: Dict[str, float]) -> float:
        """Fusionne les signaux avec les poids adaptatifs"""
        if not signals:
            return 0.0

        # Mapper les noms de signaux alternatifs
        # "prediction" est un alias pour "lstm"
        signal_mapping = {
            "prediction": "lstm",
            "price": "price",
            "sentiment": "sentiment",
            "lstm": "lstm"
        }

        # Normaliser les signaux et mapper les noms
        normalized_signals = {}
        for signal_type, value in signals.items():
            # Mapper le nom du signal
            mapped_type = signal_mapping.get(signal_type, signal_type)

            if mapped_type in self.current_weights:
                normalized_signals[mapped_type] = np.tanh(value)  # Normaliser entre -1 et 1

        # Extraire les signaux individuels
        price_signal = normalized_signals.get("price", 0.0)
        sentiment_signal = normalized_signals.get("sentiment", 0.0)
        lstm_signal = normalized_signals.get("lstm", 0.0)

        # Utiliser add_signal pour calculer la fusion complète avec détection de régime
        result = self.add_signal(
            price_signal=price_signal,
            sentiment_signal=sentiment_signal,
            lstm_signal=lstm_signal,
            price_volatility=0.1,  # Valeur par défaut
            volume_ratio=1.0  # Valeur par défaut
        )

        return float(result["fused_signal"])

    def get_fusion_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de la fusion adaptative"""
        if not self.history:
            return {
                "total_signals": 0,
                "avg_fused_signal": 0.0,
                "current_weights": self.current_weights,
                "regime": None,
            }

        recent_results = self.history[-10:]  # 10 derniers signaux

        return {
            "total_signals": len(self.history),
            "avg_fused_signal": np.mean([r["fused_signal"] for r in recent_results]),
            "current_weights": self.current_weights,
            "regime": recent_results[-1]["regime"] if recent_results else None,
            "stats": self._get_current_stats(),
        }

    def reset(self):
        """Remet à zéro la fusion adaptative"""
        self.history = []
        self.current_weights = {
            "price": self.config.base_price_weight,
            "sentiment": self.config.base_sentiment_weight,
            "lstm": self.config.base_lstm_weight,
        }
        self.rolling_stats = {key: [] for key in self.rolling_stats}
        logger.info("🔄 Fusion adaptative réinitialisée")
