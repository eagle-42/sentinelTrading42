"""
Analyse de Sentiment FinBERT
Analyse de sentiment financier avec FinBERT et agrégation temporelle
"""

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger

from src.constants import CONSTANTS


class FinBertAnalyzer:
    """Analyseur de sentiment basé sur FinBERT"""

    def __init__(self, mode: str = None, timeout_ms: int = None):
        """Initialise l'analyseur FinBERT"""
        self.mode = mode or CONSTANTS.FINBERT_MODE
        self.timeout_ms = timeout_ms or CONSTANTS.FINBERT_TIMEOUT_MS
        self.model = None
        self.tokenizer = None
        self.device = None
        self._model_loaded = False

        logger.info(f"💭 FinBERT initialisé - Mode: {self.mode}")

    def _lazy_load_model(self):
        """Chargement paresseux du modèle FinBERT"""
        if self._model_loaded:
            return

        if self.mode == "stub":
            logger.info("💭 Mode stub - FinBERT simulé")
            self._model_loaded = True
            return

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            logger.info("💭 Chargement du modèle FinBERT...")

            # Charger le modèle et tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(CONSTANTS.FINBERT_MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(CONSTANTS.FINBERT_MODEL_NAME)

            # Configuration du device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

            self.model.to(self.device)
            self.model.eval()

            self._model_loaded = True
            logger.info(f"💭 FinBERT chargé sur {self.device}")

        except Exception as e:
            logger.error(f"❌ Erreur chargement FinBERT: {e}")
            logger.info("💭 Basculement en mode stub")
            self.mode = "stub"
            self._model_loaded = True

    def score_texts(self, texts: List[str]) -> List[float]:
        """Score des textes avec FinBERT"""
        self._lazy_load_model()

        if not texts:
            return []

        if self.mode == "stub":
            return self._score_texts_stub(texts)

        start_time = time.time()

        try:
            import torch

            # Tokenize texts
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                if self.device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)

            # FinBERT has 3 classes: 0=negative, 1=neutral, 2=positive
            probabilities = torch.softmax(outputs.logits, dim=-1)
            sentiment_scores = []

            for prob in probabilities:
                # Calculate weighted sentiment score
                negative_score = prob[0].item()
                neutral_score = prob[1].item()
                positive_score = prob[2].item()

                # Convert to -1 to 1 scale
                sentiment = (positive_score - negative_score) * (1 - neutral_score)
                sentiment_scores.append(sentiment)

            # Record latency
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"💭 FinBERT traité {len(texts)} textes en {elapsed_ms:.1f}ms")

            return sentiment_scores

        except Exception as e:
            logger.error(f"❌ Erreur FinBERT: {e}")
            return self._score_texts_stub(texts)

    def _score_texts_stub(self, texts: List[str]) -> List[float]:
        """Score simulé pour les tests"""
        # Simulation basée sur des mots-clés
        positive_words = ["bull", "rise", "gain", "profit", "growth", "positive", "strong"]
        negative_words = ["bear", "fall", "loss", "decline", "weak", "negative", "crash"]

        scores = []
        for text in texts:
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)

            if pos_count > neg_count:
                score = 0.3 + (pos_count - neg_count) * 0.1
            elif neg_count > pos_count:
                score = -0.3 - (neg_count - pos_count) * 0.1
            else:
                score = 0.0

            # Ajouter un peu de bruit
            score += np.random.normal(0, 0.1)
            score = np.clip(score, -1, 1)
            scores.append(score)

        return scores


class SentimentAnalyzer:
    """Analyseur de sentiment avec agrégation temporelle"""

    def __init__(self, window_minutes: int = None):
        """Initialise l'analyseur de sentiment"""
        self.window_minutes = window_minutes or CONSTANTS.SENTIMENT_WINDOW
        self.finbert = FinBertAnalyzer()
        self.sentiment_data = {}  # {ticker: [(timestamp, sentiment, confidence)]}
        self.price_data = {}  # {ticker: [(timestamp, price)]}
        self.volume_data = {}  # {ticker: [(timestamp, volume)]}

        logger.info(f"💭 Analyseur de sentiment initialisé - Fenêtre: {self.window_minutes}min")

    def add_sentiment(self, ticker: str, sentiment: float, confidence: float = 1.0):
        """Ajoute un sentiment pour un ticker"""
        if ticker not in self.sentiment_data:
            self.sentiment_data[ticker] = []

        timestamp = datetime.now()
        self.sentiment_data[ticker].append((timestamp, sentiment, confidence))

        # Garder seulement les données récentes
        cutoff = timestamp - timedelta(hours=24)
        self.sentiment_data[ticker] = [
            (ts, sent, conf) for ts, sent, conf in self.sentiment_data[ticker] if ts > cutoff
        ]

        logger.debug(f"💭 Sentiment ajouté {ticker}: {sentiment:.3f}")

    def add_price(self, ticker: str, price: float):
        """Ajoute un prix pour un ticker"""
        if ticker not in self.price_data:
            self.price_data[ticker] = []

        timestamp = datetime.now()
        self.price_data[ticker].append((timestamp, price))

        # Garder seulement les données récentes
        cutoff = timestamp - timedelta(hours=24)
        self.price_data[ticker] = [(ts, p) for ts, p in self.price_data[ticker] if ts > cutoff]

    def add_volume(self, ticker: str, volume: float):
        """Ajoute un volume pour un ticker"""
        if ticker not in self.volume_data:
            self.volume_data[ticker] = []

        timestamp = datetime.now()
        self.volume_data[ticker].append((timestamp, volume))

        # Garder seulement les données récentes
        cutoff = timestamp - timedelta(hours=24)
        self.volume_data[ticker] = [(ts, v) for ts, v in self.volume_data[ticker] if ts > cutoff]

    def get_sentiment(self, ticker: str, window_minutes: int = None) -> float:
        """Récupère le sentiment agrégé pour un ticker"""
        window = window_minutes or self.window_minutes

        if ticker not in self.sentiment_data or not self.sentiment_data[ticker]:
            return 0.0

        # Filtrer par fenêtre temporelle
        cutoff = datetime.now() - timedelta(minutes=window)
        recent_sentiments = [sent for ts, sent, conf in self.sentiment_data[ticker] if ts > cutoff]

        if not recent_sentiments:
            return 0.0

        # Agrégation pondérée par confiance
        recent_data = [(sent, conf) for ts, sent, conf in self.sentiment_data[ticker] if ts > cutoff]

        if not recent_data:
            return 0.0

        sentiments, confidences = zip(*recent_data)
        weights = np.array(confidences)
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)

        weighted_sentiment = np.average(sentiments, weights=weights)
        return float(weighted_sentiment)

    def get_volatility(self, ticker: str, window_minutes: int = 20) -> float:
        """Calcule la volatilité des prix"""
        if ticker not in self.price_data or len(self.price_data[ticker]) < 2:
            return 0.0

        # Filtrer par fenêtre temporelle
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        recent_prices = [price for ts, price in self.price_data[ticker] if ts > cutoff]

        if len(recent_prices) < 2:
            return 0.0

        # Calculer les rendements
        prices = np.array(recent_prices)
        returns = np.diff(prices) / prices[:-1]

        # Volatilité annualisée
        volatility = np.std(returns) * np.sqrt(252)
        return float(volatility)

    def get_volume_ratio(self, ticker: str, current_volume: float, window_minutes: int = 20) -> float:
        """Calcule le ratio de volume"""
        if ticker not in self.volume_data or not self.volume_data[ticker]:
            return 1.0

        # Filtrer par fenêtre temporelle
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        recent_volumes = [volume for ts, volume in self.volume_data[ticker] if ts > cutoff]

        if not recent_volumes:
            return 1.0

        avg_volume = np.mean(recent_volumes)
        return float(current_volume / avg_volume) if avg_volume > 0 else 1.0

    def get_adaptive_sentiment(
        self, ticker: str, sentiment: float, volatility: float, volume_ratio: float
    ) -> Dict[str, Any]:
        """Calcule le sentiment adaptatif basé sur le contexte de marché"""
        # Sentiment de base
        base_sentiment = self.get_sentiment(ticker)

        # Ajustements basés sur la volatilité (utilise CONSTANTS)
        if volatility > CONSTANTS.VOLATILITY_HIGH_THRESHOLD:  # Haute volatilité
            # Réduire l'impact du sentiment en haute volatilité
            adjusted_sentiment = base_sentiment * 0.7
            confidence = 0.6
        elif volatility < CONSTANTS.VOLATILITY_LOW_THRESHOLD:  # Basse volatilité
            # Augmenter l'impact du sentiment en basse volatilité
            adjusted_sentiment = base_sentiment * 1.2
            confidence = 0.8
        else:  # Volatilité normale
            adjusted_sentiment = base_sentiment
            confidence = 0.7

        # Ajustements basés sur le volume (utilise CONSTANTS)
        if volume_ratio > CONSTANTS.VOLUME_RATIO_HIGH:  # Volume élevé
            # Volume élevé = plus de confiance dans le sentiment
            adjusted_sentiment *= 1.1
            confidence = min(0.9, confidence * 1.1)
        elif volume_ratio < CONSTANTS.VOLUME_RATIO_LOW:  # Volume faible
            # Volume faible = moins de confiance
            adjusted_sentiment *= 0.9
            confidence *= 0.8

        # Limiter les valeurs
        adjusted_sentiment = np.clip(adjusted_sentiment, -1, 1)
        confidence = np.clip(confidence, 0.1, 1.0)

        return {
            "base_sentiment": base_sentiment,
            "adjusted_sentiment": adjusted_sentiment,
            "confidence": confidence,
            "volatility": volatility,
            "volume_ratio": volume_ratio,
            "window_minutes": self.window_minutes,
        }

    def analyze_news_batch(self, news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyse un lot de news avec FinBERT"""
        if not news_items:
            return []

        # Extraire les textes
        texts = []
        for item in news_items:
            title = item.get("title", "")
            body = item.get("body", "")
            combined_text = f"{title} {body}".strip()
            texts.append(combined_text)

        # Analyser avec FinBERT
        sentiment_scores = self.finbert.score_texts(texts)

        # Créer les résultats
        results = []
        for item, score in zip(news_items, sentiment_scores):
            result = item.copy()
            result["sentiment_score"] = score
            result["sentiment_confidence"] = min(1.0, abs(score) + 0.3)
            results.append(result)

        logger.info(f"💭 {len(results)} articles analysés avec FinBERT")
        return results

    def get_sentiment_summary(self, ticker: str) -> Dict[str, Any]:
        """Retourne un résumé du sentiment pour un ticker"""
        if ticker not in self.sentiment_data:
            return {
                "ticker": ticker,
                "total_sentiments": 0,
                "avg_sentiment": 0.0,
                "latest_sentiment": 0.0,
                "confidence": 0.0,
            }

        sentiments = [sent for _, sent, _ in self.sentiment_data[ticker]]
        confidences = [conf for _, _, conf in self.sentiment_data[ticker]]

        return {
            "ticker": ticker,
            "total_sentiments": len(sentiments),
            "avg_sentiment": np.mean(sentiments) if sentiments else 0.0,
            "latest_sentiment": sentiments[-1] if sentiments else 0.0,
            "confidence": np.mean(confidences) if confidences else 0.0,
            "volatility": self.get_volatility(ticker),
            "window_minutes": self.window_minutes,
        }

    def reset(self):
        """Remet à zéro l'analyseur de sentiment"""
        self.sentiment_data = {}
        self.price_data = {}
        self.volume_data = {}
        logger.info("💭 Analyseur de sentiment réinitialisé")
