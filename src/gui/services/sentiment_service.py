"""
Service de sentiment pour Streamlit
Analyse de sentiment avec FinBERT
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    from src.core.sentiment import FinBertAnalyzer, SentimentAnalyzer

    SENTIMENT_AVAILABLE = True
    logger.info("✅ SentimentAnalyzer importé avec succès")
except ImportError as e:
    logger.warning(f"⚠️ SentimentAnalyzer non disponible: {e}")
    SENTIMENT_AVAILABLE = False
    SentimentAnalyzer = None
    FinBertAnalyzer = None


class SentimentService:
    """Service de sentiment pour l'onglet Production"""

    def __init__(self):
        self.sentiment_analyzer = None
        self.finbert_analyzer = None

        if SENTIMENT_AVAILABLE:
            try:
                self.sentiment_analyzer = SentimentAnalyzer()
                self.finbert_analyzer = FinBertAnalyzer()
                logger.info("✅ Services de sentiment initialisés")
            except Exception as e:
                logger.warning(f"⚠️ Erreur initialisation sentiment: {e}")

    def get_news_articles(self, ticker: str = "SPY", limit: int = 10) -> List[Dict[str, Any]]:
        """Récupère les articles de news récents"""
        try:
            # Simulation d'articles (en production, récupérer depuis la base)
            articles = []
            for i in range(limit):
                articles.append(
                    {
                        "id": f"article_{i+1}",
                        "title": f"Article financier {i+1} sur {ticker}",
                        "source": "Financial News",
                        "timestamp": datetime.now() - timedelta(hours=i),
                        "content": f"Contenu de l'article {i+1} concernant {ticker}...",
                        "url": f"https://example.com/article_{i+1}",
                    }
                )
            return articles
        except Exception as e:
            logger.error(f"❌ Erreur récupération articles: {e}")
            return []

    def analyze_article_sentiment(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse le sentiment d'un article"""
        try:
            if not self.finbert_analyzer:
                # Fallback simulation
                sentiment_score = np.random.uniform(-1, 1)
                confidence = np.random.uniform(0.6, 0.9)
            else:
                # Analyse réelle avec FinBERT
                scores = self.finbert_analyzer.score_texts([article["content"]])
                sentiment_score = scores[0] if scores else 0.0
                confidence = 0.8  # Valeur par défaut

            # Déterminer l'emoji et la couleur
            if sentiment_score > 0.2:
                emoji = "😊"
                color = "green"
            elif sentiment_score < -0.2:
                emoji = "😡"
                color = "red"
            else:
                emoji = "😐"
                color = "orange"

            return {
                "sentiment_score": sentiment_score,
                "confidence": confidence,
                "emoji": emoji,
                "color": color,
                "label": self._get_sentiment_label(sentiment_score),
            }
        except Exception as e:
            logger.error(f"❌ Erreur analyse sentiment: {e}")
            return {"sentiment_score": 0.0, "confidence": 0.5, "emoji": "😐", "color": "gray", "label": "Neutre"}

    def get_sentiment_summary(self, ticker: str = "SPY") -> Dict[str, Any]:
        """Récupère le résumé de sentiment global"""
        try:
            articles = self.get_news_articles(ticker, 10)
            sentiments = []

            for article in articles:
                sentiment = self.analyze_article_sentiment(article)
                sentiments.append(sentiment["sentiment_score"])

            if sentiments:
                avg_sentiment = np.mean(sentiments)
                total_articles = len(articles)
                positive_count = sum(1 for s in sentiments if s > 0.2)
                negative_count = sum(1 for s in sentiments if s < -0.2)
                neutral_count = total_articles - positive_count - negative_count
            else:
                avg_sentiment = 0.0
                total_articles = 0
                positive_count = negative_count = neutral_count = 0

            return {
                "avg_sentiment": avg_sentiment,
                "total_articles": total_articles,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "neutral_count": neutral_count,
                "sentiment_trend": self._get_sentiment_trend(avg_sentiment),
            }
        except Exception as e:
            logger.error(f"❌ Erreur résumé sentiment: {e}")
            return {
                "avg_sentiment": 0.0,
                "total_articles": 0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "sentiment_trend": "stable",
            }

    def get_sentiment_score(self, ticker: str = "SPY") -> float:
        """Récupère le score de sentiment moyen pour un ticker"""
        try:
            summary = self.get_sentiment_summary(ticker)
            return summary.get("avg_sentiment", 0.0)
        except Exception as e:
            logger.error(f"❌ Erreur score sentiment: {e}")
            return 0.0

    def get_keywords(self, ticker: str = "SPY") -> List[Dict[str, Any]]:
        """Récupère les mots-clés impactants"""
        try:
            # Simulation de mots-clés (en production, extraire des articles)
            keywords = [
                {"word": "earnings", "impact": 0.8, "count": 15},
                {"word": "growth", "impact": 0.7, "count": 12},
                {"word": "volatility", "impact": -0.6, "count": 8},
                {"word": "bullish", "impact": 0.9, "count": 6},
                {"word": "recession", "impact": -0.8, "count": 4},
            ]
            return keywords
        except Exception as e:
            logger.error(f"❌ Erreur mots-clés: {e}")
            return []

    def _get_sentiment_label(self, score: float) -> str:
        """Convertit le score en label"""
        if score > 0.5:
            return "Très Positif"
        elif score > 0.2:
            return "Positif"
        elif score > -0.2:
            return "Neutre"
        elif score > -0.5:
            return "Négatif"
        else:
            return "Très Négatif"

    def _get_sentiment_trend(self, score: float) -> str:
        """Détermine la tendance du sentiment"""
        if score > 0.3:
            return "haussière"
        elif score < -0.3:
            return "baissière"
        else:
            return "stable"
