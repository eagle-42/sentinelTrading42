#!/usr/bin/env python3
"""
📰 Script de mise à jour des données de news et sentiment
Met à jour les news et calcule le sentiment toutes les 4 minutes
"""

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import feedparser
import pandas as pd
import requests
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.constants import CONSTANTS
from src.core.sentiment import FinBertAnalyzer, SentimentAnalyzer
from src.data import ParquetStorage


class NewsRefresher:
    """Gestionnaire de mise à jour des données de news et sentiment"""

    def __init__(self):
        self.storage = ParquetStorage()
        self.tickers = CONSTANTS.TICKER_NAMES
        self.newsapi_key = os.getenv("NEWSAPI_KEY")
        self.newsapi_enabled = os.getenv("NEWSAPI_ENABLED", "false").lower() == "true"
        self.finbert_mode = os.getenv("FINBERT_MODE", "stub")

        # Configuration des feeds RSS
        self.rss_feeds = [
            "https://www.cnbc.com/id/100003114/device/rss/rss.html",
            "https://feeds.bloomberg.com/markets/news.rss",
            "https://www.investing.com/rss/news.rss",
            "https://feeds.marketwatch.com/marketwatch/topstories/",
            "https://seekingalpha.com/api/sa/combined/RSS.xml",
        ]

        # Mots-clés pour la détection de tickers
        self.ticker_keywords = {
            "NVDA": ["nvidia", "nvdia", "gpu", "ai", "artificial intelligence", "cuda", "tensor"],
            "SPY": [
                "spy",
                "s&p",
                "s&p 500",
                "sp500",
                "sp-500",
                "s&p500",
                "s&p-500",
                "etf",
                "index",
                "market",
                "stock market",
                "equity market",
                "wall street",
                "dow jones",
                "nasdaq",
                "broad market",
                "market index",
                "us market",
                "american market",
                "stock index",
                "market benchmark",
                "market performance",
                "market sentiment",
                "market outlook",
                "market analysis",
                "market trends",
                "market volatility",
                "market rally",
                "market decline",
                "market correction",
                "bull market",
                "bear market",
                "market cap",
                "market capitalization",
                "sector performance",
                "market sector",
                "financial market",
                "securities market",
            ],
        }

        # Initialiser les analyseurs
        self.finbert = FinBertAnalyzer(mode=self.finbert_mode)
        self.sentiment_analyzer = SentimentAnalyzer()

    def get_rss_news(self) -> List[Dict[str, Any]]:
        """Récupère les news depuis les feeds RSS"""
        news_items = []

        for feed_url in self.rss_feeds:
            try:
                logger.info(f"📰 Récupération depuis {feed_url}")
                feed = feedparser.parse(feed_url)

                if feed.bozo:
                    logger.warning(f"⚠️ Feed malformé: {feed_url}")
                    continue

                for entry in feed.entries:
                    # Extraire les informations
                    title = getattr(entry, "title", "")
                    summary = getattr(entry, "summary", "")
                    link = getattr(entry, "link", "")
                    published = getattr(entry, "published", "")

                    # Détecter le ticker
                    ticker = self._detect_ticker(title + " " + summary)

                    if ticker:
                        news_items.append(
                            {
                                "title": title,
                                "summary": summary,
                                "body": summary,  # Pour compatibilité
                                "link": link,
                                "published": published,
                                "source": "RSS",
                                "ticker": ticker,
                                "ts_utc": datetime.now(timezone.utc),
                            }
                        )

                logger.info(f"✅ {len(feed.entries)} articles récupérés depuis {feed_url}")

            except Exception as e:
                logger.warning(f"⚠️ Erreur RSS {feed_url}: {e} (passé)")
                continue  # Continue avec les autres feeds

        return news_items

    def get_newsapi_news(self) -> List[Dict[str, Any]]:
        """Récupère les news depuis NewsAPI"""
        if not self.newsapi_enabled or not self.newsapi_key:
            logger.info("⚠️ NewsAPI désactivé ou clé manquante")
            return []

        news_items = []

        try:
            logger.info("📰 Récupération depuis NewsAPI")

            # Paramètres de recherche
            params = {
                "apiKey": self.newsapi_key,
                "q": "finance OR stock OR market OR trading",
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 50,
            }

            url = "https://newsapi.org/v2/everything"
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if data.get("status") != "ok":
                logger.warning(f"⚠️ NewsAPI status: {data.get('status')}")
                return []

            for article in data.get("articles", []):
                title = article.get("title", "")
                description = article.get("description", "")
                content = article.get("content", "")
                url = article.get("url", "")
                published = article.get("publishedAt", "")
                source = article.get("source", {}).get("name", "NewsAPI")

                # Détecter le ticker
                ticker = self._detect_ticker(title + " " + description + " " + content)

                if ticker:
                    news_items.append(
                        {
                            "title": title,
                            "summary": description,
                            "body": content or description,
                            "link": url,
                            "published": published,
                            "source": source,
                            "ticker": ticker,
                            "ts_utc": datetime.now(timezone.utc),
                        }
                    )

            logger.info(f"✅ {len(news_items)} articles récupérés depuis NewsAPI")

        except Exception as e:
            logger.error(f"❌ Erreur NewsAPI: {e}")

        return news_items

    def _detect_ticker(self, text: str) -> Optional[str]:
        """Détecte le ticker dans un texte"""
        text_lower = text.lower()

        for ticker, keywords in self.ticker_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    return ticker

        return None

    def score_sentiment(self, news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score le sentiment des articles de news"""
        if not news_items:
            return []

        logger.info(f"💭 Scoring du sentiment pour {len(news_items)} articles")

        # Préparer les textes pour le scoring
        texts = []
        for item in news_items:
            combined_text = f"{item['title']} {item['summary']}".strip()
            texts.append(combined_text)

        # Score le sentiment par lots
        batch_size = 32
        sentiment_scores = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_scores = self.finbert.score_texts(batch_texts)
            sentiment_scores.extend(batch_scores)

            logger.debug(f"Lot traité {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

        # Ajouter les scores aux articles
        for i, item in enumerate(news_items):
            item["sentiment_score"] = sentiment_scores[i] if i < len(sentiment_scores) else 0.0
            item["sentiment_confidence"] = 0.8  # Valeur par défaut

        logger.info(f"✅ Sentiment calculé pour {len(news_items)} articles")
        return news_items

    def aggregate_sentiment(self, news_items: List[Dict[str, Any]]) -> Dict[str, float]:
        """Agrège le sentiment par ticker"""
        sentiment_by_ticker = {}

        for ticker in self.tickers.keys():
            ticker_news = [item for item in news_items if item.get("ticker") == ticker]

            if ticker_news:
                # Calculer la moyenne pondérée du sentiment
                scores = [item["sentiment_score"] for item in ticker_news]
                weights = [item.get("sentiment_confidence", 0.8) for item in ticker_news]

                # Moyenne pondérée
                weighted_sentiment = sum(s * w for s, w in zip(scores, weights)) / sum(weights) if weights else 0.0

                sentiment_by_ticker[ticker] = {
                    "sentiment_score": weighted_sentiment,
                    "article_count": len(ticker_news),
                    "confidence": sum(weights) / len(weights) if weights else 0.0,
                }

                logger.info(f"📊 {ticker}: sentiment={weighted_sentiment:.3f}, articles={len(ticker_news)}")
            else:
                sentiment_by_ticker[ticker] = {"sentiment_score": 0.0, "article_count": 0, "confidence": 0.0}

        return sentiment_by_ticker

    def save_news_data(self, news_items: List[Dict[str, Any]]) -> Path:
        """Sauvegarde les données de news via storage"""
        if not news_items:
            logger.warning("⚠️ Aucune donnée de news à sauvegarder")
            return None

        df = pd.DataFrame(news_items)
        df['timestamp'] = df['ts_utc']  # Assurer colonne timestamp pour dedup
        return self.storage.save_data(df, data_type="news")

    def save_sentiment_data(self, sentiment_data: Dict[str, Any]) -> Path:
        """Sauvegarde les données de sentiment via storage"""
        sentiment_records = []
        for ticker, data in sentiment_data.items():
            sentiment_records.append({
                "ticker": ticker,
                "timestamp": datetime.now(timezone.utc),
                "sentiment_score": data["sentiment_score"],
                "confidence": data["confidence"],
                "article_count": data["article_count"],
            })

        df = pd.DataFrame(sentiment_records)
        return self.storage.save_data(df, data_type="sentiment", ticker="SPY")

    def refresh_news_and_sentiment(self) -> Dict[str, Any]:
        """Met à jour les news et le sentiment"""
        logger.info("📰 === MISE À JOUR DES NEWS ET SENTIMENT ===")
        start_time = datetime.now()

        # Récupérer les news
        rss_news = self.get_rss_news()
        newsapi_news = self.get_newsapi_news()

        all_news = rss_news + newsapi_news
        logger.info(f"📊 Total articles récupérés: {len(all_news)}")

        if not all_news:
            logger.warning("⚠️ Aucune news récupérée")
            return {
                "status": "no_news",
                "articles_processed": 0,
                "sentiment_data": {},
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
            }

        # Score le sentiment
        scored_news = self.score_sentiment(all_news)

        # Agrège le sentiment
        sentiment_data = self.aggregate_sentiment(scored_news)

        # Sauvegarder les données
        news_file = self.save_news_data(scored_news)
        sentiment_file = self.save_sentiment_data(sentiment_data)

        # Sauvegarder l'état
        state = {
            "last_update": datetime.now().isoformat(),
            "articles_processed": len(scored_news),
            "rss_articles": len(rss_news),
            "newsapi_articles": len(newsapi_news),
            "sentiment_data": sentiment_data,
            "files": {
                "news": str(news_file) if news_file else None,
                "sentiment": str(sentiment_file) if sentiment_file else None,
            },
            "duration_seconds": (datetime.now() - start_time).total_seconds(),
        }

        state_path = CONSTANTS.DATA_ROOT / "logs" / "news_refresh_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)

        import json

        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"\n📊 Résumé de la mise à jour:")
        logger.info(f"   Articles traités: {len(scored_news)}")
        logger.info(f"   RSS: {len(rss_news)}, NewsAPI: {len(newsapi_news)}")
        logger.info(f"   Durée: {(datetime.now() - start_time).total_seconds():.1f}s")
        logger.info(f"   État sauvé: {state_path}")

        return state


def main():
    """Fonction principale"""
    logger.info("🚀 Démarrage du refresh des news et sentiment")

    try:
        refresher = NewsRefresher()
        state = refresher.refresh_news_and_sentiment()

        logger.info("✅ Refresh des news et sentiment terminé avec succès")
        return 0

    except Exception as e:
        logger.error(f"❌ Erreur lors du refresh des news: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
