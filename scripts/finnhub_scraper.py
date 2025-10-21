#!/usr/bin/env python3
"""
Finnhub API Scraper
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data import ParquetStorage

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")


class FinnhubScraper:
    """Scraper Finnhub API"""
    def __init__(self, api_key: str = None):
        self.api_key = api_key or FINNHUB_API_KEY
        if not self.api_key:
            raise ValueError("❌ FINNHUB_API_KEY manquante. Définir dans .env")
        self.base_url = "https://finnhub.io/api/v1"
        self.session = requests.Session()
    
    def get_quote(self, ticker: str) -> Optional[dict]:
        """Récupère le prix actuel"""
        try:
            url = f"{self.base_url}/quote"
            params = {
                'symbol': ticker,
                'token': self.api_key
            }
            
            logger.info(f"📊 Requête Finnhub pour {ticker}")
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Structure retournée:
            # {
            #   "c": 665.32,  # Current price
            #   "d": 2.1,     # Change
            #   "dp": 0.32,   # Percent change
            #   "h": 667.0,   # High
            #   "l": 663.5,   # Low
            #   "o": 664.0,   # Open
            #   "pc": 663.22, # Previous close
            #   "t": 1696176000  # Timestamp
            # }
            
            if 'c' in data and data['c'] > 0:
                logger.info(f"✅ Prix Finnhub: ${data['c']:.2f}")
                return data
            
            logger.warning(f"⚠️ Données invalides: {data}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Erreur Finnhub: {e}")
            return None


def refresh_prices_finnhub(ticker: str = "SPY") -> bool:
    """
    Récupère le prix depuis Finnhub et l'ajoute à l'historique
    """
    try:
        logger.info(f"📊 Récupération prix Finnhub pour {ticker}")
        
        scraper = FinnhubScraper()
        
        # Récupérer le quote
        quote = scraper.get_quote(ticker)
        
        if quote is None:
            logger.error(f"❌ Impossible de récupérer le prix pour {ticker}")
            return False
        
        current_price = quote['c']  # Current price
        
        # Créer une barre pour maintenant
        now = datetime.now(timezone.utc)
        # Arrondir à la fenêtre 15min la plus proche
        minutes = (now.minute // 15) * 15
        ts_rounded = now.replace(minute=minutes, second=0, microsecond=0)
        
        new_bar = pd.DataFrame([{
            'ts_utc': ts_rounded,
            'open': quote.get('o', current_price),    # Open
            'high': quote.get('h', current_price),    # High
            'low': quote.get('l', current_price),     # Low
            'close': current_price,                    # Current/Close
            'volume': 0,  # Finnhub ne donne pas le volume dans quote
            'ticker': ticker
        }])
        
        logger.info(f"📊 Nouvelle barre: {ts_rounded} | ${current_price:.2f}")
        logger.info(f"   Open: ${quote.get('o', 0):.2f} | High: ${quote.get('h', 0):.2f} | Low: ${quote.get('l', 0):.2f}")
        
        # Sauvegarder via storage (sauvegarde incrémentale)
        storage = ParquetStorage()
        file_path = storage.save_data(new_bar, data_type="prices", ticker=ticker, interval="15min")
        
        logger.info(f"✅ Prix sauvegardé via storage: {file_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur refresh Finnhub: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Point d'entrée du script"""
    logger.info("🚀 Finnhub API Scraper")
    logger.info("=" * 60)
    if FINNHUB_API_KEY:
        logger.info("API Key: configured (hidden)")
    else:
        logger.warning("API Key: not configured")
    logger.info(f"Limite: 60 appels/minute (FREE)")
    logger.info("=" * 60)
    
    success = refresh_prices_finnhub("SPY")
    
    if success:
        logger.info("\n✅ Récupération Finnhub terminée avec succès")
    else:
        logger.error("\n❌ Échec de la récupération Finnhub")
        sys.exit(1)


if __name__ == "__main__":
    main()
