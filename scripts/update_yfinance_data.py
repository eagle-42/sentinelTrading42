#!/usr/bin/env python3
"""
📊 Mise à jour des données yfinance
Télécharge les données historiques à jour pour SPY et NVDA
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yfinance as yf
import pandas as pd
from datetime import datetime
from loguru import logger

from src.constants import CONSTANTS


def update_yfinance_data(ticker: str) -> bool:
    """
    Télécharge et sauvegarde les données yfinance à jour

    Args:
        ticker: Symbole de l'action (SPY, NVDA, etc.)

    Returns:
        True si succès, False sinon
    """
    try:
        logger.info(f"📈 Téléchargement {ticker} depuis 1999-01-01...")

        # Télécharger les données
        stock = yf.Ticker(ticker)
        hist = stock.history(start="1999-01-01", interval="1d")

        if hist.empty:
            logger.error(f"❌ Aucune donnée pour {ticker}")
            return False

        # Nettoyer les données
        hist = hist.reset_index()
        hist["ticker"] = ticker
        hist = hist.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        hist = hist[["ticker", "date", "open", "high", "low", "close", "volume"]]
        hist = hist.dropna()

        # Créer le nom de fichier avec format année
        year_from = hist["date"].min().year
        year_to = hist["date"].max().year
        filename = f"{ticker}_{year_from}_{year_to}.parquet"

        # Sauvegarder
        yfinance_dir = CONSTANTS.DATA_ROOT / "historical" / "yfinance"
        yfinance_dir.mkdir(parents=True, exist_ok=True)

        filepath = yfinance_dir / filename
        hist.to_parquet(filepath, index=False)

        logger.info(f"✅ {ticker} sauvegardé: {filename}")
        logger.info(f"   Période: {year_from} à {year_to}")
        logger.info(f"   Jours: {len(hist)}")
        logger.info(f"   Dernier prix: ${hist['close'].iloc[-1]:.2f} ({hist['date'].iloc[-1].strftime('%Y-%m-%d')})")

        return True

    except Exception as e:
        logger.error(f"❌ Erreur {ticker}: {e}")
        return False


def main():
    """Fonction principale"""
    logger.info("🚀 Démarrage mise à jour yfinance")

    success_count = 0
    total_tickers = len(CONSTANTS.TICKERS)

    for ticker in CONSTANTS.TICKERS:
        logger.info(f"\n{'='*50}")
        logger.info(f"📊 {ticker}")
        logger.info(f"{'='*50}")

        if update_yfinance_data(ticker):
            success_count += 1

    # Résumé
    logger.info(f"\n{'='*50}")
    logger.info("🎯 RÉSUMÉ")
    logger.info(f"{'='*50}")
    logger.info(f"Succès: {success_count}/{total_tickers}")

    if success_count == total_tickers:
        logger.info("🎉 Toutes les données sont à jour!")
        return 0
    elif success_count > 0:
        logger.warning(f"⚠️ {total_tickers - success_count} ticker(s) en échec")
        return 1
    else:
        logger.error("❌ Tous les téléchargements ont échoué")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
