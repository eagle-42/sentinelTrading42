#!/usr/bin/env python3
"""
Script simple de mise à jour des prix
Appelé par le bouton Streamlit "📈 Mettre à jour les prix"
"""
import sys
from pathlib import Path

# Ajouter le chemin src au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

# Import du scraper Finnhub existant
sys.path.insert(0, str(Path(__file__).parent))
from finnhub_scraper import refresh_prices_finnhub


def main():
    """Point d'entrée du script"""
    logger.info("🔄 Mise à jour des prix...")

    # Mettre à jour SPY
    success = refresh_prices_finnhub("SPY")

    if success:
        logger.info("✅ Prix mis à jour avec succès")
        return 0
    else:
        logger.error("❌ Échec de la mise à jour des prix")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
