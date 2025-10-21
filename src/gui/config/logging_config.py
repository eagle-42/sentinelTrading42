"""
Configuration centralisée des logs pour Sentinel42
Configure loguru pour écrire dans des fichiers
"""

import sys
from datetime import datetime
from pathlib import Path

from loguru import logger


def setup_logging():
    """Configure le système de logging centralisé"""

    # Créer le répertoire de logs s'il n'existe pas
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Supprimer le handler par défaut
    logger.remove()

    # Configuration du format des logs
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    # Handler pour la console (niveau INFO)
    logger.add(sys.stdout, format=log_format, level="INFO", colorize=True)

    # Handler pour le fichier principal (niveau DEBUG)
    logger.add(
        log_dir / "sentinel_main.log",
        format=log_format,
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        encoding="utf-8",
    )

    # Handler pour les erreurs uniquement
    logger.add(
        log_dir / "errors.log",
        format=log_format,
        level="ERROR",
        rotation="5 MB",
        retention="30 days",
        compression="zip",
        encoding="utf-8",
    )

    # Handler pour les décisions de trading
    logger.add(
        log_dir / "trading_decisions.log",
        format=log_format,
        level="INFO",
        filter=lambda record: "trading" in record["name"].lower() or "decision" in record["name"].lower(),
        rotation="1 MB",
        retention="30 days",
        encoding="utf-8",
    )

    # Log de démarrage
    logger.info("🚀 Configuration des logs initialisée")
    logger.info(f"📁 Répertoire de logs: {log_dir.absolute()}")
    logger.info(f"🕒 Démarrage: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return logger


def get_logger(name: str = None):
    """Récupère un logger configuré"""
    if name:
        return logger.bind(name=name)
    return logger
