"""
Flows Sentinel42 pour déploiement Prefect 3.0
Appelle les scripts réels de collecte de données et trading
"""

import subprocess
import sys
from pathlib import Path
from prefect import flow, get_run_logger

# Chemin vers les scripts
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"

@flow(name="prix_15min")
def flow_prix_15min():
    """Flow de récupération des prix 15min via Finnhub"""
    logger = get_run_logger()
    logger.info("📊 Récupération des prix SPY (15min)")

    try:
        script_path = SCRIPTS_DIR / "finnhub_scraper.py"
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            logger.info("✅ Prix 15min récupérés avec succès")
            return {"status": "success", "data": "prix_15min"}
        else:
            logger.error(f"❌ Erreur récupération prix: {result.stderr}")
            return {"status": "error", "error": result.stderr}

    except Exception as e:
        logger.error(f"❌ Exception flow prix: {e}")
        return {"status": "error", "error": str(e)}

@flow(name="news_sentiment")
def flow_news_sentiment():
    """Flow de récupération des news et analyse de sentiment"""
    logger = get_run_logger()
    logger.info("📰 Récupération news + analyse sentiment")

    try:
        script_path = SCRIPTS_DIR / "refresh_news.py"
        if not script_path.exists():
            logger.warning("⚠️ Script refresh_news.py non trouvé")
            return {"status": "skipped", "reason": "script not found"}

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            logger.info("✅ News et sentiment analysés")
            return {"status": "success", "data": "news_sentiment"}
        else:
            logger.error(f"❌ Erreur news/sentiment: {result.stderr}")
            return {"status": "error", "error": result.stderr}

    except Exception as e:
        logger.error(f"❌ Exception flow news: {e}")
        return {"status": "error", "error": str(e)}

@flow(name="trading")
def flow_trading():
    """
    Flow de trading COMPLET : Prix 15min → News → Décision
    Exécution: Toutes les 15 minutes (heures marché)
    Ordre CRITIQUE pour la stratégie
    """
    logger = get_run_logger()
    logger.info("🚀 Démarrage Trading Flow (Prix → News → Décision)")

    # 1. Récupérer les prix 15min
    logger.info("1️⃣ Rafraîchissement prix 15min...")
    prix_result = flow_prix_15min()

    # 2. Récupérer les news + sentiment
    logger.info("2️⃣ Rafraîchissement news + sentiment...")
    news_result = flow_news_sentiment()

    # 3. Générer décision de trading
    logger.info("3️⃣ Génération décision de trading...")
    try:
        script_path = SCRIPTS_DIR / "trading_pipeline.py"
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            logger.info("✅ Trading Flow terminé avec succès")
            return {
                "status": "success",
                "prix": prix_result,
                "news": news_result,
                "decision": {"status": "success", "data": "trading"}
            }
        else:
            logger.warning(f"⚠️ Trading terminé avec code {result.returncode}")
            return {
                "status": "completed_with_warnings",
                "prix": prix_result,
                "news": news_result,
                "decision": {"returncode": result.returncode}
            }

    except Exception as e:
        logger.error(f"❌ Exception flow trading: {e}")
        return {
            "status": "error",
            "prix": prix_result,
            "news": news_result,
            "decision": {"error": str(e)}
        }

@flow(name="update_model")
def flow_update_model():
    """
    Flow de mise à jour du modèle LSTM
    1. Télécharge données yfinance à jour
    2. Réentraîne le modèle LSTM
    Exécution: Au démarrage de l'app
    """
    logger = get_run_logger()
    logger.info("🔄 Mise à jour modèle LSTM")

    # 1. Télécharger données yfinance
    logger.info("1️⃣ Téléchargement données yfinance...")
    try:
        script_path = SCRIPTS_DIR / "update_yfinance_data.py"
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            logger.info("✅ Données yfinance mises à jour")
            yfinance_result = {"status": "success"}
        else:
            logger.warning(f"⚠️ Données yfinance: code {result.returncode}")
            yfinance_result = {"status": "warning", "returncode": result.returncode}

    except Exception as e:
        logger.error(f"❌ Erreur téléchargement yfinance: {e}")
        yfinance_result = {"status": "error", "error": str(e)}

    # 2. Réentraîner modèle LSTM
    logger.info("2️⃣ Réentraînement modèle LSTM...")
    try:
        script_path = SCRIPTS_DIR / "retrain_lstm.py"
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes max pour l'entraînement
        )

        if result.returncode == 0:
            logger.info("✅ Modèle LSTM réentraîné")
            lstm_result = {"status": "success"}
        else:
            logger.warning(f"⚠️ Réentraînement LSTM: code {result.returncode}")
            lstm_result = {"status": "warning", "returncode": result.returncode}

    except Exception as e:
        logger.error(f"❌ Erreur réentraînement LSTM: {e}")
        lstm_result = {"status": "error", "error": str(e)}

    logger.info("✅ Mise à jour modèle terminée")
    return {
        "status": "success",
        "yfinance": yfinance_result,
        "lstm": lstm_result
    }


@flow(name="full_system")
def flow_full_system():
    """
    Flow système complet : Mise à jour modèle → Prix → News → Trading
    Exécution: Au démarrage et toutes les 2 heures
    """
    logger = get_run_logger()
    logger.info("🚀 Exécution système complet")

    # 0. Mise à jour modèle LSTM (yfinance + réentraînement)
    logger.info("0️⃣ Mise à jour modèle LSTM...")
    model_result = flow_update_model()

    # 1. Récupérer les prix
    prix_result = flow_prix_15min()

    # 2. Récupérer les news
    news_result = flow_news_sentiment()

    # 3. Générer décision de trading
    trading_result = flow_trading()

    logger.info("✅ Système complet exécuté")
    return {
        "status": "success",
        "model": model_result,
        "prix": prix_result,
        "news": news_result,
        "trading": trading_result
    }
