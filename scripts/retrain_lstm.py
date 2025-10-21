#!/usr/bin/env python3
"""
🔮 Réentraînement du modèle LSTM
Réentraîne le modèle LSTM avec les données yfinance à jour
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from loguru import logger

from src.constants import CONSTANTS
from src.core.prediction import PricePredictor


def retrain_lstm_model(ticker: str) -> bool:
    """
    Réentraîne le modèle LSTM pour un ticker

    Args:
        ticker: Symbole de l'action (SPY, NVDA, etc.)

    Returns:
        True si succès, False sinon
    """
    try:
        logger.info(f"🔮 Réentraînement LSTM pour {ticker}")

        # 1. Charger les données yfinance
        yfinance_path = CONSTANTS.DATA_ROOT / "historical" / "yfinance" / f"{ticker.upper()}_1999_2025.parquet"

        if not yfinance_path.exists():
            logger.error(f"❌ Fichier yfinance manquant: {yfinance_path}")
            return False

        logger.info(f"📊 Chargement données: {yfinance_path}")
        data = pd.read_parquet(yfinance_path)

        # Renommer 'date' en 'ts_utc' si nécessaire
        if "date" in data.columns:
            data = data.rename(columns={"date": "ts_utc"})

        data["ts_utc"] = pd.to_datetime(data["ts_utc"])
        data = data.sort_values("ts_utc")

        logger.info(f"   Lignes: {len(data)}")
        logger.info(f"   Période: {data['ts_utc'].min()} → {data['ts_utc'].max()}")

        # 2. Calculer les features RETURNS (OHLC)
        logger.info("🔄 Calcul des RETURNS...")

        # Uniformiser les noms de colonnes
        data = data.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
            }
        )

        # Calculer les RETURNS
        data["Open_RETURN"] = data["Open"].pct_change()
        data["High_RETURN"] = data["High"].pct_change()
        data["Low_RETURN"] = data["Low"].pct_change()
        data["TARGET"] = data["Close"].pct_change()

        # Supprimer les NaN
        data = data.dropna(subset=["Open_RETURN", "High_RETURN", "Low_RETURN", "TARGET"])

        # Garder seulement les colonnes RETURNS
        features_df = data[["Open_RETURN", "High_RETURN", "Low_RETURN", "TARGET"]].copy()

        # Normaliser les noms en MAJUSCULES
        features_df.columns = features_df.columns.str.upper()

        logger.info(f"   Features RETURNS: {len(features_df)} lignes")

        # 3. Réentraîner le modèle
        logger.info(f"🚀 Entraînement du modèle LSTM...")

        predictor = PricePredictor(ticker)
        results = predictor.train(features_df=features_df, epochs=CONSTANTS.LSTM_EPOCHS)

        if results:
            logger.info("✅ Modèle LSTM réentraîné avec succès")
            logger.info(f"   MAE: ${results.get('mae', 0):.2f}")
            logger.info(f"   RMSE: ${results.get('rmse', 0):.2f}")
            logger.info(f"   Accuracy: {results.get('accuracy', 0):.2%}")
            return True
        else:
            logger.error("❌ Échec de l'entraînement")
            return False

    except Exception as e:
        logger.error(f"❌ Erreur réentraînement {ticker}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def main():
    """Fonction principale"""
    logger.info("🚀 Démarrage réentraînement LSTM")

    success_count = 0
    total_tickers = len(CONSTANTS.TICKERS)

    for ticker in CONSTANTS.TICKERS:
        logger.info(f"\n{'='*50}")
        logger.info(f"📊 {ticker}")
        logger.info(f"{'='*50}")

        if retrain_lstm_model(ticker):
            success_count += 1

    # Résumé
    logger.info(f"\n{'='*50}")
    logger.info("🎯 RÉSUMÉ")
    logger.info(f"{'='*50}")
    logger.info(f"Modèles réentraînés: {success_count}/{total_tickers}")

    if success_count == total_tickers:
        logger.info("🎉 Tous les modèles sont à jour!")
        return 0
    elif success_count > 0:
        logger.warning(f"⚠️ {total_tickers - success_count} modèle(s) en échec")
        return 1
    else:
        logger.error("❌ Tous les réentraînements ont échoué")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
