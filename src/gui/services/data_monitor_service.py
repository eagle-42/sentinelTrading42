"""
Service de monitoring des données en temps réel
Surveille et met à jour automatiquement les données 15min
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class DataMonitorService:
    """Service de monitoring des données en temps réel"""

    def __init__(self):
        from src.constants import CONSTANTS
        self.data_path = CONSTANTS.PRICES_DIR
        self.cache = {}
        self.last_update = {}

    def get_latest_15min_data(self, ticker: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Récupère les dernières données 15min avec métadonnées"""
        try:
            file_path = self.data_path / f"{ticker.lower()}_15min.parquet"

            if not file_path.exists():
                return pd.DataFrame(), {"status": "no_file", "message": "Fichier 15min non trouvé"}

            # Charger les données
            data = pd.read_parquet(file_path)

            if data.empty:
                return pd.DataFrame(), {"status": "empty", "message": "Fichier 15min vide"}

            # Convertir les timestamps
            data["ts_utc"] = pd.to_datetime(data["ts_utc"])

            # Calculer les métadonnées
            latest_time = data["ts_utc"].iloc[-1]
            # Convertir en datetime naif pour la comparaison
            if latest_time.tzinfo is not None:
                latest_time_naive = latest_time.replace(tzinfo=None)
            else:
                latest_time_naive = latest_time
            time_since_update = datetime.now() - latest_time_naive

            metadata = {
                "status": "ok",
                "latest_time": latest_time,
                "time_since_update": time_since_update,
                "total_records": len(data),
                "is_fresh": time_since_update < timedelta(hours=1),  # Fraîche si < 1h
                "last_price": data["close"].iloc[-1],
                "price_change_24h": self._calculate_24h_change(data),
                "volume_avg": data["volume"].mean(),
            }

            return data, metadata

        except Exception as e:
            logger.error(f"Erreur monitoring 15min {ticker}: {e}")
            return pd.DataFrame(), {"status": "error", "message": str(e)}

    def _calculate_24h_change(self, data: pd.DataFrame) -> float:
        """Calcule la variation sur 24h (96 périodes de 15min)"""
        try:
            if len(data) > 96:
                current_price = data["close"].iloc[-1]
                price_24h_ago = data["close"].iloc[-97]  # 96 + 1 pour l'index
                return current_price - price_24h_ago
            return 0.0
        except:
            return 0.0

    def check_data_freshness(self, ticker: str) -> Dict[str, Any]:
        """Vérifie la fraîcheur des données"""
        data, metadata = self.get_latest_15min_data(ticker)

        if metadata["status"] != "ok":
            return {
                "is_fresh": False,
                "status": metadata["status"],
                "message": metadata["message"],
                "needs_update": True,
            }

        # Vérifier si les données sont fraîches
        is_fresh = metadata["is_fresh"]
        needs_update = not is_fresh

        return {
            "is_fresh": is_fresh,
            "status": "ok",
            "last_update": metadata["latest_time"],
            "time_since_update": metadata["time_since_update"],
            "needs_update": needs_update,
            "last_price": metadata["last_price"],
            "price_change_24h": metadata["price_change_24h"],
        }

    def get_data_summary(self, ticker: str) -> Dict[str, Any]:
        """Résumé des données pour l'interface"""
        data, metadata = self.get_latest_15min_data(ticker)

        if metadata["status"] != "ok":
            return {
                "available": False,
                "message": metadata["message"],
                "status_color": "red",
                "status_text": "❌ Données indisponibles",
            }

        # Déterminer le statut
        if metadata["is_fresh"]:
            status_color = "green"
            status_text = "✅ Données à jour"
        elif metadata["time_since_update"] < timedelta(hours=6):
            status_color = "orange"
            status_text = "⚠️ Données anciennes"
        else:
            status_color = "red"
            status_text = "❌ Données obsolètes"

        return {
            "available": True,
            "last_update": metadata["latest_time"],
            "time_since_update": metadata["time_since_update"],
            "last_price": metadata["last_price"],
            "price_change_24h": metadata["price_change_24h"],
            "volume_avg": metadata["volume_avg"],
            "total_records": metadata["total_records"],
            "status_color": status_color,
            "status_text": status_text,
            "needs_update": not metadata["is_fresh"],
        }

    def trigger_data_refresh(self, ticker: str) -> bool:
        """Déclenche une mise à jour des données en utilisant l'architecture existante"""
        try:
            # Utiliser le script simple de mise à jour
            import subprocess
            import sys

            # Chemin correct vers le script
            script_path = Path(__file__).parent.parent.parent.parent / "scripts" / "update_prices_simple.py"

            if not script_path.exists():
                logger.error(f"❌ Script non trouvé: {script_path}")
                return False

            # Exécuter le script de mise à jour
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent.parent,
            )

            success = result.returncode == 0

            if success:
                logger.info(f"✅ Données 15min mises à jour pour {ticker}")
                # Vider le cache pour forcer le rechargement
                self.cache.clear()
                return True
            else:
                logger.warning(f"⚠️ Échec mise à jour 15min pour {ticker}: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"❌ Erreur refresh 15min {ticker}: {e}")
            return False
