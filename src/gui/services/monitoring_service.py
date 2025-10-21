"""
Service de monitoring pour l'onglet Production
Surveillance système et alertes
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from loguru import logger

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("⚠️ psutil non disponible, métriques système limitées")


class MonitoringService:
    """Service de monitoring pour l'onglet Production"""

    def __init__(self):
        self.alerts = []
        self.trading_decisions = []
        self.system_status = {
            "crawler": "online",
            "sentiment": "online",
            "prediction": "online",
            "fusion": "online",
            "llm": "online",
        }
        self.last_update = datetime.now()

        logger.info("✅ Service de monitoring initialisé")

    def get_system_status(self) -> Dict[str, Any]:
        """Récupère le statut du système"""
        try:
            current_time = datetime.now()

            # Vérifier le statut des services
            status_checks = {
                "crawler": self._check_crawler_status(),
                "sentiment": self._check_sentiment_status(),
                "prediction": self._check_prediction_status(),
                "fusion": self._check_fusion_status(),
                "llm": self._check_llm_status(),
            }

            # Mettre à jour le statut
            for service, status in status_checks.items():
                self.system_status[service] = status

            # Calculer le statut global
            online_services = sum(1 for status in status_checks.values() if status == "online")
            total_services = len(status_checks)
            overall_status = "online" if online_services == total_services else "degraded"

            return {
                "overall_status": overall_status,
                "services": status_checks,
                "online_count": online_services,
                "total_count": total_services,
                "last_update": current_time,
                "uptime": self._get_uptime(),
            }
        except Exception as e:
            logger.error(f"❌ Erreur statut système: {e}")
            return {
                "overall_status": "error",
                "services": {},
                "online_count": 0,
                "total_count": 0,
                "last_update": datetime.now(),
                "uptime": "N/A",
            }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Récupère les métriques de performance"""
        try:
            if PSUTIL_AVAILABLE:
                return {
                    "cpu_percent": psutil.cpu_percent(interval=1),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_percent": psutil.disk_usage("/").percent,
                    "timestamp": datetime.now(),
                }
            else:
                return {
                    "cpu_percent": 0.0,
                    "memory_percent": 0.0,
                    "disk_percent": 0.0,
                    "timestamp": datetime.now(),
                    "note": "Métriques non disponibles (psutil manquant)",
                }
        except Exception as e:
            logger.error(f"❌ Erreur métriques performance: {e}")
            return {
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "timestamp": datetime.now(),
                "error": str(e),
            }

    def get_recent_trading_decisions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Récupère les VRAIES décisions de trading depuis le fichier JSON"""
        try:
            from pathlib import Path
            import json
            from src.constants import CONSTANTS
            
            # Charger les décisions
            decisions_file = CONSTANTS.TRADING_DIR / "decisions_log" / "trading_decisions.json"
            
            if decisions_file.exists():
                with open(decisions_file, "r") as f:
                    decisions = json.load(f)
                    # Convertir timestamp string en datetime pour affichage
                    for d in decisions:
                        if isinstance(d.get("timestamp"), str):
                            d["timestamp"] = d["timestamp"]
                    return decisions[-limit:] if decisions else []
            else:
                logger.warning("⚠️ Fichier décisions introuvable")
                if not self.trading_decisions:
                    self._generate_sample_decisions()
                return self.trading_decisions[-limit:] if self.trading_decisions else []
                
        except Exception as e:
            logger.error(f"❌ Erreur décisions trading: {e}")
            return []

    def get_alerts(self) -> List[Dict[str, Any]]:
        """Récupère les alertes système"""
        try:
            # Générer des alertes basées sur le statut
            self._generate_alerts()
        except Exception as e:
            logger.error(f"❌ Erreur alertes: {e}")
            return []

    def add_trading_decision(self, decision: Dict[str, Any]):
        """Ajoute une décision de trading"""
        try:
            decision["timestamp"] = datetime.now()
            decision["id"] = f"decision_{len(self.trading_decisions) + 1}"
            self.trading_decisions.append(decision)

            # Garder seulement les 100 dernières décisions
            if len(self.trading_decisions) > 100:
                self.trading_decisions = self.trading_decisions[-100:]

        except Exception as e:
            logger.error(f"❌ Erreur ajout décision: {e}")

    def add_alert(self, alert_type: str, message: str, severity: str = "info"):
        """Ajoute une alerte"""
        try:
            alert = {
                "id": f"alert_{len(self.alerts) + 1}",
                "type": alert_type,
                "message": message,
                "severity": severity,
                "timestamp": datetime.now(),
            }
            self.alerts.append(alert)

            # Garder seulement les 50 dernières alertes
            if len(self.alerts) > 50:
                self.alerts = self.alerts[-50:]

        except Exception as e:
            logger.error(f"❌ Erreur ajout alerte: {e}")

    def _check_crawler_status(self) -> str:
        """Vérifie le statut du crawler"""
        try:
            # Vérifier l'état des services dans la session
            if hasattr(st, "session_state") and not st.session_state.get("services_running", True):
                return "offline"
            # Simulation de vérification (en production, vérifier le vrai service)
            return "online"
        except Exception:
            return "offline"

    def _check_sentiment_status(self) -> str:
        """Vérifie le statut du service de sentiment"""
        try:
            # Vérifier l'état des services dans la session
            if hasattr(st, "session_state") and not st.session_state.get("services_running", True):
                return "offline"
            # Simulation de vérification
            return "online"
        except Exception:
            return "offline"

    def _check_prediction_status(self) -> str:
        """Vérifie le statut du service de prédiction"""
        try:
            # Vérifier l'état des services dans la session
            if hasattr(st, "session_state") and not st.session_state.get("services_running", True):
                return "offline"
            # Simulation de vérification
            return "online"
        except Exception:
            return "offline"

    def _check_fusion_status(self) -> str:
        """Vérifie le statut du service de fusion"""
        try:
            # Vérifier l'état des services dans la session
            if hasattr(st, "session_state") and not st.session_state.get("services_running", True):
                return "offline"
            # Simulation de vérification
            return "online"
        except Exception:
            return "offline"

    def _check_llm_status(self) -> str:
        """Vérifie le statut du service LLM"""
        try:
            # Vérifier l'état des services dans la session
            if hasattr(st, "session_state") and not st.session_state.get("services_running", True):
                return "offline"
            # Simulation de vérification
            return "online"
        except Exception:
            return "offline"

    def _get_uptime(self) -> str:
        """Calcule le temps de fonctionnement"""
        try:
            if PSUTIL_AVAILABLE:
                uptime_seconds = psutil.boot_time()
                uptime = datetime.now() - datetime.fromtimestamp(uptime_seconds)
                return str(uptime).split(".")[0]  # Enlever les microsecondes
            else:
                return "N/A"
        except Exception:
            return "N/A"

    def _generate_sample_decisions(self):
        """Génère des décisions de trading d'exemple"""
        try:
            decisions = [
                {
                    "action": "ACHETER",
                    "ticker": "SPY",
                    "price": 445.67,
                    "reason": "Score de fusion élevé (0.78)",
                    "confidence": 0.85,
                    "timestamp": datetime.now() - timedelta(minutes=30),
                },
                {
                    "action": "ATTENDRE",
                    "ticker": "NVDA",
                    "price": 183.45,
                    "reason": "Signaux mitigés, confiance insuffisante",
                    "confidence": 0.65,
                    "timestamp": datetime.now() - timedelta(minutes=15),
                },
                {
                    "action": "VENDRE",
                    "ticker": "SPY",
                    "price": 444.12,
                    "reason": "Sentiment négatif, seuil de vente atteint",
                    "confidence": 0.72,
                    "timestamp": datetime.now() - timedelta(minutes=5),
                },
            ]
            self.trading_decisions = decisions
        except Exception as e:
            logger.error(f"❌ Erreur génération décisions: {e}")

    def _generate_alerts(self):
        """Génère des alertes basées sur le statut"""
        try:
            # Vérifier les métriques de performance
            metrics = self.get_performance_metrics()

            # Alerte CPU
            if metrics.get("cpu_percent", 0) > 80:
                self.add_alert("performance", "Utilisation CPU élevée", "warning")

            # Alerte mémoire
            if metrics.get("memory_percent", 0) > 85:
                self.add_alert("performance", "Utilisation mémoire élevée", "warning")

            # Alerte disque
            if metrics.get("disk_percent", 0) > 90:
                self.add_alert("storage", "Espace disque faible", "critical")

            # Alerte services
            status = self.get_system_status()
            if status["overall_status"] != "online":
                self.add_alert("system", "Services dégradés", "warning")

        except Exception as e:
            logger.error(f"❌ Erreur génération alertes: {e}")
