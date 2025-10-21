"""
Gestionnaire centralisé des services
Évite l'initialisation automatique des services
"""

from typing import Any, Dict, Optional

import streamlit as st


class ServiceManager:
    """Gestionnaire centralisé des services avec lazy loading"""

    def __init__(self):
        self._services = {}
        self._services_initialized = False

    def get_services(self) -> Dict[str, Any]:
        """Récupère tous les services (lazy loading)"""
        # Vérifier si les services doivent être actifs
        services_running = st.session_state.get("services_running", True)

        if services_running and not self._services_initialized:
            self._initialize_services()
        elif not services_running:
            # Services arrêtés - retourner des services vides
            self._services = {
                "data_service": None,
                "chart_service": None,
                "prediction_service": None,
                "sentiment_service": None,
                "fusion_service": None,
                "llm_service": None,
                "monitoring_service": None,
                "data_monitor_service": None,
                "decision_validation_service": None,
            }
            self._services_initialized = False

        return self._services

    def _initialize_services(self):
        """Initialise tous les services seulement si nécessaire"""
        try:
            # Import des services seulement quand nécessaire
            from src.gui.services.chart_service import ChartService
            from src.gui.services.data_monitor_service import DataMonitorService
            from src.gui.services.data_service import DataService
            from src.gui.services.decision_validation_service import DecisionValidationService
            from src.gui.services.fusion_service import FusionService
            from src.gui.services.llm_service import LLMService
            from src.gui.services.monitoring_service import MonitoringService
            from src.gui.services.prediction_service import PredictionService
            from src.gui.services.sentiment_service import SentimentService

            # Initialisation des services
            self._services = {
                "data_service": DataService(),
                "chart_service": ChartService(),
                "prediction_service": PredictionService(),
                "sentiment_service": SentimentService(),
                "fusion_service": FusionService(),
                "llm_service": LLMService(),
                "monitoring_service": MonitoringService(),
                "data_monitor_service": DataMonitorService(),
                "decision_validation_service": DecisionValidationService(),
            }
            self._services_initialized = True

        except Exception as e:
            st.error(f"❌ Erreur initialisation services: {e}")
            self._services = {
                "data_service": None,
                "chart_service": None,
                "prediction_service": None,
                "sentiment_service": None,
                "fusion_service": None,
                "llm_service": None,
                "monitoring_service": None,
                "data_monitor_service": None,
                "decision_validation_service": None,
            }

    def stop_all_services(self):
        """Arrête tous les services et libère la mémoire"""
        try:
            # Supprimer les services de la session
            if "production_services" in st.session_state:
                del st.session_state.production_services

            # Réinitialiser le gestionnaire
            self._services = {}
            self._services_initialized = False

            # Forcer le garbage collection
            import gc

            gc.collect()

            # Vider les caches Streamlit
            st.cache_data.clear()
            st.cache_resource.clear()

        except Exception as e:
            st.error(f"❌ Erreur lors de l'arrêt des services: {e}")

    def get_service(self, service_name: str) -> Optional[Any]:
        """Récupère un service spécifique"""
        services = self.get_services()
        return services.get(service_name)


# Instance globale du gestionnaire
service_manager = ServiceManager()
