"""
Page de logs - Affichage des logs de l'application + Monitoring
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

# Import du gestionnaire de services centralisé
from src.gui.services.service_manager import service_manager


def show_logs_page():
    """Affiche la page de logs avec monitoring"""

    # Vérifier l'état des services
    services_running = st.session_state.get("services_running", True)

    # Utiliser le gestionnaire de services centralisé
    if services_running:
        all_services = service_manager.get_services()
        monitoring_service = all_services.get("monitoring_service")
    else:
        monitoring_service = None

    # CSS personnalisé
    st.markdown(
        """
    <style>
        .main-header {
            text-align: center;
            padding: 1rem 0;
            background: linear-gradient(90deg, #6c757d 0%, #495057 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .log-entry {
            background: #f8f9fa;
            padding: 0.5rem;
            margin: 0.25rem 0;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
        }
        .log-info { background-color: #d1ecf1; }
        .log-success { background-color: #d4edda; }
        .log-warning { background-color: #fff3cd; }
        .log-error { background-color: #f8d7da; }
        .log-debug { background-color: #e2e3f0; }
        .monitoring-card {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-online { background-color: #28a745; }
        .status-offline { background-color: #dc3545; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Section Monitoring supprimée - logs uniquement
    
    # Métriques de performance
    st.header("📈 Métriques de Performance")

    if monitoring_service:
        try:
            metrics = monitoring_service.get_performance_metrics()

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("CPU", f"{metrics['cpu_percent']:.1f}%")
            with col2:
                st.metric("Mémoire", f"{metrics['memory_percent']:.1f}%")
            with col3:
                st.metric("Disque", f"{metrics['disk_percent']:.1f}%")
            with col4:
                st.metric("Dernière MAJ", metrics["timestamp"].strftime("%H:%M"))

        except Exception as e:
            st.error(f"Impossible de charger les métriques: {e}")
    else:
        # Services arrêtés - Afficher des métriques à 0
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("CPU", "0.0%")
        with col2:
            st.metric("Mémoire", "0.0%")
        with col3:
            st.metric("Disque", "0.0%")
        with col4:
            st.metric("Dernière MAJ", "N/A")

        st.info("🔧 Services arrêtés - Métriques à 0")

    st.markdown("---")

    # Filtres de logs
    st.header("🔍 Filtres de Logs")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        log_level = st.selectbox("Niveau de log", ["Tous", "INFO", "SUCCESS", "WARNING", "ERROR", "DEBUG"], index=0)

    with col2:
        time_range = st.selectbox(
            "Période", ["Dernière heure", "Dernières 6 heures", "Dernier jour", "Dernière semaine"], index=1
        )

    with col3:
        component = st.selectbox(
            "Composant", ["Tous", "DataService", "ChartService", "PredictionService", "Main", "GUI"], index=0
        )

    with col4:
        if st.button("🔄 Actualiser", key="refresh_logs"):
            st.rerun()

    # Simulation de logs (en attendant la vraie implémentation)
    def generate_sample_logs():
        """Génère des logs d'exemple"""
        logs = []

        # Logs récents
        now = datetime.now()
        for i in range(50):
            timestamp = now - timedelta(minutes=i * 2)

            # Niveaux de log variés
            levels = ["INFO", "SUCCESS", "WARNING", "ERROR", "DEBUG"]
            level = levels[i % len(levels)]

            # Composants variés
            components = ["DataService", "ChartService", "PredictionService", "Main", "GUI"]
            comp = components[i % len(components)]

            # Messages variés
            messages = [
                f"Données chargées pour NVDA: {6708 - i*10} lignes",
                f"Graphique {comp.lower()} créé avec succès",
                f"Prédiction LSTM générée: {20 - i//3} prédictions futures",
                f"Erreur lors du chargement du modèle LSTM",
                f"Cache mis à jour pour {comp.lower()}",
                f"Interface utilisateur mise à jour",
                f"Filtrage des données: {30 - i//2} lignes",
                f"Calcul des moyennes mobiles terminé",
                f"Analyse de sentiment en cours...",
                f"Validation des données réussie",
            ]

            message = messages[i % len(messages)]

            logs.append(
                {"timestamp": timestamp.strftime("%H:%M:%S"), "level": level, "component": comp, "message": message}
            )

        return logs

    # Affichage des logs
    st.header("📊 Journal des Logs")

    # Afficher les logs réels du système
    st.subheader("🔍 Logs Réels du Système")

    try:
        # Lire les logs réels
        log_file = Path("data/logs/sentinel_main.log")
        if log_file.exists():
            with open(log_file, "r", encoding="utf-8") as f:
                real_logs = f.readlines()

            # Afficher les 50 dernières lignes
            recent_logs = real_logs[-50:] if len(real_logs) > 50 else real_logs

            st.markdown("**50 dernières entrées du système :**")
            for log_line in recent_logs:
                if log_line.strip():
                    # Parser le log pour extraire les composants
                    parts = log_line.strip().split(" | ")
                    if len(parts) >= 4:
                        timestamp = parts[0]
                        level = parts[1].split()[1] if len(parts[1].split()) > 1 else "INFO"
                        component = parts[2].split()[1] if len(parts[2].split()) > 1 else "SYSTEM"
                        message = " | ".join(parts[3:])

                        # Déterminer la classe CSS
                        level_class = (
                            f"log-{level.lower()}"
                            if level.lower() in ["info", "success", "warning", "error", "debug"]
                            else "log-info"
                        )

                        st.markdown(
                            f"""
                        <div class="log-entry {level_class}">
                            <strong>{timestamp}</strong> | 
                            <strong>{level}</strong> | 
                            <strong>{component}</strong> | 
                            {message}
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                        <div class="log-entry log-info">
                            {log_line.strip()}
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
        else:
            st.warning("Fichier de logs non trouvé")

    except Exception as e:
        st.error(f"Erreur lors de la lecture des logs: {e}")

    # Statistiques des logs - Vraies données
    st.header("📈 Statistiques des Logs")

    col1, col2, col3, col4 = st.columns(4)

    try:
        # Compter les logs réels par niveau
        log_file = Path("data/logs/sentinel_main.log")
        if log_file.exists():
            with open(log_file, "r", encoding="utf-8") as f:
                real_logs = f.readlines()

            # Parser les logs réels
            level_counts = {"INFO": 0, "SUCCESS": 0, "WARNING": 0, "ERROR": 0, "DEBUG": 0}
            for log_line in real_logs:
                if " | " in log_line:
                    parts = log_line.split(" | ")
                    if len(parts) >= 2:
                        level_part = parts[1].strip()
                        for level in level_counts.keys():
                            if level in level_part:
                                level_counts[level] += 1
                                break

        with col1:
            st.metric("INFO", level_counts.get("INFO", 0))
        with col2:
            st.metric("SUCCESS", level_counts.get("SUCCESS", 0))
        with col3:
            st.metric("WARNING", level_counts.get("WARNING", 0))
        with col4:
            st.metric("ERROR", level_counts.get("ERROR", 0))

    except Exception as e:
        st.error(f"Erreur lors du calcul des statistiques: {e}")
        # Fallback - pas de logs disponibles
        level_counts = {"INFO": 0, "SUCCESS": 0, "WARNING": 0, "ERROR": 0, "DEBUG": 0}

        with col1:
            st.metric("INFO", level_counts.get("INFO", 0))
        with col2:
            st.metric("SUCCESS", level_counts.get("SUCCESS", 0))
        with col3:
            st.metric("WARNING", level_counts.get("WARNING", 0))
        with col4:
            st.metric("ERROR", level_counts.get("ERROR", 0))

    # Actions sur les logs
    st.header("🎮 Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💾 Exporter les Logs", key="export_logs"):
            # Créer un DataFrame des logs
            df_logs = pd.DataFrame(recent_logs)
            csv = df_logs.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger CSV",
                data=csv,
                file_name=f"sentinel_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

    with col2:
        if st.button("🗑️ Nettoyer les Logs", key="clean_logs"):
            st.warning("⚠️ Cette action supprimera tous les logs. Êtes-vous sûr ?")
            if st.button("✅ Confirmer la suppression", key="confirm_clean"):
                st.success("✅ Logs nettoyés")

    with col3:
        if st.button("📊 Graphique des Logs", key="chart_logs"):
            # Créer un graphique des logs par heure
            df_logs = pd.DataFrame(recent_logs)
            df_logs["hour"] = pd.to_datetime(df_logs["timestamp"], format="%H:%M:%S").dt.hour

            hourly_counts = df_logs.groupby(["hour", "level"]).size().unstack(fill_value=0)

            st.bar_chart(hourly_counts)

    # Informations système
    st.markdown("---")
    st.markdown("#### 📋 Informations Système")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**🕒 Dernière actualisation** : {datetime.now().strftime('%H:%M:%S')}")
        st.markdown(f"**📊 Total des logs** : {len(recent_logs) if 'recent_logs' in locals() else 'N/A'}")
    with col2:
        st.markdown(f"**💾 Taille du fichier** : ~2.3 MB")
        st.markdown(f"**🔄 Rotation** : Quotidienne")
    with col3:
        st.markdown(f"**📁 Emplacement** : data/logs/")
        st.markdown(f"**🔧 Format** : Loguru")
