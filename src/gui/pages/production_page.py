"""
Page de production - Dashboard Trading
Interface optimisée selon les spécifications utilisateur
"""

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from loguru import logger

from src.constants import CONSTANTS
from src.gui.services.chart_service import ChartService
from src.gui.services.data_monitor_service import DataMonitorService
from src.gui.services.data_service import DataService
from src.gui.services.decision_validation_service import DecisionValidationService
from src.gui.services.fusion_service import FusionService
from src.gui.services.llm_service import LLMService
from src.gui.services.monitoring_service import MonitoringService
from src.gui.services.prediction_service import PredictionService
from src.gui.services.sentiment_service import SentimentService


def show_production_page():
    """Affiche la page de production optimisée"""

    # Initialiser les services
    if "production_services" not in st.session_state:
        st.session_state.production_services = {
            "data_service": DataService(),
            "chart_service": ChartService(),
            "prediction_service": PredictionService(),
            "sentiment_service": SentimentService(),
            "fusion_service": FusionService(),
            "llm_service": LLMService(),
            "monitoring_service": MonitoringService(),
            "data_monitor_service": DataMonitorService(),
        }

    # Cache pour les prédictions LSTM (éviter de recalculer à chaque fois)
    if "prediction_cache" not in st.session_state:
        st.session_state.prediction_cache = {}

    services = st.session_state.production_services

    # CSS personnalisé optimisé
    st.markdown(
        """
    <style>
        .feature-card {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .feature-card:hover {
            transform: translateY(-2px);
        }
        .price-card {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        }
        .sentiment-card {
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        }
        .fusion-card {
            background: linear-gradient(135deg, #e2e3f0 0%, #d1d5f0 100%);
        }
        .llm-card {
            background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        }
        .monitoring-card {
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
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
        .status-warning { background-color: #ffc107; }
        .gauge-container {
            text-align: center;
            padding: 1rem;
        }
        .recommendation-badge {
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 25px;
            font-weight: bold;
            font-size: 1.1rem;
        }
        .recommendation-buy {
            background-color: #28a745;
            color: white;
        }
        .recommendation-sell {
            background-color: #dc3545;
            color: white;
        }
        .recommendation-wait {
            background-color: #ffc107;
            color: #212529;
        }
        .action-buttons {
            display: flex;
            gap: 10px;
            margin-left: 20px;
        }
        .error-alert {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .market-status {
            text-align: center;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .market-open {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .market-closed {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .service-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            margin: 0.5rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .service-card h6 {
            margin: 0 0 0.5rem 0;
            font-size: 0.9rem;
            color: #666;
        }
        .service-card p {
            margin: 0.25rem 0;
            font-size: 1rem;
        }
        .service-card small {
            color: #888;
            font-size: 0.8rem;
        }
        .status-line {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            font-size: 1.1rem;
            text-align: center;
        }
        .timeline-container {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .timeline-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0;
            border-bottom: 1px solid #e9ecef;
        }
        .timeline-item:last-child {
            border-bottom: none;
        }
        .kpi-box {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            margin: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .kpi-value {
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
        }
        .kpi-label {
            font-size: 0.9rem;
            color: #666;
            margin-top: 0.5rem;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar - Nettoyée selon demande
    with st.sidebar:
        st.header("⚙️ Configuration Production")

        # Configuration simple
        ticker = st.selectbox(
            "Ticker",
            ["SPY", "NVDA"],
            index=0,
            help="Symbole de l'action à analyser. SPY: Prédictions disponibles. NVDA: Analyse uniquement.",
        )

        # Période fixe à 7 jours pour l'analyse
        period = "7 derniers jours"

        # Configuration des seuils adaptatifs
        st.subheader("🎯 Seuils Adaptatifs")

        # Afficher les seuils actuels
        fusion_stats = services["fusion_service"].get_fusion_stats()
        current_thresholds = fusion_stats.get("current_thresholds", {"buy": 0.1, "sell": -0.1})

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Seuil BUY", f"{current_thresholds.get('buy', 0.1):.3f}", help="Signal > seuil → ACHETER")
        with col2:
            st.metric("Seuil SELL", f"{current_thresholds.get('sell', -0.1):.3f}", help="Signal < seuil → VENDRE")

        # Information sur l'adaptation
        st.info("🔄 Les seuils s'adaptent automatiquement selon la volatilité du marché")

        # Actions du système
        st.subheader("🎮 Actions")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button(
                "🚀 Démarrer Services",
                type="primary",
                key="start_services",
                help="Démarre tous les services de trading (crawler, prédiction, sentiment, fusion)",
            ):
                st.success("✅ Services démarrés")
                st.session_state.services_running = True
        with col2:
            if st.button(
                "🛑 Arrêter Services",
                type="secondary",
                key="stop_services",
                help="Arrête tous les services de trading (simulation)",
            ):
                st.warning("⚠️ Services arrêtés")
                st.session_state.services_running = False
        with col3:
            if st.button(
                "🔄 Rafraîchir",
                type="secondary",
                key="refresh_page",
                help="Recharge la page et actualise tous les affichages",
            ):
                # Vider le cache de prédictions
                st.session_state.prediction_cache = {}
                st.rerun()

        # Mise à jour des données
        st.subheader("📊 Données")

        if st.button(
            "📈 Mettre à jour les prix",
            type="secondary",
            key="update_prices",
            help="Met à jour les données de prix 15min depuis l'API et recharge les graphiques",
        ):
            with st.spinner("Mise à jour des données de prix..."):
                try:
                    # Utiliser le service de monitoring pour la mise à jour
                    success = services["data_monitor_service"].trigger_data_refresh(ticker)

                    if success:
                        # Vider tous les caches Streamlit pour forcer le rechargement
                        st.cache_data.clear()
                        st.cache_resource.clear()
                        st.success("✅ Données de prix mises à jour")
                        st.rerun()  # Recharger pour voir les nouvelles données
                    else:
                        st.warning("⚠️ Mise à jour partielle - Vérifiez les logs")

                except Exception as e:
                    st.error(f"❌ Erreur lors de la mise à jour: {e}")
                    st.info("ℹ️ Utilisation des données en cache")

        # Section d'aide
        st.subheader("❓ Aide")
        with st.expander("💡 Que font ces boutons ?", expanded=False):
            st.markdown(
                """
            **🎮 Actions :**
            - **🚀 Démarrer Services** : Lance tous les services de trading (crawler, prédiction, sentiment, fusion)
            - **🛑 Arrêter Services** : Arrête tous les services (simulation pour l'interface)
            - **🔄 Rafraîchir** : Recharge la page et actualise tous les affichages
            
            **📊 Données :**
            - **📈 Mettre à jour les prix** : Met à jour les données de prix 15min depuis l'API et recharge les graphiques
            """
            )

    # 1. COMPTE RENDU D'ACTIVITÉ - Version restructurée en haut
    st.header("📊 Compte Rendu d'Activité")

    # Vérification de l'état du marché
    market_status = _check_market_status()

    # Vérification des dépendances critiques
    fusion_available, fusion_errors = _check_fusion_dependencies(services, ticker, market_status)

    # Affichage des erreurs centralisées - Un seul message par module
    if fusion_errors:
        critical_errors = [e for e in fusion_errors if e["severity"] == "critical"]
        if critical_errors:
            st.error(f"🚨 **{critical_errors[0]['title']}** – {critical_errors[0]['message']}")

    # Calculer fusion_data avant l'affichage des KPIs
    fusion_data = None
    if fusion_available:
        try:
            price_signal = _get_price_signal(services, ticker)
            sentiment_signal = _get_sentiment_signal(services, ticker)
            prediction_signal = _get_prediction_signal(services, ticker)

            fusion_data = services["fusion_service"].calculate_fusion_score(
                price_signal, sentiment_signal, prediction_signal
            )
        except Exception as e:
            st.error(f"Erreur calcul fusion: {e}")
            fusion_data = None

    # KPI principaux en haut - Ordre réorganisé
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        # 1. Statut marché
        market_color = "#28a745" if market_status["is_open"] else "#dc3545"
        market_text = "Ouvert" if market_status["is_open"] else "Fermé"
        st.markdown(
            f"""
        <div class="kpi-box">
            <div class="kpi-value" style="color: {market_color};">{market_text}</div>
            <div class="kpi-label">Marché</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        # 2. Dernière MAJ + Prochain Update (fenêtres fixes de 15 minutes)
        current_time = datetime.now().strftime("%H:%M:%S")
        next_decision_time = _get_next_decision_time()
        st.markdown(
            f"""
        <div class="kpi-box">
            <div class="kpi-value" style="color: #667eea;">{current_time}</div>
            <div class="kpi-label">Dernière MAJ</div>
            <div class="kpi-label" style="font-size: 0.8rem; color: #888; margin-top: 0.5rem;">Prochain: {next_decision_time}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        # 3. Confiance avec tooltip
        if fusion_available and fusion_data:
            try:
                st.markdown(
                    f"""
                <div class="kpi-box" title="Niveau de confiance basé sur la cohérence des signaux (prix, sentiment, prédiction)">
                    <div class="kpi-value">{fusion_data['confidence']:.1%}</div>
                    <div class="kpi-label">Confiance</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            except:
                st.markdown(
                    f"""
                <div class="kpi-box">
                    <div class="kpi-value" style="color: #dc3545;">N/A</div>
                    <div class="kpi-label">Confiance</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f"""
            <div class="kpi-box">
                <div class="kpi-value" style="color: #dc3545;">N/A</div>
                <div class="kpi-label">Confiance</div>
        </div>
        """,
                unsafe_allow_html=True,
            )

    with col4:
        # 4. Score de fusion avec détails
        if fusion_available and fusion_data:
            try:
                st.markdown(
                    f"""
                <div class="kpi-box">
                    <div class="kpi-value">{fusion_data['fusion_score']:.2f}</div>
                    <div class="kpi-label">Score Fusion</div>
                    <div class="kpi-label" style="font-size: 0.7rem; color: #888; margin-top: 0.3rem;">
                        Prix: {price_signal:.2f} | News: {sentiment_signal:.2f}
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.markdown(
                    f"""
                <div class="kpi-box">
                    <div class="kpi-value" style="color: #dc3545;">N/A</div>
                    <div class="kpi-label">Score Fusion</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f"""
            <div class="kpi-box">
                <div class="kpi-value" style="color: #dc3545;">N/A</div>
                <div class="kpi-label">Score Fusion</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col5:
        # 5. Recommandation - Afficher le dernier résultat ou statut approprié
        if fusion_available and market_status["is_open"] and ticker == "SPY" and fusion_data:
            try:
                st.markdown(
                    f"""
                <div class="kpi-box">
                    <div class="kpi-value" style="color: {fusion_data['color']};">{fusion_data['recommendation']}</div>
                    <div class="kpi-label">Recommandation</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            except:
                st.markdown(
                    f"""
                <div class="kpi-box">
                    <div class="kpi-value" style="color: #ffc107;">En cours</div>
                    <div class="kpi-label">Recommandation</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
        elif ticker == "NVDA":
            st.markdown(
                f"""
            <div class="kpi-box">
                <div class="kpi-value" style="color: #ffc107;">Analyse</div>
                <div class="kpi-label">Recommandation</div>
                <div class="kpi-label" style="font-size: 0.8rem; color: #888;">NVDA: Analyse uniquement</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
            <div class="kpi-box">
                <div class="kpi-value" style="color: #ffc107;">En attente</div>
                <div class="kpi-label">Recommandation</div>
                <div class="kpi-label" style="font-size: 0.8rem; color: #888;">Ouverture marché</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # 3. GRAPHIQUE 15MIN DU JOUR - Données réelles avec monitoring
    st.header("📈 Graphique 15min du Jour")

    # Vérifier l'état des données 15min
    data_summary = services["data_monitor_service"].get_data_summary(ticker)

    # Afficher le graphique si les données sont disponibles
    if data_summary.get("available", False):
        try:
            data_15min, metadata = services["data_monitor_service"].get_latest_15min_data(ticker)

            if not data_15min.empty:
                # Filtrer les données des 7 derniers jours
                seven_days_ago = datetime.now() - timedelta(days=7)
                # Créer une copie pour éviter le warning SettingWithCopyWarning
                data_15min_copy = data_15min.copy()
                # Convertir les timestamps en datetime naif pour la comparaison
                data_15min_copy = data_15min_copy.copy()
                data_15min_copy["ts_utc_naive"] = data_15min_copy["ts_utc"].apply(
                    lambda x: x.replace(tzinfo=None) if x.tzinfo is not None else x
                )
                data_recent = data_15min_copy[data_15min_copy["ts_utc_naive"] >= seven_days_ago].copy()

                if not data_recent.empty:
                    # Convertir UTC → Paris pour l'affichage
                    import pytz
                    paris_tz = pytz.timezone('Europe/Paris')
                    data_recent["ts_paris"] = data_recent["ts_utc"].dt.tz_convert(paris_tz)

                    # Créer le graphique 15min
                    fig = go.Figure()

                    fig.add_trace(
                        go.Scatter(
                            x=data_recent["ts_paris"],  # Utiliser heure de Paris
                            y=data_recent["close"],
                            mode="lines+markers",
                            name=f"{ticker} - Prix 15min",
                            line=dict(color="#1f77b4", width=2),
                            marker=dict(size=4),
                            hovertemplate="<b>%{fullData.name}</b><br>Heure (Paris): %{x}<br>Prix: $%{y:.2f}<extra></extra>",
                        )
                    )

                    # Ajouter les moyennes mobiles
                    if len(data_recent) > 20:
                        data_recent["ma_20"] = data_recent["close"].rolling(window=20).mean()
                        fig.add_trace(
                            go.Scatter(
                                x=data_recent["ts_paris"],  # Utiliser heure de Paris
                                y=data_recent["ma_20"],
                                mode="lines",
                                name="MA 20",
                                line=dict(color="#ff7f0e", width=1, dash="dash"),
                                hovertemplate="<b>MA 20</b><br>Heure (Paris): %{x}<br>Prix: $%{y:.2f}<extra></extra>",
                            )
                        )

                    fig.update_layout(
                        title=f"Prix {ticker} - 15min (7 derniers jours) - Heure de Paris",
                        xaxis_title="Heure (Paris)",
                        yaxis_title="Prix ($)",
                        showlegend=True,
                        hovermode="x unified",
                    )

                    st.plotly_chart(fig, config={"displayModeBar": False})

                    # Afficher les statistiques
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Dernier Prix", f"${data_recent['close'].iloc[-1]:.2f}")
                    with col2:
                        price_change = data_summary.get("price_change_24h", 0)
                        st.metric("Variation 24h", f"{price_change:+.2f}$")
                    with col3:
                        st.metric("Volume Moyen", f"{data_summary.get('volume_avg', 0):.0f}")
                    with col4:
                        st.metric("Total Records", f"{data_summary.get('total_records', 0)}")
                else:
                    st.info("Aucune donnée 15min récente disponible")
            else:
                st.warning("Données 15min vides")
        except Exception as e:
            st.error(f"Erreur graphique 15min: {e}")
    else:
        st.warning(f"Données 15min non disponibles: {data_summary.get('message', 'Erreur inconnue')}")

    # Layout en 2 colonnes centrées : Analyse sentiment, Service LLM
    col1, col2 = st.columns(2, gap="large")

    with col1:
        # Analyse de Sentiment
        st.subheader("💭 Analyse de Sentiment")

        # Analyse de sentiment simple
        try:
            articles = services["sentiment_service"].get_news_articles(ticker, 10)

            if articles:
                # Trier les articles par date (plus récent en premier)
                articles = sorted(articles, key=lambda x: x["timestamp"], reverse=True)

                # Afficher le dernier article analysé
                dernier_article = articles[0]
                sentiment = services["sentiment_service"].analyze_article_sentiment(dernier_article)

                # Justification du sentiment
                justification_elements = []
                if sentiment["sentiment_score"] > 0.1:
                    justification_elements.append("📈 Tendance positive")
                elif sentiment["sentiment_score"] < -0.1:
                    justification_elements.append("📉 Tendance négative")
                else:
                    justification_elements.append("📊 Tendance neutre")

                if len(articles) > 5:
                    justification_elements.append("📰 Volume d'articles élevé")
                elif len(articles) > 2:
                    justification_elements.append("📰 Volume d'articles modéré")
                else:
                    justification_elements.append("📰 Volume d'articles faible")

                # Affichage simple
                st.markdown(
                    f"**Dernier sentiment :** {sentiment['label']} ({sentiment['sentiment_score']:.2f}) • {dernier_article['timestamp'].strftime('%H:%M')}"
                )
                st.markdown(f"**Justification :** {' | '.join(justification_elements)}")

                # Liste des articles
                st.markdown("**📰 Liste des articles :**")
                for i, article in enumerate(articles[:5], 1):  # Limiter à 5 articles
                    try:
                        article_sentiment = services["sentiment_service"].analyze_article_sentiment(article)
                        with st.expander(f"Article {i}: {article['title'][:50]}...", expanded=False):
                            st.write(f"**Source:** {article['source']} • {article['timestamp'].strftime('%H:%M')}")
                            st.write(
                                f"**Sentiment:** {article_sentiment['emoji']} {article_sentiment['label']} ({article_sentiment['sentiment_score']:.2f})"
                            )
                    except Exception as e:
                        with st.expander(f"Article {i}: {article['title'][:50]}...", expanded=False):
                            st.error(f"❌ Erreur d'analyse: {str(e)}")
            else:
                st.info("Aucun article disponible")

        except Exception as e:
            st.error(f"Erreur Analyse Sentiment: {str(e)}")

    with col2:
        # Service LLM
        st.subheader("🧠 Service LLM")

        # Service LLM simple
        try:
            llm_service = LLMService()
            llm_status = llm_service.check_service_status()

            # Affichage du statut
            if llm_status["online"] and llm_status["model_available"]:
                st.success(f"✅ {llm_status['status']}")

                # Génération automatique de synthèse si fusion disponible
                if fusion_available and market_status["is_open"] and ticker == "SPY" and fusion_data:
                    try:
                        current_price = services["data_service"].load_data(ticker)["CLOSE"].iloc[-1]
                        sentiment_score = services["sentiment_service"].get_sentiment_score(ticker)

                        synthesis = llm_service.generate_trading_synthesis(
                            ticker,
                            fusion_data["recommendation"],
                            fusion_data.get("fusion_score", 0.0),
                            current_price,
                            sentiment_score,
                        )

                        if synthesis["success"]:
                            st.session_state["llm_synthesis"] = synthesis
                            llm_service.save_synthesis(ticker, synthesis)

                            with st.expander("📝 Analyse LLM Automatique", expanded=True):
                                st.write(synthesis["synthesis"])
                                st.caption(f"Modèle: {synthesis['model']} • {synthesis['timestamp'][:19]}")
                        else:
                            st.warning(f"⚠️ {synthesis['synthesis']}")

                    except Exception as e:
                        if "timeout" in str(e).lower():
                            st.warning("⚠️ Service LLM occupé - Réessayez plus tard")
                        else:
                            st.warning(f"⚠️ Erreur génération: {str(e)[:50]}...")
                else:
                    st.info("ℹ️ Synthèse LLM disponible uniquement pour SPY en marché ouvert")

                # Afficher l'historique des synthèses
                try:
                    from src.constants import CONSTANTS

                    synthesis_path = CONSTANTS.get_data_path() / "trading" / "llm_synthesis"
                    synthesis_files = list(synthesis_path.glob(f"{ticker}_synthesis_*.json"))

                    if synthesis_files:
                        # Prendre le fichier le plus récent
                        latest_file = max(synthesis_files, key=lambda x: x.stat().st_mtime)

                        with open(latest_file, "r") as f:
                            syntheses = json.load(f)

                        if syntheses:
                            st.markdown("**📚 Historique des Synthèses:**")
                            for i, synthesis in enumerate(syntheses[-3:], 1):
                                with st.expander(f"Synthèse #{len(syntheses) - 3 + i}", expanded=False):
                                    st.write(synthesis["synthesis"])
                                    st.caption(f"{synthesis['timestamp'][:19]} • {synthesis['tokens_used']} mots")
                        else:
                            st.info("Aucune synthèse disponible dans l'historique")
                    else:
                        st.info("Aucun historique de synthèses trouvé")

                except Exception as e:
                    st.warning(f"⚠️ Erreur chargement historique: {str(e)}")

            else:
                st.error(f"❌ {llm_status['status']}")
                st.info("💡 Pour activer le service LLM: `ollama pull phi3:mini`")

            # Section supprimée selon demande utilisateur

        except Exception as e:
            st.error(f"❌ Erreur Service LLM: {str(e)}")

    # 5. TABLEAU DES DÉCISIONS RÉCENTES
    show_decisions_table(ticker)

    # 6. ÉTATS DES SERVICES - Déplacé en fin de page et corrigé
    st.header("🔧 États des Services")

    try:
        # Récupérer les statuts des services
        articles = services["sentiment_service"].get_news_articles(ticker, 10)
        article_count = len(articles) if articles else 0
        data = services["data_service"].load_data(ticker)

        # Utiliser l'heure fixe de la dernière décision au lieu de l'heure actuelle
        try:
            # Charger les décisions récentes pour avoir l'heure fixe
            decisions_path = Path("data/trading/decisions_log/trading_decisions.json")
            if decisions_path.exists():
                with open(decisions_path, "r") as f:
                    decisions = json.load(f)
                if decisions:
                    # Utiliser l'heure de la dernière décision
                    last_decision = decisions[-1]
                    timestamp_str = last_decision.get("timestamp", "")
                    if timestamp_str:
                        import pytz

                        # Parser le timestamp
                        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                        # Convertir en heure de Paris
                        paris_tz = pytz.timezone("Europe/Paris")
                        paris_time = dt.astimezone(paris_tz)
                        last_update = paris_time.strftime("%H:%M")
                    else:
                        last_update = "N/A"
                else:
                    last_update = "N/A"
            else:
                last_update = "N/A"
        except Exception as e:
            last_update = "N/A"

        # Construire la ligne de statut
        if market_status["is_open"]:
            market_status_text = "🟢 Marché Ouvert (EST)"
            market_time = f"Heure: {market_status['current_time']}"
            next_info = f"Fermeture: {market_status['next_close']}"
            warning = ""
        else:
            market_status_text = "🔴 Marché Fermé (EST)"
            market_time = f"Heure: {market_status['current_time']}"
            next_info = f"Ouverture: {market_status['next_open']}"
            warning = "⚠️ Aucune prédiction disponible en dehors des heures de marché"

        # Services status - Logique corrigée selon état du marché
        crawler_status = "🟢" if article_count > 0 else "🔴"

        # Prix : données historiques toujours disponibles, mais pas de prix en temps réel si marché fermé
        if not data.empty:
            if market_status["is_open"]:
                price_status = "🟢"  # Données historiques + marché ouvert
            else:
                price_status = "🟡"  # Données historiques seulement (pas de temps réel)
        else:
            price_status = "🔴"  # Pas de données du tout

        # Fusion : seulement si données prix disponibles ET marché ouvert
        if market_status["is_open"] and not data.empty:
            fusion_status = "🟢"
        else:
            fusion_status = "🔴"

        status_line = f"{market_status_text} | {market_time} | {next_info}"
        if warning:
            status_line += f" | {warning}"
        status_line += f" | Articles: {crawler_status} {article_count} | Prix: {price_status} {last_update} | Fusion: {fusion_status}"

        st.markdown(
            f"""
        <div class="status-line">
            {status_line}
        </div>
        """,
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"Erreur statut services: {e}")


def _check_fusion_dependencies(services, ticker, market_status):
    """Vérifie les dépendances critiques pour le calcul de fusion"""
    errors = []
    fusion_available = True

    try:
        # 1. Vérifier la disponibilité des données de prix
        try:
            data = services["data_service"].load_data(ticker)
            if data.empty:
                errors.append(
                    {
                        "severity": "critical",
                        "title": "Données Prix Manquantes",
                        "message": "Aucune donnée de prix disponible pour le calcul de fusion.",
                    }
                )
                fusion_available = False
        except Exception as e:
            errors.append(
                {
                    "severity": "critical",
                    "title": "Erreur Chargement Prix",
                    "message": f"Impossible de charger les données de prix: {str(e)}",
                }
            )
            fusion_available = False

        # 2. Vérifier la prédiction (seulement pour SPY) - Utiliser le cache
        if ticker == "SPY":
            try:
                cache_key = f"{ticker}_prediction"
                if cache_key in st.session_state.prediction_cache:
                    prediction = st.session_state.prediction_cache[cache_key]
                else:
                    prediction = services["prediction_service"].predict_with_features(ticker, horizon=20)
                    st.session_state.prediction_cache[cache_key] = prediction

                if not prediction or ("predictions" not in prediction and "trend_signal" not in prediction):
                    errors.append(
                        {
                            "severity": "warning",
                            "title": "Prédiction Indisponible",
                            "message": "Les prédictions LSTM ne sont pas disponibles. Utilisation de valeurs par défaut.",
                        }
                    )
            except Exception as e:
                errors.append(
                    {
                        "severity": "warning",
                        "title": "Erreur Prédiction",
                        "message": f"Erreur lors du calcul de prédiction: {str(e)}",
                    }
                )

        # 3. Vérifier le sentiment
        try:
            articles = services["sentiment_service"].get_news_articles(ticker, 5)
            if not articles:
                errors.append(
                    {
                        "severity": "warning",
                        "title": "Articles Manquants",
                        "message": "Aucun article disponible pour l'analyse de sentiment.",
                    }
                )
        except Exception as e:
            errors.append(
                {
                    "severity": "warning",
                    "title": "Erreur Sentiment",
                    "message": f"Erreur lors de la récupération des articles: {str(e)}",
                }
            )

        # 4. Vérifier l'état du marché
        if not market_status["is_open"] and ticker == "SPY":
            errors.append(
                {
                    "severity": "info",
                    "title": "Marché Fermé",
                    "message": "Le marché est fermé. Les prédictions peuvent être moins fiables.",
                }
            )

    except Exception as e:
        errors.append(
            {
                "severity": "critical",
                "title": "Erreur Système",
                "message": f"Erreur lors de la vérification des dépendances: {str(e)}",
            }
        )
        fusion_available = False

    return fusion_available, errors


def _get_price_signal(services, ticker):
    """Récupère le signal de prix"""
    try:
        data = services["data_service"].load_data(ticker)
        if data.empty:
            return 0.0

        # Calculer la tendance sur les 5 dernières périodes
        recent_data = data.tail(5)
        if len(recent_data) < 2:
            return 0.0

        # Tendance basée sur la variation moyenne
        price_change = (recent_data["Close"].iloc[-1] - recent_data["Close"].iloc[0]) / recent_data["Close"].iloc[0]

        # Normaliser entre 0 et 1 (0.5 = neutre)
        signal = 0.5 + (price_change * 2)  # Multiplier par 2 pour amplifier
        return max(0.0, min(1.0, signal))  # Clamper entre 0 et 1

    except Exception:
        return 0.0


def _get_sentiment_signal(services, ticker):
    """Récupère le signal de sentiment"""
    try:
        sentiment_summary = services["sentiment_service"].get_sentiment_summary(ticker)
        if "avg_sentiment" in sentiment_summary:
            # Normaliser de [-1, 1] vers [0, 1]
            return (sentiment_summary["avg_sentiment"] + 1) / 2
        return 0.5  # Neutre par défaut
    except Exception:
        return 0.5


def _get_prediction_signal(services, ticker):
    """Récupère le signal de prédiction (utilise le cache)"""
    try:
        if ticker != "SPY":
            return 0.5  # Neutre pour NVDA

        # Utiliser le cache de prédiction au lieu de recalculer
        cache_key = f"{ticker}_prediction"
        if cache_key in st.session_state.prediction_cache:
            prediction = st.session_state.prediction_cache[cache_key]
        else:
            prediction = services["prediction_service"].predict_with_features(ticker, horizon=20)
            st.session_state.prediction_cache[cache_key] = prediction

        if prediction and "trend_signal" in prediction:
            # Utiliser le trend_signal normalisé (-1 à +1) et le convertir en 0-1
            trend_signal = prediction["trend_signal"]
            return (trend_signal + 1) / 2  # Convertir -1,+1 vers 0,1
        elif prediction and "predictions" in prediction and prediction["predictions"]:
            # Fallback: utiliser la première prédiction
            pred_value = prediction["predictions"][0] if prediction["predictions"] else 0.5
            return max(0.0, min(1.0, pred_value))

        return 0.5  # Neutre par défaut
    except Exception:
        return 0.5


def show_decisions_table(ticker):
    """Affiche le tableau des décisions récentes"""
    st.header("📋 Décisions Récentes - Synthèse")

    try:
        # Service de validation unifié
        decision_validation = DecisionValidationService()
        
        # Récupérer l'historique des validations (décisions déjà validées)
        validation_history_df = decision_validation.get_validation_history(ticker, days=7)
        validation_results = validation_history_df.to_dict('records') if not validation_history_df.empty else []

        # Récupérer les décisions en attente de validation
        pending_decisions = decision_validation.get_pending_decisions(ticker)

        # Combiner les décisions validées et en attente (SANS DOUBLONS)
        all_decisions = []
        seen_timestamps = set()

        def normalize_timestamp(ts):
            """Normalise le timestamp pour la comparaison (enlève microsecondes et timezone)"""
            if isinstance(ts, str):
                # Parse et normalise
                try:
                    if "T" in ts:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    else:
                        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                    # Retourner format normalisé sans microsecondes
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    return str(ts)[:19]  # Garder YYYY-MM-DD HH:MM:SS
            elif hasattr(ts, "strftime"):
                return ts.strftime("%Y-%m-%d %H:%M:%S")
            return str(ts)[:19]

        # IMPORTANT: Ajouter PENDING d'abord pour afficher "⏳ Attente..." au lieu des prix validés
        # Ajouter les décisions en attente EN PREMIER
        for decision in pending_decisions:
            decision["status"] = "pending"
            timestamp_normalized = normalize_timestamp(decision.get("timestamp", ""))
            if timestamp_normalized not in seen_timestamps:
                all_decisions.append(decision)
                seen_timestamps.add(timestamp_normalized)

        # Ajouter les décisions validées (ignorées si déjà en attente)
        for decision in validation_results:
            decision["status"] = "validated"
            timestamp_normalized = normalize_timestamp(decision.get("timestamp", ""))
            if timestamp_normalized not in seen_timestamps:
                all_decisions.append(decision)
                seen_timestamps.add(timestamp_normalized)

        if all_decisions:
            # Trier par timestamp pour avoir les plus récentes en premier
            def get_timestamp_key(decision):
                timestamp = decision.get("timestamp", "")
                if isinstance(timestamp, str):
                    try:
                        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        return dt
                    except:
                        return datetime.min.replace(tzinfo=timezone.utc)
                elif hasattr(timestamp, "timestamp"):
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                    return timestamp
                else:
                    return datetime.min.replace(tzinfo=timezone.utc)

            recent_decisions = sorted(all_decisions, key=get_timestamp_key, reverse=True)

            # Limiter à 10 lignes pour l'affichage
            display_decisions = recent_decisions[:10]

            # Créer le tableau avec les colonnes spécifiées
            table_data = []
            for i, decision in enumerate(display_decisions):
                # Gérer l'affichage de l'heure correctement (Heure de Paris)
                timestamp = decision.get("timestamp", "N/A")
                if hasattr(timestamp, "strftime"):
                    # Objet datetime - convertir en heure de Paris
                    import pytz

                    try:
                        # Si c'est déjà en UTC, convertir en Paris
                        if timestamp.tzinfo is None:
                            timestamp = timestamp.replace(tzinfo=pytz.UTC)
                        paris_tz = pytz.timezone("Europe/Paris")
                        paris_time = timestamp.astimezone(paris_tz)
                        heure_str = paris_time.strftime("%H:%M")
                        date_str = paris_time.strftime("%d/%m")
                    except:
                        heure_str = timestamp.strftime("%H:%M")
                        date_str = timestamp.strftime("%d/%m")
                elif isinstance(timestamp, str):
                    # Chaîne de caractères - essayer de parser
                    try:
                        if "T" in timestamp:
                            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                        else:
                            dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

                        # Convertir en heure de Paris
                        import pytz

                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=pytz.UTC)
                        paris_tz = pytz.timezone("Europe/Paris")
                        paris_time = dt.astimezone(paris_tz)
                        heure_str = paris_time.strftime("%H:%M")
                        date_str = paris_time.strftime("%d/%m")
                    except Exception as e:
                        if " " in timestamp and ":" in timestamp:
                            time_part = timestamp.split(" ")[1]
                            heure_str = time_part[:5]
                            date_str = timestamp.split(" ")[0][5:]
                        else:
                            heure_str = str(timestamp)[:5]
                            date_str = "N/A"
                else:
                    heure_str = str(timestamp)[:5]
                    date_str = "N/A"

                # Calculer les prix et le gain INSTANTANÉMENT
                current_price = decision.get("current_price", 0)
                decision_status = decision.get("status", "validated")

                # Récupérer le timestamp de la décision
                timestamp = decision.get("timestamp", "")
                try:
                    if isinstance(timestamp, str):
                        if "T" in timestamp:
                            decision_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                        else:
                            decision_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                            decision_time = decision_time.replace(tzinfo=timezone.utc)
                    elif hasattr(timestamp, "timestamp"):
                        decision_time = timestamp
                        if decision_time.tzinfo is None:
                            decision_time = decision_time.replace(tzinfo=timezone.utc)
                    else:
                        decision_time = None
                except:
                    decision_time = None

                # Calculer le prix 15min APRÈS la décision
                future_price = None
                if decision_time:
                    try:
                        # Charger les prix 15min
                        prices_path = CONSTANTS.get_data_path("prices", ticker, "15min")
                        if prices_path.exists():
                            prices_df = pd.read_parquet(prices_path)
                            prices_df['ts_utc'] = pd.to_datetime(prices_df['ts_utc'])

                            # Trouver le prix 15 minutes APRÈS la décision
                            target_time = decision_time + pd.Timedelta(minutes=15)

                            # Chercher le prix le plus proche dans une fenêtre de +/- 5 minutes
                            window_start = target_time - pd.Timedelta(minutes=5)
                            window_end = target_time + pd.Timedelta(minutes=5)

                            mask = (prices_df['ts_utc'] >= window_start) & (prices_df['ts_utc'] <= window_end)
                            matching_prices = prices_df[mask]

                            if not matching_prices.empty:
                                # Prendre le prix le plus proche de target_time
                                matching_prices = matching_prices.copy()
                                matching_prices['time_diff'] = abs((matching_prices['ts_utc'] - target_time).dt.total_seconds())
                                closest_row = matching_prices.loc[matching_prices['time_diff'].idxmin()]
                                future_price = float(closest_row['close'])
                    except Exception as e:
                        logger.error(f"Erreur récupération prix 15min: {e}")

                # Afficher les résultats
                if future_price is not None:
                    # Calculer le gain en dollars
                    gain_dollars = future_price - current_price

                    # Déterminer le résultat basé sur la comparaison des prix
                    # Si prix -15min < prix +15min → Positif (hausse)
                    # Si prix -15min > prix +15min → Négatif (baisse)
                    is_positive = current_price < future_price
                    result_text = "Positif ✅" if is_positive else "Négatif ❌"

                    future_price_display = f"${future_price:.2f}"
                    gain_display = f"${gain_dollars:+.2f}"
                else:
                    # Prix 15min pas encore disponible
                    future_price_display = "⏳ Attente..."
                    gain_display = "⏳"
                    result_text = "⏳ En attente..."

                table_data.append(
                    {
                        "N°": i + 1,
                        "Date": date_str,
                        "Heure": heure_str,
                        "Prix -15min": f"${current_price:.2f}",
                        "Prix +15min": future_price_display,
                        "Décision": decision.get("decision", "N/A"),
                        "Résultat": result_text,
                        "Gain": gain_display,
                    }
                )

            if table_data:
                df_table = pd.DataFrame(table_data)

                # Afficher le tableau en pleine largeur
                st.dataframe(
                    df_table,
                    width="stretch",
                    height=300,
                    column_config={
                        "N°": st.column_config.NumberColumn("N°", width="small"),
                        "Date": st.column_config.TextColumn("Date", width="small"),
                        "Heure": st.column_config.TextColumn("Heure", width="small"),
                        "Prix -15min": st.column_config.TextColumn("Prix -15min", width="medium"),
                        "Prix +15min": st.column_config.TextColumn("Prix +15min", width="medium"),
                        "Décision": st.column_config.TextColumn("Décision", width="small"),
                        "Résultat": st.column_config.TextColumn("Résultat", width="medium"),
                        "Gain": st.column_config.TextColumn("Gain", width="medium"),
                    },
                )

                # Statistiques simplifiées
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Décisions", len(table_data))
                with col2:
                    positive_count = sum(1 for d in table_data if d["Résultat"] == "Positif")
                    st.metric("Résultats Positifs", f"{positive_count}/{len(table_data)}")
                with col3:
                    # Calculer le gain total (ignorer les décisions en attente avec "⏳")
                    validated_gains = [d for d in table_data if d["Gain"] != "⏳"]
                    if validated_gains:
                        total_gain = sum(float(d["Gain"].replace("$", "").replace("+", "")) for d in validated_gains)
                        st.metric("Gain Total", f"${total_gain:+.2f}")
                    else:
                        st.metric("Gain Total", "En attente...")
                with col4:
                    # Bouton de téléchargement pour toutes les données
                    if len(validation_results) > 10:
                        # Créer un DataFrame complet pour le téléchargement
                        full_table_data = []
                        for i, decision in enumerate(validation_results):
                            # Même logique que pour l'affichage mais pour toutes les données
                            timestamp = decision.get("timestamp", "N/A")
                            if hasattr(timestamp, "strftime"):
                                import pytz

                                try:
                                    if timestamp.tzinfo is None:
                                        timestamp = timestamp.replace(tzinfo=pytz.UTC)
                                    paris_tz = pytz.timezone("Europe/Paris")
                                    paris_time = timestamp.astimezone(paris_tz)
                                    heure_str = paris_time.strftime("%H:%M")
                                    date_str = paris_time.strftime("%d/%m")
                                except:
                                    heure_str = timestamp.strftime("%H:%M")
                                    date_str = timestamp.strftime("%d/%m")
                            elif isinstance(timestamp, str):
                                try:
                                    if "T" in timestamp:
                                        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                                    else:
                                        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                                    import pytz

                                    if dt.tzinfo is None:
                                        dt = dt.replace(tzinfo=pytz.UTC)
                                    paris_tz = pytz.timezone("Europe/Paris")
                                    paris_time = dt.astimezone(paris_tz)
                                    heure_str = paris_time.strftime("%H:%M")
                                    date_str = paris_time.strftime("%d/%m")
                                except:
                                    if " " in timestamp and ":" in timestamp:
                                        time_part = timestamp.split(" ")[1]
                                        heure_str = time_part[:5]
                                        date_str = timestamp.split(" ")[0][5:]
                                    else:
                                        heure_str = str(timestamp)[:5]
                                        date_str = "N/A"
                            else:
                                heure_str = str(timestamp)[:5]
                                date_str = "N/A"

                            current_price = decision.get("current_price", 0)
                            future_price = decision.get("future_price", 0)
                            price_change = decision.get("price_change", 0)
                            gain_dollars = future_price - current_price

                            # Déterminer le résultat basé sur la comparaison des prix
                            # Si prix -15min < prix +15min → Positif (hausse)
                            # Si prix -15min > prix +15min → Négatif (baisse)
                            is_positive = current_price < future_price
                            result_text = "Positif" if is_positive else "Négatif"

                            full_table_data.append(
                                {
                                    "N°": i + 1,
                                    "Date": date_str,
                                    "Heure": heure_str,
                                    "Prix -15min": f"${current_price:.2f}",
                                    "Prix +15min": f"${future_price:.2f}",
                                    "Décision": decision.get("decision", "N/A"),
                                    "Résultat": result_text,
                                    "Gain": f"${gain_dollars:+.2f}",
                                }
                            )

                        # Créer le CSV pour le téléchargement
                        full_df = pd.DataFrame(full_table_data)
                        csv = full_df.to_csv(index=False)

                        st.download_button(
                            label="📥 Télécharger toutes les données",
                            data=csv,
                            file_name=f"decisions_historiques_{ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            help=f"Télécharge toutes les {len(validation_results)} décisions historiques",
                        )
                    else:
                        st.info("Toutes les données sont affichées")
            else:
                st.info("Aucune décision récente disponible")
        else:
            st.info("Aucune donnée de validation disponible")
    except Exception as e:
        st.warning(f"⚠️ Erreur chargement tableau: {str(e)}")


def _check_market_status():
    """Vérifie l'état du marché (ouvert/fermé) - Horaires US (EDT/EST)"""
    import pytz

    # Utiliser pytz pour gérer correctement l'heure d'été américaine
    try:
        # Timezone US Eastern (gère automatiquement EST/EDT)
        us_eastern = pytz.timezone("US/Eastern")
        now_est = datetime.now(us_eastern)
        current_time = now_est.strftime("%H:%M")

        # Heures de marché US (9h30 - 16h00, du lundi au vendredi)
        market_open = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_est.replace(hour=16, minute=0, second=0, microsecond=0)

        # Vérifier si c'est un jour de semaine
        is_weekday = now_est.weekday() < 5  # 0-4 = lundi-vendredi

        if is_weekday and market_open <= now_est <= market_close:
            return {
                "is_open": True,
                "current_time": current_time,
                "timezone": "EDT" if now_est.dst() else "EST",
                "next_close": market_close.strftime("%H:%M"),
                "next_open": "09:30" if now_est.date() == market_open.date() else "Lundi 09:30",
            }
        else:
            # Calculer la prochaine ouverture
            if now_est.weekday() >= 5:  # Weekend
                days_until_monday = 7 - now_est.weekday()
                next_open = now_est + timedelta(days=days_until_monday)
            elif now_est < market_open:  # Avant l'ouverture
                next_open = now_est
            else:  # Après la fermeture
                next_open = now_est + timedelta(days=1)

            next_open = next_open.replace(hour=9, minute=30, second=0, microsecond=0)

            return {
                "is_open": False,
                "current_time": current_time,
                "timezone": "EDT" if now_est.dst() else "EST",
                "next_open": next_open.strftime("%A %H:%M"),
                "next_close": "16:00",
            }
    except ImportError:
        # Fallback si pytz n'est pas disponible
        # Utiliser EDT (UTC-4) pour septembre 2025
        edt = timezone(timedelta(hours=-4))
        now_est = datetime.now(edt)
        current_time = now_est.strftime("%H:%M")

        # Heures de marché US (9h30 - 16h00 EDT, du lundi au vendredi)
        market_open = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_est.replace(hour=16, minute=0, second=0, microsecond=0)

        # Vérifier si c'est un jour de semaine
        is_weekday = now_est.weekday() < 5  # 0-4 = lundi-vendredi

        if is_weekday and market_open <= now_est <= market_close:
            return {
                "is_open": True,
                "current_time": current_time,
                "timezone": "EDT",
                "next_close": market_close.strftime("%H:%M"),
                "next_open": "09:30" if now_est.date() == market_open.date() else "Lundi 09:30",
            }
        else:
            # Calculer la prochaine ouverture
            if now_est.weekday() >= 5:  # Weekend
                days_until_monday = 7 - now_est.weekday()
                next_open = now_est + timedelta(days=days_until_monday)
            elif now_est < market_open:  # Avant l'ouverture
                next_open = now_est
            else:  # Après la fermeture
                next_open = now_est + timedelta(days=1)

            next_open = next_open.replace(hour=9, minute=30, second=0, microsecond=0)

            return {
                "is_open": False,
                "current_time": current_time,
                "timezone": "EDT",
                "next_open": next_open.strftime("%A %H:%M"),
                "next_close": "16:00",
            }


def _get_next_decision_time():
    """
    Calcule la prochaine heure de décision basée sur des fenêtres fixes de 15 minutes
    Heures de marché US : 9:30-16:00 (EDT/EST)
    Fenêtres : 9:30, 9:45, 10:00, 10:15... 15:45, 16:00
    """
    try:
        import pytz

        # Timezone US Eastern (gère automatiquement EST/EDT)
        us_eastern = pytz.timezone("US/Eastern")
        now_est = datetime.now(us_eastern)

        # Heures de marché US (9:30-16:00)
        market_open = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_est.replace(hour=16, minute=0, second=0, microsecond=0)

        # Si on est avant l'ouverture, prochaine décision à 9:30
        if now_est < market_open:
            next_decision = market_open
        # Si on est après la fermeture, prochaine décision demain à 9:30
        elif now_est >= market_close:
            next_decision = market_open + timedelta(days=1)
        else:
            # On est pendant les heures de marché, calculer la prochaine fenêtre de 15 minutes
            current_minute = now_est.minute
            current_hour = now_est.hour

            # Calculer la prochaine fenêtre de 15 minutes
            if current_minute < 30:
                next_minute = 30
            elif current_minute < 45:
                next_minute = 45
            else:
                next_minute = 0
                current_hour += 1

            # Si on dépasse 16:00, prochaine décision demain à 9:30
            if current_hour >= 16:
                next_decision = market_open + timedelta(days=1)
            else:
                next_decision = now_est.replace(hour=current_hour, minute=next_minute, second=0, microsecond=0)

        # Convertir en heure de Paris pour l'affichage
        paris_tz = pytz.timezone("Europe/Paris")
        paris_time = next_decision.astimezone(paris_tz)
        return paris_time.strftime("%H:%M")

    except ImportError:
        # Fallback si pytz n'est pas disponible
        # Utiliser EDT (UTC-4) pour septembre 2025
        edt = timezone(timedelta(hours=-4))
        now_est = datetime.now(edt)

        # Heures de marché US (9:30-16:00 EDT)
        market_open = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_est.replace(hour=16, minute=0, second=0, microsecond=0)

        # Si on est avant l'ouverture, prochaine décision à 9:30
        if now_est < market_open:
            next_decision = market_open
        # Si on est après la fermeture, prochaine décision demain à 9:30
        elif now_est >= market_close:
            next_decision = market_open + timedelta(days=1)
        else:
            # On est pendant les heures de marché, calculer la prochaine fenêtre de 15 minutes
            current_minute = now_est.minute
            current_hour = now_est.hour

            # Calculer la prochaine fenêtre de 15 minutes
            if current_minute < 30:
                next_minute = 30
            elif current_minute < 45:
                next_minute = 45
            else:
                next_minute = 0
                current_hour += 1

            # Si on dépasse 16:00, prochaine décision demain à 9:30
            if current_hour >= 16:
                next_decision = market_open + timedelta(days=1)
            else:
                next_decision = now_est.replace(hour=current_hour, minute=next_minute, second=0, microsecond=0)

        # Convertir en heure de Paris pour l'affichage (fallback)
        # Calculer le décalage horaire (EDT = UTC-4, Paris = UTC+2 en été)
        paris_offset = 6  # 6 heures de décalage entre EDT et Paris en été
        paris_hour = (next_decision.hour + paris_offset) % 24
        paris_time = next_decision.replace(hour=paris_hour)
        return paris_time.strftime("%H:%M")
