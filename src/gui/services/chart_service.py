"""
Service de graphiques pour Streamlit
Génération des graphiques Plotly optimisés
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from loguru import logger
from plotly.subplots import make_subplots


class ChartService:
    """Service de graphiques robuste pour Streamlit"""

    def __init__(self):
        # Configuration déterministe
        self.colors = {
            "price": "#1f77b4",
            "ma_20": "#ff7f0e",
            "ma_50": "#d62728",
            "ma_100": "#9467bd",
            "volume": "#2ca02c",
            "volatility": "#e377c2",
            "sentiment": "#17becf",
            "prediction": "#bcbd22",
        }
        logger.info("📊 Service de graphiques initialisé")

    def _get_xaxis_config(self, period: str, dates: pd.Series) -> dict:
        """Configure l'axe X selon la période pour un affichage optimal"""
        if period in ["7 derniers jours"]:
            return {"tickformat": "%d/%m %H:%M", "tickmode": "auto", "nticks": 7, "tickangle": 45}
        elif period in ["1 mois"]:
            return {"tickformat": "%d/%m", "tickmode": "auto", "nticks": 10, "tickangle": 45}
        elif period in ["3 mois", "6 derniers mois"]:
            return {"tickformat": "%d/%m", "tickmode": "auto", "nticks": 8, "tickangle": 45}
        elif period in ["1 an"]:
            return {"tickformat": "%m/%Y", "tickmode": "auto", "nticks": 6, "tickangle": 45}
        elif period in ["3 ans", "5 ans", "10 ans"]:
            return {"tickformat": "%m/%Y", "tickmode": "auto", "nticks": 8, "tickangle": 45}
        else:  # Total
            return {"tickformat": "%Y", "tickmode": "auto", "nticks": 10, "tickangle": 45}

    def create_price_chart(self, df: pd.DataFrame, ticker: str, period: str) -> go.Figure:
        """Crée un graphique de prix avec moyennes mobiles"""
        try:
            if df.empty:
                return self._create_error_figure("Aucune donnée disponible")

            # Normaliser les colonnes pour chart (minuscules)
            df_work = df.copy()
            if "DATE" in df_work.index.names:
                df_work = df_work.reset_index()
            df_work.columns = df_work.columns.str.lower()

            # Nouvelle figure à chaque appel
            fig = go.Figure()

            # Données triées par date
            df_sorted = df_work.sort_values("date").reset_index(drop=True)

            # Prix de clôture
            fig.add_trace(
                go.Scatter(
                    x=df_sorted["date"],
                    y=df_sorted["close"],
                    mode="lines",
                    name=f"{ticker} - Prix",
                    line=dict(color=self.colors["price"], width=2),
                    hovertemplate="<b>%{fullData.name}</b><br>Date: %{x}<br>Prix: $%{y:.2f}<extra></extra>",
                )
            )

            # MA 20
            if len(df_sorted) >= 5:
                window_20 = min(20, len(df_sorted))
                ma_20 = df_sorted["close"].rolling(window=20, min_periods=1).mean()
                fig.add_trace(
                    go.Scatter(
                        x=df_sorted["date"],
                        y=ma_20,
                        mode="lines",
                        name=f"MA {window_20}",
                        line=dict(color=self.colors["ma_20"], width=2, dash="dash"),
                        hovertemplate="<b>MA 20</b><br>Date: %{x}<br>Prix: $%{y:.2f}<extra></extra>",
                    )
                )

            # MA 50
            if len(df_sorted) >= 10:
                window_50 = min(50, len(df_sorted))
                if window_50 > window_20:
                    ma_50 = df_sorted["close"].rolling(window=50, min_periods=1).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=df_sorted["date"],
                            y=ma_50,
                            mode="lines",
                            name=f"MA {window_50}",
                            line=dict(color=self.colors["ma_50"], width=2, dash="dot"),
                            hovertemplate="<b>MA 50</b><br>Date: %{x}<br>Prix: $%{y:.2f}<extra></extra>",
                        )
                    )

            # MA 100
            if len(df_sorted) >= 20:
                window_100 = min(100, len(df_sorted))
                if window_100 > window_50:
                    ma_100 = df_sorted["close"].rolling(window=100, min_periods=1).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=df_sorted["date"],
                            y=ma_100,
                            mode="lines",
                            name=f"MA {window_100}",
                            line=dict(color=self.colors["ma_100"], width=2, dash="dashdot"),
                            hovertemplate="<b>MA 100</b><br>Date: %{x}<br>Prix: $%{y:.2f}<extra></extra>",
                        )
                    )

            # Configuration déterministe
            # Configuration de l'axe X selon la période
            xaxis_config = self._get_xaxis_config(period, df_sorted["date"])

            fig.update_layout(
                title=f"{ticker} - Prix avec Moyennes Mobiles ({period})",
                xaxis_title="Date",
                yaxis_title="Prix ($)",
                hovermode="x unified",
                showlegend=True,
                template="plotly_white",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                margin=dict(l=50, r=50, t=50, b=50),
                xaxis=xaxis_config,
            )

            # Limites d'axes déterministes
            if not df_sorted.empty:
                fig.update_xaxes(range=[df_sorted["date"].min(), df_sorted["date"].max()])
                price_min = df_sorted["close"].min() * 0.98
                price_max = df_sorted["close"].max() * 1.02
                fig.update_yaxes(range=[price_min, price_max])

            return fig

        except Exception as e:
            logger.error(f"❌ Erreur création graphique prix: {e}")
            return self._create_error_figure(str(e))

    def create_volume_chart(self, df: pd.DataFrame, ticker: str, period: str) -> go.Figure:
        """Crée un graphique de volume avec volatilité sur un seul graphique"""
        try:
            if df.empty:
                return self._create_error_figure("Aucune donnée disponible")

            # Nouvelle figure à chaque appel
            fig = go.Figure()

            df_sorted = df.sort_values("date").reset_index(drop=True)

            # Volume (barres)
            fig.add_trace(
                go.Bar(
                    x=df_sorted["date"],
                    y=df_sorted["volume"],
                    name=f"{ticker} - Volume",
                    marker_color=self.colors["volume"],
                    opacity=0.7,
                    hovertemplate="<b>Volume</b><br>Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>",
                )
            )

            # Volatilité du volume (ligne)
            if len(df_sorted) >= 3:
                window_size = min(20, len(df_sorted))
                volatility = df_sorted["volume"].rolling(window=window_size, min_periods=1).std().fillna(0)
                fig.add_trace(
                    go.Scatter(
                        x=df_sorted["date"],
                        y=volatility,
                        mode="lines",
                        name=f"{ticker} - Volatilité Volume",
                        line=dict(color=self.colors["volatility"], width=2),
                        hovertemplate="<b>Volatilité</b><br>Date: %{x}<br>Volatilité: %{y:,.0f}<extra></extra>",
                    )
                )

            # Configuration déterministe
            # Configuration de l'axe X selon la période
            xaxis_config = self._get_xaxis_config(period, df_sorted["date"])

            fig.update_layout(
                title=f"{ticker} - Volume et Volatilité ({period})",
                xaxis_title="Date",
                yaxis_title="Volume / Volatilité",
                template="plotly_white",
                showlegend=True,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                margin=dict(l=50, r=50, t=50, b=50),
                xaxis=xaxis_config,
            )

            # Limites d'axes
            if not df_sorted.empty:
                fig.update_xaxes(range=[df_sorted["date"].min(), df_sorted["date"].max()])

            return fig

        except Exception as e:
            logger.error(f"❌ Erreur création graphique volume: {e}")
            return self._create_error_figure(str(e))

    def create_sentiment_chart(self, df: pd.DataFrame, ticker: str, period: str) -> go.Figure:
        """Crée un graphique de sentiment"""
        try:
            if df.empty:
                return self._create_error_figure("Aucune donnée disponible")

            # Nouvelle figure à chaque appel
            fig = go.Figure()

            df_sorted = df.sort_values("date").reset_index(drop=True)

            # Calcul du sentiment
            df_sentiment = self._calculate_sentiment(df_sorted)

            # Score de sentiment
            fig.add_trace(
                go.Scatter(
                    x=df_sentiment["date"],
                    y=df_sentiment["sentiment_pct"],
                    mode="lines",
                    name=f"{ticker} - Sentiment",
                    line=dict(color=self.colors["sentiment"], width=2),
                    hovertemplate="<b>Sentiment</b><br>Date: %{x}<br>Score: %{y:.1f}%<extra></extra>",
                )
            )

            # Lignes de référence
            fig.add_hline(
                y=30, line_dash="dash", line_color="green", annotation_text="ACHETER", annotation_position="top right"
            )
            fig.add_hline(
                y=-30, line_dash="dash", line_color="red", annotation_text="VENDRE", annotation_position="bottom right"
            )
            fig.add_hline(
                y=0, line_dash="dot", line_color="gray", annotation_text="NEUTRE", annotation_position="top left"
            )

            # Configuration déterministe
            # Configuration de l'axe X selon la période
            xaxis_config = self._get_xaxis_config(period, df_sentiment["date"])

            fig.update_layout(
                title=f"{ticker} - Score de Sentiment ({period})",
                xaxis_title="Date",
                yaxis_title="Score de Sentiment (%)",
                template="plotly_white",
                showlegend=True,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                xaxis=xaxis_config,
            )

            # Limites d'axes
            if not df_sentiment.empty:
                fig.update_xaxes(range=[df_sentiment["date"].min(), df_sentiment["date"].max()])
                fig.update_yaxes(range=[-100, 100])

            return fig

        except Exception as e:
            logger.error(f"❌ Erreur création graphique sentiment: {e}")
            return self._create_error_figure(str(e))

    def create_prediction_chart(self, df: pd.DataFrame, prediction_data: Dict, ticker: str, period: str) -> go.Figure:
        """Crée un graphique de prédiction LSTM - UNIQUEMENT POUR SPY"""
        try:
            if df.empty:
                return self._create_error_figure("Aucune donnée disponible")

            # Vérifier que c'est bien SPY
            if ticker != "SPY":
                return self._create_error_figure(f"Prédiction LSTM disponible uniquement pour SPY, pas pour {ticker}")

            # Nouvelle figure à chaque appel
            fig = go.Figure()

            df_sorted = df.sort_values("date").reset_index(drop=True)

            # 1. Prix réel (bleu) - Style HOLD_FRONT
            fig.add_trace(
                go.Scatter(
                    x=df_sorted["date"],
                    y=df_sorted["close"],
                    mode="lines+markers",
                    name="Prix Réel SPY",
                    line=dict(color="blue", width=3),
                    marker=dict(size=8),
                    hovertemplate="<b>Prix Réel SPY</b><br>Date: %{x}<br>Prix: $%{y:.2f}<extra></extra>",
                )
            )

            # 2. Prédictions historiques (vert) - si disponibles
            if "historical_predictions" in prediction_data and prediction_data["historical_predictions"]:
                hist_preds = prediction_data["historical_predictions"]

                # Vérifier que les tailles correspondent
                if len(hist_preds) == len(df_sorted):
                    fig.add_trace(
                        go.Scatter(
                            x=df_sorted["date"],
                            y=hist_preds,
                            mode="lines+markers",
                            name="Prédictions Historiques (LSTM)",
                            line=dict(color="green", width=2, dash="dot"),
                            marker=dict(size=6),
                            hovertemplate="<b>Prédiction Historique</b><br>Date: %{x}<br>Prix prédit: $%{y:.2f}<extra></extra>",
                        )
                    )
                else:
                    # Tailles différentes : ne pas afficher les prédictions historiques
                    logger.warning(
                        f"⚠️ Prédictions historiques non affichées: {len(hist_preds)} prédictions vs {len(df_sorted)} données filtrées"
                    )

            # 3. Prédictions futures (rouge) - Style HOLD_FRONT
            if (
                "predictions" in prediction_data
                and prediction_data["predictions"]
                and "prediction_dates" in prediction_data
                and prediction_data["prediction_dates"]
            ):

                pred_dates = prediction_data["prediction_dates"]
                predictions = prediction_data["predictions"]

                # Vérifier que les dates sont bien des Timestamps
                if pred_dates and not isinstance(pred_dates[0], pd.Timestamp):
                    pred_dates = [pd.to_datetime(d) for d in pred_dates]

                fig.add_trace(
                    go.Scatter(
                        x=pred_dates,
                        y=predictions,
                        mode="lines+markers",
                        name="Prédictions Futures LSTM (+20j)",
                        line=dict(color="red", width=4, dash="solid"),
                        marker=dict(size=10, symbol="diamond", color="red"),
                        hovertemplate="<b>Prédiction Future LSTM (+20j)</b><br>Date: %{x}<br>Prix prédit: $%{y:.2f}<extra></extra>",
                    )
                )

                # Ligne de séparation - Style HOLD_FRONT
                last_hist_date = pd.to_datetime(df_sorted["date"].iloc[-1])
                y_min = min(df_sorted["close"].min(), min(predictions)) * 0.98
                y_max = max(df_sorted["close"].max(), max(predictions)) * 1.02

                fig.add_trace(
                    go.Scatter(
                        x=[last_hist_date, last_hist_date],
                        y=[y_min, y_max],
                        mode="lines",
                        name="Séparation",
                        line=dict(color="gray", width=2, dash="dot"),
                        showlegend=False,
                        hovertemplate="<b>Début prédiction future</b><br>Date: %{x}<extra></extra>",
                    )
                )

            # Calcul des métriques de performance
            performance_text = ""
            if "historical_predictions" in prediction_data and prediction_data["historical_predictions"]:
                hist_preds = prediction_data["historical_predictions"]
                real_prices = df_sorted["close"].values

                # VÉRIFIER que les tailles correspondent avant de calculer
                if len(hist_preds) == len(real_prices):
                    # Calcul des métriques historiques
                    mae_hist = np.mean(np.abs(np.array(hist_preds) - real_prices))
                    mape_hist = np.mean(np.abs((np.array(hist_preds) - real_prices) / real_prices)) * 100
                    correlation_hist = np.corrcoef(hist_preds, real_prices)[0, 1]

                    performance_text += f"<b>📊 PERFORMANCE HISTORIQUE:</b><br>"
                    performance_text += f"• Erreur Moyenne: {mae_hist:.2f}$<br>"
                    performance_text += f"• Erreur Relative: {mape_hist:.1f}%<br>"
                    performance_text += f"• Corrélation: {correlation_hist:.3f}<br><br>"
                else:
                    # Tailles différentes : ne pas afficher les métriques historiques
                    logger.warning(f"⚠️ Tailles différentes: prédictions={len(hist_preds)}, prix={len(real_prices)}")
                    performance_text += f"<b>📊 PERFORMANCE HISTORIQUE:</b><br>"
                    performance_text += f"• Non disponible (période filtrée)<br><br>"

            if "predictions" in prediction_data and prediction_data["predictions"]:
                future_preds = prediction_data["predictions"]
                last_real_price = df_sorted["close"].iloc[-1]

                # Calcul des métriques futures
                price_change = (future_preds[-1] - last_real_price) / last_real_price * 100
                volatility_future = np.std(np.diff(future_preds)) / np.mean(future_preds) * 100

                performance_text += f"<b>🔮 PRÉDICTIONS FUTURES (20j):</b><br>"
                performance_text += f"• Prix initial: ${last_real_price:.2f}<br>"
                performance_text += f"• Prix final prédit: ${future_preds[-1]:.2f}<br>"
                performance_text += f"• Variation attendue: {price_change:+.1f}%<br>"
                performance_text += f"• Volatilité prédite: {volatility_future:.1f}%<br>"
                performance_text += f"• Confiance modèle: {prediction_data.get('confidence', 0.7)*100:.0f}%"

            # Configuration améliorée - Style HOLD_FRONT
            # Configuration de l'axe X selon la période
            xaxis_config = self._get_xaxis_config(period, df_sorted["date"])

            fig.update_layout(
                title={
                    "text": f"🔮 PRÉDICTIONS LSTM SPY - {period}",
                    "x": 0.5,
                    "xanchor": "center",
                    "font": {"size": 18},
                },
                xaxis_title="Date",
                yaxis_title="Prix SPY ($)",
                hovermode="x unified",
                showlegend=True,
                template="plotly_white",
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99,
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="gray",
                    borderwidth=1,
                ),
                xaxis=xaxis_config,
                yaxis=dict(tickfont=dict(size=12), title_font=dict(size=14)),
                annotations=[
                    dict(
                        x=0.02,
                        y=0.98,
                        xref="paper",
                        yref="paper",
                        text=performance_text,
                        showarrow=False,
                        bgcolor="rgba(255,255,255,0.95)",
                        bordercolor="gray",
                        borderwidth=1,
                        font=dict(size=10),
                    )
                ],
            )

            return fig

        except Exception as e:
            logger.error(f"❌ Erreur création graphique prédiction: {e}")
            return self._create_error_figure(str(e))

    def _calculate_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule le sentiment basé sur le prix"""
        df_sentiment = df.copy()

        # Log-rendement
        df_sentiment["log_returns"] = np.log(df_sentiment["close"] / df_sentiment["close"].shift(1)).fillna(0)

        # Z-score sur fenêtre glissante
        window = min(20, len(df_sentiment))
        mean_returns = df_sentiment["log_returns"].rolling(window=window, min_periods=1).mean().fillna(0)
        std_returns = df_sentiment["log_returns"].rolling(window=window, min_periods=1).std().fillna(1)

        z_score = (df_sentiment["log_returns"] - mean_returns) / std_returns
        z_score = z_score.fillna(0)

        # Compression tanh
        alpha = 2.0
        sentiment_score = np.tanh(alpha * z_score)
        sentiment_pct = sentiment_score * 100

        df_sentiment["sentiment_pct"] = sentiment_pct
        return df_sentiment

    def _create_error_figure(self, message: str) -> go.Figure:
        """Crée une figure d'erreur propre"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"❌ Erreur: {message}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="red"),
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
        )
        return fig
