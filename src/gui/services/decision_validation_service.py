"""
Service de validation des décisions de trading
Utilise des règles métier pour valider les décisions avant exécution
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from src.constants import CONSTANTS


class DecisionValidationService:
    """Service de validation en temps réel des décisions de trading"""
    def __init__(self):
        self.data_path = CONSTANTS.get_data_path()
        self.decisions_path = self.data_path / "trading" / "decisions_log"
        self.validation_path = self.data_path / "trading" / "validation_log"
        self.validation_path.mkdir(parents=True, exist_ok=True)

        # Fichier parquet pour l'historique des validations
        self.validation_file = self.validation_path / "decision_validation_history.parquet"

        # Fichier pour les décisions en attente de validation
        self.pending_file = self.validation_path / "pending_decisions.json"

        logger.info("🔍 Service de validation des décisions initialisé")

    def validate_decision(
        self, ticker: str, decision: str, fusion_score: float, current_price: float, timestamp: datetime
    ) -> Dict[str, Any]:
        """
        Valide une décision de trading en temps réel

        Args:
            ticker: Symbole de l'action
            decision: Décision prise (BUY/SELL/HOLD)
            fusion_score: Score de fusion utilisé
            current_price: Prix actuel
            timestamp: Timestamp de la décision

        Returns:
            Dict contenant les résultats de validation
        """
        try:
            # Marquer TOUTES les décisions comme en attente de validation (y compris HOLD)
            validation_result = {
                "status": "pending",
                "message": "En attente de validation (15 minutes)...",
                "accuracy": None,
                "price_change": None,
                "validation_time": None,
                "is_correct": None,
                "current_price": current_price,
                "future_price": None,
                "wait_until": timestamp + timedelta(minutes=15),
            }

            # Sauvegarder la décision en attente
            self._save_pending_decision(ticker, decision, fusion_score, current_price, timestamp)

            return validation_result

        except Exception as e:
            logger.error(f"❌ Erreur validation décision: {e}")
            return {
                "status": "error",
                "message": f"Erreur validation: {str(e)}",
                "accuracy": None,
                "price_change": None,
                "validation_time": None,
                "is_correct": None,
            }

    def _save_pending_decision(
        self, ticker: str, decision: str, fusion_score: float, current_price: float, timestamp: datetime
    ):
        """Sauvegarde une décision en attente de validation"""
        try:
            pending_data = {
                "ticker": ticker,
                "decision": decision,
                "fusion_score": fusion_score,
                "current_price": current_price,
                "timestamp": timestamp.isoformat(),
                "wait_until": (timestamp + timedelta(minutes=15)).isoformat(),
            }

            # Charger les décisions en attente existantes
            if self.pending_file.exists():
                with open(self.pending_file, "r") as f:
                    pending_decisions = json.load(f)
            else:
                pending_decisions = []

            # Ajouter la nouvelle décision
            pending_decisions.append(pending_data)

            # Sauvegarder
            with open(self.pending_file, "w") as f:
                json.dump(pending_decisions, f, indent=2)

            logger.info(f"⏳ Décision en attente sauvegardée: {ticker} - {decision}")

        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde décision en attente: {e}")

    def process_pending_validations(self) -> int:
        """
        Traite toutes les décisions en attente de validation
        Retourne le nombre de validations traitées
        """
        try:
            if not self.pending_file.exists():
                return 0

            # Charger les décisions en attente
            with open(self.pending_file, "r") as f:
                pending_decisions = json.load(f)

            if not pending_decisions:
                return 0

            processed_count = 0
            remaining_decisions = []
            current_time = datetime.now(timezone.utc)

            for decision_data in pending_decisions:
                wait_until = datetime.fromisoformat(decision_data["wait_until"].replace("Z", "+00:00"))

                # Vérifier si 15 minutes se sont écoulées
                if current_time >= wait_until:
                    # Traiter la validation
                    validation_result = self._process_real_validation(decision_data)

                    # Sauvegarder la validation complète
                    self._save_validation(
                        decision_data["ticker"],
                        decision_data["decision"],
                        decision_data["fusion_score"],
                        decision_data["current_price"],
                        datetime.fromisoformat(decision_data["timestamp"].replace("Z", "+00:00")),
                        validation_result,
                    )

                    processed_count += 1
                    logger.info(f"✅ Validation traitée: {decision_data['ticker']} - {decision_data['decision']}")
                else:
                    # Garder la décision en attente
                    remaining_decisions.append(decision_data)

            # Sauvegarder les décisions restantes
            with open(self.pending_file, "w") as f:
                json.dump(remaining_decisions, f, indent=2)

            if processed_count > 0:
                logger.info(f"🔄 {processed_count} validations traitées, {len(remaining_decisions)} en attente")

            return processed_count

        except Exception as e:
            logger.error(f"❌ Erreur traitement validations en attente: {e}")
            return 0

    def get_pending_decisions(self, ticker: str) -> List[Dict[str, Any]]:
        """Récupère les décisions en attente de validation pour un ticker"""
        try:
            if not self.pending_file.exists():
                return []

            # Charger les décisions en attente
            with open(self.pending_file, "r") as f:
                pending_decisions = json.load(f)

            # Filtrer par ticker
            ticker_decisions = [d for d in pending_decisions if d.get("ticker") == ticker]

            # Convertir les timestamps en objets datetime pour le tri
            for decision in ticker_decisions:
                decision["timestamp"] = datetime.fromisoformat(decision["timestamp"].replace("Z", "+00:00"))

            # Trier par timestamp (plus récent en premier)
            ticker_decisions.sort(key=lambda x: x.get("timestamp", datetime.min), reverse=True)

            return ticker_decisions

        except Exception as e:
            logger.error(f"❌ Erreur récupération décisions en attente: {e}")
            return []

    def _process_real_validation(self, decision_data: Dict[str, Any]) -> Dict[str, Any]:
        """Traite une validation réelle avec les vraies données de prix"""
        try:
            ticker = decision_data["ticker"]
            decision = decision_data["decision"]
            current_price = decision_data["current_price"]
            timestamp = datetime.fromisoformat(decision_data["timestamp"].replace("Z", "+00:00"))

            # Charger les données de prix actuelles
            price_data = self._load_price_data(ticker)

            if price_data.empty:
                return {
                    "status": "no_data",
                    "message": "Données de prix indisponibles pour validation",
                    "accuracy": None,
                    "price_change": None,
                    "validation_time": datetime.now(),
                    "is_correct": None,
                    "current_price": current_price,
                    "future_price": None,
                }

            # Récupérer le prix actuel (15 minutes plus tard)
            future_price = self._get_current_price(price_data, timestamp + timedelta(minutes=15))

            if future_price is None:
                return {
                    "status": "no_future_data",
                    "message": "Prix futur non disponible",
                    "accuracy": None,
                    "price_change": None,
                    "validation_time": datetime.now(),
                    "is_correct": None,
                    "current_price": current_price,
                    "future_price": None,
                }

            # Calculer le changement de prix
            price_change = (future_price - current_price) / current_price * 100

            # Déterminer si la décision était correcte
            is_correct = self._evaluate_decision_correctness(decision, price_change)

            # Calculer la précision
            accuracy = self._calculate_accuracy(decision, price_change)

            # Déterminer le statut
            if accuracy >= 0.8:
                status = "✅ Correct"
                message = f"Prix: ${current_price:.2f} → ${future_price:.2f} ({price_change:+.2f}%)"
            elif accuracy >= 0.5:
                status = "⚠️ Partiellement correct"
                message = f"Prix: ${current_price:.2f} → ${future_price:.2f} ({price_change:+.2f}%)"
            else:
                status = "❌ Incorrect"
                message = f"Prix: ${current_price:.2f} → ${future_price:.2f} ({price_change:+.2f}%)"

            return {
                "status": status,
                "message": message,
                "accuracy": accuracy,
                "price_change": price_change,
                "validation_time": datetime.now(),
                "is_correct": is_correct,
                "current_price": current_price,
                "future_price": future_price,
            }

        except Exception as e:
            logger.error(f"❌ Erreur validation réelle: {e}")
            return {
                "status": "error",
                "message": f"Erreur validation: {str(e)}",
                "accuracy": None,
                "price_change": None,
                "validation_time": datetime.now(),
                "is_correct": None,
                "current_price": current_price,
                "future_price": None,
            }

    def _get_current_price(self, price_data: pd.DataFrame, target_time: datetime) -> Optional[float]:
        """Récupère le prix actuel le plus proche du temps cible"""
        try:
            if price_data.empty:
                return None

            # Trouver le prix le plus proche du temps cible
            price_data["time_diff"] = abs((price_data["ts_utc"] - target_time).dt.total_seconds())
            closest_idx = price_data["time_diff"].idxmin()

            # Vérifier que le prix est dans une fenêtre acceptable (max 30 minutes)
            time_diff_minutes = price_data.iloc[closest_idx]["time_diff"] / 60
            if time_diff_minutes > 30:
                logger.warning(f"⚠️ Prix le plus proche à {time_diff_minutes:.1f} minutes du temps cible")
                return None

            return price_data.iloc[closest_idx]["close"]

        except Exception as e:
            logger.error(f"❌ Erreur récupération prix actuel: {e}")
            return None

    def _simulate_validation(
        self, ticker: str, decision: str, current_price: float, timestamp: datetime
    ) -> Dict[str, Any]:
        """
        FONCTION DÉSACTIVÉE - Utilise VRAIES données historiques uniquement (pas de simulation)
        Valide avec les données réelles du marché
        """
        try:
            # Charger les VRAIES données de prix du marché
            price_data = self._load_price_data(ticker)

            if price_data.empty:
                logger.warning(f"⚠️ Aucune donnée réelle disponible pour valider {ticker}")
                return {
                    "status": "no_data",
                    "message": "Données de prix réelles indisponibles",
                    "accuracy": None,
                    "price_change": None,
                    "validation_time": None,
                    "is_correct": None,
                }

            # Récupérer le VRAI prix 15 minutes plus tard depuis les données du marché
            future_price = self._get_future_price(price_data, current_price, timestamp)

            if future_price is None:
                logger.warning(f"⚠️ Prix futur réel non disponible pour {ticker}")
                return {
                    "status": "pending",
                    "message": "Validation en attente de données réelles",
                    "accuracy": None,
                    "price_change": None,
                    "validation_time": None,
                    "is_correct": None,
                }

            # Calculer le changement de prix RÉEL
            price_change = (future_price - current_price) / current_price * 100

            # Déterminer si la décision était correcte basé sur les VRAIES données
            is_correct = self._evaluate_decision_correctness(decision, price_change)

            # Calculer la précision
            accuracy = self._calculate_accuracy(decision, price_change)

            # Déterminer le statut
            if accuracy >= 0.8:
                status = "✅ Correct"
                message = f"Prix réel: ${current_price:.2f} → ${future_price:.2f} ({price_change:+.2f}%)"
            elif accuracy >= 0.5:
                status = "⚠️ Partiellement correct"
                message = f"Prix réel: ${current_price:.2f} → ${future_price:.2f} ({price_change:+.2f}%)"
            else:
                status = "❌ Incorrect"
                message = f"Prix réel: ${current_price:.2f} → ${future_price:.2f} ({price_change:+.2f}%)"

            logger.info(f"✅ Validation avec VRAIES données: {ticker} {decision} - {status}")

            return {
                "status": status,
                "message": message,
                "accuracy": accuracy,
                "price_change": price_change,
                "validation_time": datetime.now(),
                "is_correct": is_correct,
                "current_price": current_price,
                "future_price": future_price,
            }

        except Exception as e:
            logger.error(f"❌ Erreur validation avec données réelles: {e}")
            return {
                "status": "error",
                "message": f"Erreur validation: {str(e)}",
                "accuracy": None,
                "price_change": None,
                "validation_time": None,
                "is_correct": None,
            }

    def _load_price_data(self, ticker: str) -> pd.DataFrame:
        """Charge les données de prix pour la validation"""
        try:
            # Essayer de charger les données 15min d'abord
            price_file = self.data_path / "realtime" / "prices" / f"{ticker.lower()}_15min.parquet"

            if price_file.exists():
                df = pd.read_parquet(price_file)
                if "ts_utc" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["ts_utc"])
                elif "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                return df

            # Fallback sur les données historiques
            hist_file = self.data_path / "historical" / "yfinance" / f"{ticker}_1999_2025.parquet"
            if hist_file.exists():
                df = pd.read_parquet(hist_file)
                if "date" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["date"])
                return df

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"❌ Erreur chargement données prix: {e}")
            return pd.DataFrame()

    def _get_future_price(self, price_data: pd.DataFrame, current_price: float, timestamp: datetime) -> float:
        """
        Récupère le prix 15 minutes plus tard depuis les vraies données
        """
        try:
            if price_data.empty:
                logger.warning("⚠️ Pas de données de prix disponibles")
                return current_price

            # Trouver l'index du prix le plus proche du timestamp
            price_data["time_diff"] = abs((price_data["ts_utc"] - timestamp).dt.total_seconds())
            closest_idx = price_data["time_diff"].idxmin()
            current_idx = closest_idx

            # Récupérer le prix 15 minutes plus tard (données en 15min)
            future_idx = current_idx + 1

            if future_idx < len(price_data):
                future_price = price_data.iloc[future_idx]["close"]
                logger.info(f"📊 Prix réel: ${current_price:.2f} → ${future_price:.2f} (15min plus tard)")
                return future_price
            else:
                # Pas de données futures, utiliser le dernier prix
                future_price = price_data.iloc[-1]["close"]
                logger.info(f"📊 Prix final: ${current_price:.2f} → ${future_price:.2f} (dernière donnée)")
                return future_price

        except Exception as e:
            logger.error(f"❌ Erreur récupération prix futur: {e}")
            return current_price

    def _evaluate_decision_correctness(self, decision: str, price_change: float) -> bool:
        """Évalue si une décision était correcte basée sur l'évolution du prix"""
        if decision.upper() == "BUY":
            return price_change > 0.5  # BUY correct si prix monte de plus de 0.5%
        elif decision.upper() == "SELL":
            return price_change < -0.5  # SELL correct si prix baisse de plus de 0.5%
        else:
            return True  # HOLD toujours considéré comme correct

    def _calculate_accuracy(self, decision: str, price_change: float) -> float:
        """Calcule la précision d'une décision"""
        if decision.upper() == "BUY":
            if price_change > 1.0:
                return 1.0  # Parfait
            elif price_change > 0.5:
                return 0.8  # Très bon
            elif price_change > 0:
                return 0.6  # Bon
            else:
                return 0.2  # Mauvais
        elif decision.upper() == "SELL":
            if price_change < -1.0:
                return 1.0  # Parfait
            elif price_change < -0.5:
                return 0.8  # Très bon
            elif price_change < 0:
                return 0.6  # Bon
            else:
                return 0.2  # Mauvais
        else:
            return 0.7  # HOLD neutre

    def _save_validation(
        self,
        ticker: str,
        decision: str,
        fusion_score: float,
        current_price: float,
        timestamp: datetime,
        validation_result: Dict[str, Any],
    ):
        """Sauvegarde une validation dans le fichier parquet"""
        try:
            validation_data = {
                "timestamp": timestamp.isoformat(),
                "ticker": ticker,
                "decision": decision,
                "fusion_score": fusion_score,
                "current_price": current_price,
                "future_price": validation_result.get("future_price"),  # 🔧 AJOUTÉ!
                "validation_time": validation_result.get("validation_time", datetime.now()).isoformat(),
                "accuracy": validation_result.get("accuracy"),
                "price_change": validation_result.get("price_change"),
                "is_correct": validation_result.get("is_correct"),
                "status": validation_result.get("status"),
                "message": validation_result.get("message"),
            }

            # Charger l'historique existant
            if self.validation_file.exists():
                df = pd.read_parquet(self.validation_file)
            else:
                df = pd.DataFrame()

            # Ajouter la nouvelle validation
            new_row = pd.DataFrame([validation_data])
            df = pd.concat([df, new_row], ignore_index=True)

            # Sauvegarder
            df.to_parquet(self.validation_file, index=False)
            logger.info(f"✅ Validation sauvegardée: {ticker} - {decision} - {validation_result.get('status')}")

        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde validation: {e}")

    def get_validation_history(self, ticker: str, days: int = 7) -> pd.DataFrame:
        """Récupère l'historique des validations pour un ticker"""
        try:
            if not self.validation_file.exists():
                return pd.DataFrame()

            df = pd.read_parquet(self.validation_file)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

            # Filtrer par ticker et période (avec timezone UTC)
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            filtered_df = df[(df["ticker"] == ticker) & (df["timestamp"] >= cutoff_date)].sort_values("timestamp")

            return filtered_df

        except Exception as e:
            logger.error(f"❌ Erreur récupération historique validation: {e}")
            return pd.DataFrame()

    def get_validation_stats(self, ticker: str, days: int = 7) -> Dict[str, Any]:
        """Calcule les statistiques de validation pour un ticker"""
        try:
            df = self.get_validation_history(ticker, days)

            if df.empty:
                return {
                    "total_decisions": 0,
                    "total_validations": 0,
                    "correct_decisions": 0,
                    "accuracy_rate": 0.0,
                    "average_accuracy": 0.0,
                    "buy_decisions": 0,
                    "sell_decisions": 0,
                    "hold_decisions": 0,
                    "buy_accuracy": 0.0,
                    "sell_accuracy": 0.0,
                }

            # Statistiques générales
            total_decisions = len(df)
            total_validations = len(df[df["is_correct"].notna()])
            correct_decisions = len(df[df["is_correct"] == True])
            accuracy_rate = correct_decisions / total_validations if total_validations > 0 else 0.0
            average_accuracy = df["accuracy"].mean() if "accuracy" in df.columns else 0.0

            # Statistiques par type de décision
            buy_decisions = len(df[df["decision"] == "BUY"])
            sell_decisions = len(df[df["decision"] == "SELL"])
            hold_decisions = len(df[df["decision"] == "HOLD"])

            buy_accuracy = df[df["decision"] == "BUY"]["accuracy"].mean() if buy_decisions > 0 else 0.0
            sell_accuracy = df[df["decision"] == "SELL"]["accuracy"].mean() if sell_decisions > 0 else 0.0

            return {
                "total_decisions": total_decisions,
                "total_validations": total_validations,
                "correct_decisions": correct_decisions,
                "accuracy_rate": round(accuracy_rate, 3),
                "average_accuracy": round(average_accuracy, 3),
                "buy_decisions": buy_decisions,
                "sell_decisions": sell_decisions,
                "hold_decisions": hold_decisions,
                "buy_accuracy": round(buy_accuracy, 3),
                "sell_accuracy": round(sell_accuracy, 3),
            }

        except Exception as e:
            logger.error(f"❌ Erreur calcul statistiques validation: {e}")
            return {
                "total_decisions": 0,
                "total_validations": 0,
                "correct_decisions": 0,
                "accuracy_rate": 0.0,
                "average_accuracy": 0.0,
                "buy_decisions": 0,
                "sell_decisions": 0,
                "hold_decisions": 0,
                "buy_accuracy": 0.0,
                "sell_accuracy": 0.0,
            }

    def get_adaptive_threshold_performance(self, ticker: str, days: int = 7) -> Dict[str, Any]:
        """Analyse la performance des seuils adaptatifs"""
        try:
            df = self.get_validation_history(ticker, days)

            if df.empty:
                return {
                    "threshold_analysis": "Aucune donnée disponible",
                    "recommended_adjustments": [],
                    "performance_score": 0.0,
                }

            # Analyser les décisions par seuil de fusion
            if "fusion_score" in df.columns:
                # Grouper par plages de fusion_score
                df["fusion_range"] = pd.cut(
                    df["fusion_score"],
                    bins=[-np.inf, -0.1, -0.05, 0.05, 0.1, np.inf],
                    labels=["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"],
                )

                performance_by_range = (
                    df.groupby("fusion_range")
                    .agg({"is_correct": ["count", "sum", "mean"], "accuracy": "mean"})
                    .round(3)
                )

                # Recommandations d'ajustement
                recommendations = []
                if performance_by_range.loc["Positive", ("is_correct", "mean")] < 0.6:
                    recommendations.append("Considérer augmenter le seuil BUY")
                if performance_by_range.loc["Negative", ("is_correct", "mean")] < 0.6:
                    recommendations.append("Considérer diminuer le seuil SELL")

                performance_score = df["accuracy"].mean() if "accuracy" in df.columns else 0.0

                return {
                    "threshold_analysis": performance_by_range.to_dict(),
                    "recommended_adjustments": recommendations,
                    "performance_score": round(performance_score, 3),
                }
            else:
                return {
                    "threshold_analysis": "Données de fusion_score manquantes",
                    "recommended_adjustments": [],
                    "performance_score": 0.0,
                }

        except Exception as e:
            logger.error(f"❌ Erreur analyse performance seuils: {e}")
            return {"threshold_analysis": f"Erreur: {str(e)}", "recommended_adjustments": [], "performance_score": 0.0}
