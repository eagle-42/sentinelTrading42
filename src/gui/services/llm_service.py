"""
Service LLM pour génération de synthèses de trading
Limite les tokens pour des réponses concises et efficaces
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import requests
from loguru import logger

from src.constants import CONSTANTS


class LLMService:
    """Service LLM avec limitation de tokens pour synthèses efficaces"""

    def __init__(self):
        self.ollama_url = CONSTANTS.OLLAMA_URL
        self.model = CONSTANTS.OLLAMA_MODEL
        self.max_tokens = CONSTANTS.OLLAMA_MAX_TOKENS
        self.max_context = 2000  # Limite du contexte d'entrée

        logger.info("🧠 Service LLM initialisé")

    def generate_trading_synthesis(
        self, ticker: str, recommendation: str, fusion_score: float, price: float, sentiment_score: float = None
    ) -> Dict:
        """Génère une synthèse concise de la décision de trading"""
        try:
            # Construire le prompt concis
            prompt = self._build_concise_prompt(ticker, recommendation, fusion_score, price, sentiment_score)

            # Appel à Ollama
            response = self._call_ollama(prompt)

            if response:
                return {
                    "success": True,
                    "synthesis": response,
                    "model": self.model,
                    "timestamp": datetime.now().isoformat(),
                    "tokens_used": len(response.split()),  # Estimation
                }
            else:
                return {
                    "success": False,
                    "synthesis": "Service LLM indisponible",
                    "model": self.model,
                    "timestamp": datetime.now().isoformat(),
                    "tokens_used": 0,
                }

        except Exception as e:
            logger.error(f"❌ Erreur génération synthèse: {e}")
            return {
                "success": False,
                "synthesis": f"Erreur: {str(e)}",
                "model": self.model,
                "timestamp": datetime.now().isoformat(),
                "tokens_used": 0,
            }

    def _build_concise_prompt(
        self, ticker: str, recommendation: str, fusion_score: float, price: float, sentiment_score: float = None
    ) -> str:
        """Construit un prompt concis pour la synthèse"""

        # Informations de base
        base_info = f"Ticker: {ticker}, Prix: ${price:.2f}, Recommandation: {recommendation}, Score: {fusion_score:.3f}"

        # Ajouter sentiment si disponible
        if sentiment_score is not None:
            base_info += f", Sentiment: {sentiment_score:.3f}"

        # Prompt concis avec limitation stricte
        prompt = f"""
Analyse trading concise (max 100 mots):

{base_info}

Synthèse: Explique brièvement pourquoi {recommendation} pour {ticker}.
Focus: Score de fusion, tendance prix, sentiment.
Format: 2-3 phrases maximum.
"""

        return prompt.strip()

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Appelle l'API Ollama avec limitation de tokens"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": self.max_tokens,  # Limite stricte
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stop": ["\n\n", "---", "##"],  # Arrêts pour concision
                },
            }

            response = requests.post(
                self.ollama_url, json=payload, timeout=10  # Timeout réduit pour éviter les blocages
            )

            if response.status_code == 200:
                result = response.json()
                synthesis = result.get("response", "").strip()

                # Vérifier la longueur
                if len(synthesis.split()) > self.max_tokens:
                    synthesis = " ".join(synthesis.split()[: self.max_tokens]) + "..."

                logger.info(f"✅ Synthèse générée: {len(synthesis.split())} mots")
                return synthesis
            else:
                logger.warning(f"⚠️ Erreur API Ollama: {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erreur connexion Ollama: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Erreur appel Ollama: {e}")
            return None

    def check_service_status(self) -> Dict:
        """Vérifie le statut du service Ollama"""
        try:
            response = requests.get(CONSTANTS.OLLAMA_TAGS_URL, timeout=5)

            if response.status_code == 200:
                models = response.json().get("models", [])
                phi3_available = any("phi3" in model.get("name", "") for model in models)

                return {
                    "online": True,
                    "model_available": phi3_available,
                    "models": [model.get("name", "") for model in models],
                    "model": "phi3:mini" if phi3_available else "N/A",
                    "status": "✅ Service actif" if phi3_available else "⚠️ Modèle phi3 non trouvé",
                }
            else:
                return {
                    "online": False,
                    "model_available": False,
                    "models": [],
                    "model": "N/A",
                    "status": "❌ Service indisponible",
                }

        except Exception as e:
            logger.error(f"❌ Erreur vérification statut: {e}")
            return {
                "online": False,
                "model_available": False,
                "models": [],
                "model": "N/A",
                "status": f"❌ Erreur: {str(e)}",
            }

    def save_synthesis(self, ticker: str, synthesis_data: Dict):
        """Sauvegarde une synthèse dans les logs"""
        try:
            from pathlib import Path

            data_path = CONSTANTS.get_data_path()
            synthesis_path = data_path / "trading" / "llm_synthesis"
            synthesis_path.mkdir(parents=True, exist_ok=True)

            # Fichier par ticker et date
            filename = f"{ticker}_synthesis_{datetime.now().strftime('%Y%m%d')}.json"
            file_path = synthesis_path / filename

            # Charger les synthèses existantes
            if file_path.exists():
                with open(file_path, "r") as f:
                    syntheses = json.load(f)
            else:
                syntheses = []

            # Ajouter la nouvelle synthèse
            syntheses.append(synthesis_data)

            # Sauvegarder
            with open(file_path, "w") as f:
                json.dump(syntheses, f, indent=2)

            logger.info(f"✅ Synthèse sauvegardée: {file_path}")

        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde synthèse: {e}")

    def generate_automatic_synthesis(self, ticker: str, decision_data: Dict) -> Dict:
        """Génère automatiquement une synthèse basée sur une décision de trading"""
        try:
            # Extraire les données de la décision
            recommendation = decision_data.get("recommendation", decision_data.get("decision", "HOLD"))
            confidence = decision_data.get("confidence", 0.5)
            fused_signal = decision_data.get("fused_signal", decision_data.get("score", 0.0))
            signals = decision_data.get("signals", {})

            # Construire le prompt pour synthèse automatique
            prompt = f"""
Analyse trading automatique (max 100 mots):

Ticker: {ticker}
Décision: {recommendation}
Confiance: {confidence:.1%}
Signal fusionné: {fused_signal:.3f}
Signaux: Prix={signals.get('price', 0):.3f}, Sentiment={signals.get('sentiment', 0):.3f}

Synthèse: Explique pourquoi {recommendation} pour {ticker}.
Focus: Confiance élevée, signaux cohérents.
Format: 2-3 phrases maximum.
"""

            # Appel à Ollama
            response = self._call_ollama(prompt)

            if response:
                synthesis_data = {
                    "success": True,
                    "synthesis": response,
                    "model": self.model,
                    "timestamp": datetime.now().isoformat(),
                    "tokens_used": len(response.split()),
                    "auto_generated": True,
                    "decision_timestamp": decision_data.get("timestamp", ""),
                    "confidence": confidence,
                    "fused_signal": fused_signal,
                }

                # Sauvegarder automatiquement
                self.save_synthesis(ticker, synthesis_data)

                return synthesis_data
            else:
                return {
                    "success": False,
                    "synthesis": "Synthèse automatique indisponible",
                    "model": self.model,
                    "timestamp": datetime.now().isoformat(),
                    "tokens_used": 0,
                    "auto_generated": True,
                }

        except Exception as e:
            logger.error(f"❌ Erreur synthèse automatique: {e}")
            return {
                "success": False,
                "synthesis": f"Erreur: {str(e)}",
                "model": self.model,
                "timestamp": datetime.now().isoformat(),
                "tokens_used": 0,
                "auto_generated": True,
            }
