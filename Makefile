# Sentinel2 - Système de Trading Algorithmique
# Makefile pour la gestion complète de l'application

.PHONY: help install start stop restart clean status logs

# Variables
APP_NAME = sentinel2
STREAMLIT_PORT = 8501
OLLAMA_PORT = 11434
VENV_DIR = .venv

# Couleurs pour les messages
GREEN = \033[0;32m
YELLOW = \033[1;33m
RED = \033[0;31m
NC = \033[0m # No Color

help: ## Afficher l'aide
	@echo "$(GREEN)🚀 Sentinel2 - Système de Trading Algorithmique$(NC)"
	@echo ""
	@echo "$(YELLOW)Commandes disponibles:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Installer les dépendances
	@echo "$(YELLOW)📦 Installation des dépendances...$(NC)"
	uv sync
	@echo "$(GREEN)✅ Dépendances installées$(NC)"

start: ## Démarrer l'application COMPLÈTE (Ollama + Prefect + Worker + Streamlit)
	@echo "$(YELLOW)🚀 Démarrage COMPLET de Sentinel2...$(NC)"
	@make start-ollama
	@echo "$(YELLOW)⏳ Attente démarrage Ollama...$(NC)"
	@sleep 3
	@make start-prefect-server
	@echo "$(YELLOW)⏳ Attente démarrage Prefect...$(NC)"
	@sleep 5
	@make start-prefect-worker
	@echo "$(YELLOW)⏳ Attente démarrage worker...$(NC)"
	@sleep 3
	@make start-streamlit
	@echo "$(GREEN)✅ Application démarrée !$(NC)"
	@echo "$(GREEN)   Streamlit: http://localhost:$(STREAMLIT_PORT)$(NC)"
	@echo "$(GREEN)   Prefect:   http://localhost:4200$(NC)"
	@echo "$(YELLOW)   Note: Orchestration gérée par Prefect Worker$(NC)"

start-ollama: ## Démarrer Ollama en arrière-plan
	@echo "$(YELLOW)🧠 Démarrage d'Ollama...$(NC)"
	@if ! pgrep -f "ollama serve" > /dev/null; then \
		ollama serve > /dev/null 2>&1 & \
		echo "$(GREEN)✅ Ollama démarré$(NC)"; \
	else \
		echo "$(YELLOW)⚠️ Ollama déjà en cours d'exécution$(NC)"; \
	fi

start-prefect-server: ## Démarrer le serveur Prefect avec chemin direct
	@echo "$(YELLOW)🚀 Démarrage serveur Prefect...$(NC)"
	@if ! lsof -i :4200 > /dev/null 2>&1; then \
		nohup .venv/bin/prefect server start --host 0.0.0.0 --port 4200 > data/logs/prefect_server.log 2>&1 & \
		echo "$(GREEN)✅ Prefect serveur démarré$(NC)"; \
	else \
		echo "$(YELLOW)⚠️ Prefect serveur déjà en cours$(NC)"; \
	fi

start-prefect-worker: ## Démarrer le worker Prefect avec chemin direct
	@echo "$(YELLOW)🤖 Démarrage worker Prefect...$(NC)"
	@if ! pgrep -f "prefect worker start --pool sentinel" > /dev/null; then \
		PREFECT_API_URL=http://localhost:4200/api nohup .venv/bin/prefect worker start --pool sentinel > data/logs/prefect_worker.log 2>&1 & \
		echo "$(GREEN)✅ Prefect worker démarré$(NC)"; \
	else \
		echo "$(YELLOW)⚠️ Worker déjà en cours d'exécution$(NC)"; \
	fi

# Orchestrateur supprimé - remplacé par Prefect Worker

start-streamlit: ## Démarrer Streamlit
	@echo "$(YELLOW)📊 Démarrage de Streamlit...$(NC)"
	@if ! pgrep -f "streamlit run" > /dev/null; then \
		uv run streamlit run src/gui/main.py --server.port $(STREAMLIT_PORT) > /dev/null 2>&1 & \
		echo "$(GREEN)✅ Streamlit démarré$(NC)"; \
	else \
		echo "$(YELLOW)⚠️ Streamlit déjà en cours d'exécution$(NC)"; \
	fi

stop: ## Arrêter l'application COMPLÈTE
	@echo "$(YELLOW)🛑 Arrêt COMPLET de Sentinel2...$(NC)"
	@make stop-streamlit
	@make stop-prefect
	@make stop-ollama
	@echo "$(GREEN)✅ Application complètement arrêtée$(NC)"

stop-streamlit: ## Arrêter Streamlit
	@echo "$(YELLOW)📊 Arrêt de Streamlit...$(NC)"
	@pkill -f "streamlit run" || true
	@echo "$(GREEN)✅ Streamlit arrêté$(NC)"

# stop-orchestrator supprimé - non nécessaire

stop-prefect: ## Arrêter Prefect (serveur + worker)
	@echo "$(YELLOW)🚀 Arrêt de Prefect...$(NC)"
	@pkill -f "prefect server" || true
	@pkill -f "prefect worker" || true
	@echo "$(GREEN)✅ Prefect arrêté$(NC)"

stop-ollama: ## Arrêter Ollama
	@echo "$(YELLOW)🧠 Arrêt d'Ollama...$(NC)"
	@pkill -f "ollama serve" || true
	@echo "$(GREEN)✅ Ollama arrêté$(NC)"

restart: ## Redémarrer l'application
	@echo "$(YELLOW)🔄 Redémarrage de Sentinel2...$(NC)"
	@make stop
	@sleep 2
	@make start
	@echo "$(GREEN)✅ Application redémarrée$(NC)"

status: ## Vérifier le statut de l'application COMPLÈTE
	@echo "$(YELLOW)📊 Statut de Sentinel2:$(NC)"
	@echo ""
	@echo "$(YELLOW)Prefect Server:$(NC)"
	@if lsof -i :4200 > /dev/null 2>&1; then \
		echo "  $(GREEN)✅ En cours d'exécution$(NC)"; \
		echo "  $(GREEN)   Dashboard: http://localhost:4200$(NC)"; \
	else \
		echo "  $(RED)❌ Arrêté$(NC)"; \
	fi
	@echo ""
	@echo "$(YELLOW)Prefect Worker:$(NC)"
	@if pgrep -f "prefect worker" > /dev/null; then \
		echo "  $(GREEN)✅ En cours d'exécution$(NC)"; \
	else \
		echo "  $(RED)❌ Arrêté (REQUIS pour exécuter flows!)$(NC)"; \
	fi
	@echo ""
	@echo "$(YELLOW)Streamlit:$(NC)"
	@if pgrep -f "streamlit run" > /dev/null; then \
		echo "  $(GREEN)✅ En cours d'exécution$(NC)"; \
		echo "  $(GREEN)   URL: http://localhost:$(STREAMLIT_PORT)$(NC)"; \
	else \
		echo "  $(RED)❌ Arrêté$(NC)"; \
	fi
	@echo ""
	@echo "$(YELLOW)Ollama:$(NC)"
	@if pgrep -f "ollama serve" > /dev/null; then \
		echo "  $(GREEN)✅ En cours d'exécution$(NC)"; \
		echo "  $(GREEN)   Port: $(OLLAMA_PORT)$(NC)"; \
	else \
		echo "  $(RED)❌ Arrêté$(NC)"; \
	fi

logs: ## Afficher les logs Prefect Worker
	@echo "$(YELLOW)📋 Logs Prefect Worker:$(NC)"
	@if [ -f "data/logs/prefect_worker.log" ]; then \
		tail -f data/logs/prefect_worker.log; \
	else \
		echo "$(RED)❌ Aucun fichier de log trouvé$(NC)"; \
	fi

clean: ## Nettoyer les caches et fichiers temporaires
	@echo "$(YELLOW)🧹 Nettoyage de Sentinel2...$(NC)"
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf .pytest_cache 2>/dev/null || true
	@echo "$(GREEN)✅ Nettoyage terminé$(NC)"

clean-all: clean ## Nettoyage complet (supprime .venv)
	@echo "$(YELLOW)🧹 Nettoyage complet...$(NC)"
	@rm -rf $(VENV_DIR) 2>/dev/null || true
	@echo "$(GREEN)✅ Nettoyage complet terminé$(NC)"

clean-logs: ## Nettoyer les logs et décisions (INTERACTIF)
	@echo "$(YELLOW)🧹 Nettoyage des logs et décisions...$(NC)"
	@bash scripts/clean_logs.sh

dev: ## Mode développement (sans Ollama)
	@echo "$(YELLOW)🔧 Mode développement...$(NC)"
	@make stop
	@make start-streamlit
	@echo "$(GREEN)✅ Mode développement activé$(NC)"

prod: ## Mode production (avec Ollama)
	@echo "$(YELLOW)🚀 Mode production...$(NC)"
	@make start
	@echo "$(GREEN)✅ Mode production activé$(NC)"

prefect-deploy: ## Déployer les flows Prefect depuis prefect.yaml
	@echo "$(YELLOW)🚀 Déploiement flows depuis prefect.yaml...$(NC)"
	@PREFECT_API_URL=http://localhost:4200/api uv run prefect work-pool create sentinel --type process || echo "$(YELLOW)⚠️ Work pool déjà créé$(NC)"
	@PREFECT_API_URL=http://localhost:4200/api uv run prefect deploy --all
	@echo "$(GREEN)✅ Déploiements créés$(NC)"

prefect-worker: ## Démarrer Prefect worker pour le pool sentinel
	@echo "$(YELLOW)🤖 Démarrage worker Prefect (pool sentinel)...$(NC)"
	@PREFECT_API_URL=http://localhost:4200/api uv run prefect worker start --pool sentinel

prefect-ui: ## Ouvrir Prefect UI
	@echo "$(YELLOW)📊 Ouverture Prefect UI...$(NC)"
	@command -v open >/dev/null 2>&1 && open http://localhost:4200 || (command -v xdg-open >/dev/null 2>&1 && xdg-open http://localhost:4200) || echo "Ouvrir manuellement: http://localhost:4200"

test: ## Lancer les tests
	@echo "$(YELLOW)🧪 Lancement des tests...$(NC)"
	uv run python -m pytest tests/ -v
	@echo "$(GREEN)✅ Tests terminés$(NC)"

check-prod: ## Vérifier configuration production
	@bash scripts/check_production.sh

check-all: ## Vérification complète (services + logs + erreurs)
	@echo "$(YELLOW)🔍 VÉRIFICATION COMPLÈTE SENTINEL2$(NC)"
	@echo "========================================="
	@echo ""
	@echo "1️⃣ STATUT SERVICES"
	@make status
	@echo ""
	@echo "2️⃣ LOGS RÉCENTS (dernières erreurs)"
	@echo "-----------------------------------"
	@if [ -f "data/logs/prefect_worker.log" ]; then \
		echo "$(YELLOW)Prefect Worker:$(NC)"; \
		grep -i "error\|exception\|failed" data/logs/prefect_worker.log | tail -5 || echo "  $(GREEN)✅ Pas d'erreur$(NC)"; \
	fi
	@echo ""
	@echo ""
	@if [ -f "data/logs/trading_decisions.log" ]; then \
		echo "$(YELLOW)Trading:$(NC)"; \
		grep -i "error\|exception" data/logs/trading_decisions.log | tail -5 || echo "  $(GREEN)✅ Pas d'erreur$(NC)"; \
	fi
	@echo ""
	@echo "3️⃣ FLOWS PREFECT RÉCENTS"
	@echo "------------------------"
	@uv run prefect flow-run ls --limit 3 2>/dev/null || echo "  $(RED)❌ Impossible de lister les flows$(NC)"
	@echo ""
	@echo "========================================="
	@echo "$(GREEN)✅ Vérification terminée$(NC)"

# Commande par défaut
.DEFAULT_GOAL := help
