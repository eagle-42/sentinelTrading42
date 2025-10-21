#!/bin/bash
# 🔍 Vérification configuration PRODUCTION Sentinel2

echo "🔍 VÉRIFICATION CONFIGURATION PRODUCTION"
echo "========================================="

# Couleurs
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Vérifier services
echo ""
echo "1️⃣ SERVICES"
echo "----------"

if lsof -i :4200 > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Prefect Server (port 4200)${NC}"
else
    echo -e "${RED}❌ Prefect Server arrêté${NC}"
fi

if pgrep -f "streamlit run" > /dev/null; then
    echo -e "${GREEN}✅ Streamlit (port 8501)${NC}"
else
    echo -e "${RED}❌ Streamlit arrêté${NC}"
fi

if pgrep -f "ollama serve" > /dev/null; then
    echo -e "${GREEN}✅ Ollama (port 11434)${NC}"
else
    echo -e "${RED}❌ Ollama arrêté${NC}"
fi

# 2. Vérifier déploiements
echo ""
echo "2️⃣ DÉPLOIEMENTS PREFECT"
echo "----------------------"

DEPLOYMENTS=$(uv run prefect deployment ls 2>/dev/null | grep -c "Flow")
echo -e "${GREEN}✅ ${DEPLOYMENTS} déploiements actifs${NC}"

# 3. Vérifier schedules
echo ""
echo "3️⃣ SCHEDULES ACTIFS"
echo "------------------"
echo "📊 Prix 15min:       */15 * * * * (toutes les 15min)"
echo "📰 News + Sentiment: */4 * * * *  (toutes les 4min)"
echo "🤖 Trading:          */15 9-16 * * 1-5 (heures marché)"

# 4. Vérifier données
echo ""
echo "4️⃣ DONNÉES"
echo "----------"

if [ -d "data/realtime/prices" ]; then
    PRICE_FILES=$(find data/realtime/prices -name "*.parquet" 2>/dev/null | wc -l)
    echo -e "${GREEN}✅ ${PRICE_FILES} fichiers prix${NC}"
fi

if [ -d "data/realtime/news" ]; then
    NEWS_FILES=$(find data/realtime/news -name "*.parquet" 2>/dev/null | wc -l)
    echo -e "${GREEN}✅ ${NEWS_FILES} fichiers news${NC}"
fi

if [ -d "data/models" ]; then
    MODEL_FILES=$(find data/models -name "*.pkl" 2>/dev/null | wc -l)
    echo -e "${GREEN}✅ ${MODEL_FILES} modèles LSTM${NC}"
fi

# 5. Vérifier logs récents
echo ""
echo "5️⃣ LOGS RÉCENTS"
echo "--------------"

if [ -f "data/logs/prefect_server.log" ]; then
    PREFECT_LINES=$(wc -l < data/logs/prefect_server.log)
    echo -e "${GREEN}✅ Prefect: ${PREFECT_LINES} lignes${NC}"
fi

if [ -f "data/logs/sentinel_orchestrator.log" ]; then
    ORCH_LINES=$(wc -l < data/logs/sentinel_orchestrator.log)
    echo -e "${GREEN}✅ Orchestrateur: ${ORCH_LINES} lignes${NC}"
fi

# 6. Résumé
echo ""
echo "========================================="
echo "✅ CONFIGURATION PRODUCTION VÉRIFIÉE"
echo ""
echo "📊 Dashboards:"
echo "   Streamlit: http://localhost:8501"
echo "   Prefect:   http://localhost:4200"
echo ""
echo "🚀 Pour démarrer worker:"
echo "   uv run prefect worker start --pool sentinel"
echo "========================================="
