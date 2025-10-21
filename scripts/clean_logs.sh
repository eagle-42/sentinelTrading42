#!/bin/bash
# 🧹 Script de nettoyage complet des logs et décisions Sentinel2

echo "🧹 NETTOYAGE COMPLET SENTINEL2"
echo "=============================="

# Demander confirmation
read -p "⚠️  Voulez-vous vraiment supprimer tous les logs et décisions ? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "❌ Nettoyage annulé"
    exit 0
fi

echo ""
echo "🗑️  Suppression des données..."

# Supprimer anciennes décisions
echo "  - Décisions de trading..."
rm -f data/trading/decisions_log/*.json
rm -f data/trading/decisions_log/*.log

# Supprimer validations
echo "  - Validations historiques..."
rm -f data/trading/historical_validation/*.json

# Supprimer logs de validation
echo "  - Logs de validation..."
rm -f data/trading/validation_log/*.json
rm -f data/trading/verification_log/*.json

# Nettoyer les logs principaux (garder fichiers vides)
echo "  - Logs système..."
> data/logs/errors.log
> data/logs/sentinel_main.log
> data/logs/trading_decisions.log

# Optionnel: Synthèses LLM (commenté par défaut)
# echo "  - Synthèses LLM..."
# rm -f data/trading/llm_synthesis/*.json

echo ""
echo "✅ Nettoyage terminé !"
echo ""
echo "📊 Fichiers conservés:"
echo "  - Données historiques (data/historical/)"
echo "  - Données temps réel (data/realtime/)"
echo "  - Modèles LSTM (data/models/)"
echo "  - Synthèses LLM (data/trading/llm_synthesis/)"
echo ""
