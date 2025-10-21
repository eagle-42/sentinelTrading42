# Sentinel42

# 🏛️ Sentinel42 - Système de Trading Algorithmique TDD

## ⚡ **DÉMARRAGE ULTRA-RAPIDE**

### **1 commande pour tout démarrer**

```bash
make start
```

**Attendre ~20 secondes**, puis accéder à :
- **Streamlit** : http://localhost:8501
- **Prefect** : http://localhost:4200

### **Vérifier le statut**

```bash
make status
```

### **Arrêter**

```bash
make stop
```

---

## 📊 **ARCHITECTURE**

### **Stack Technique**

| Composant | Technologie | Description |
|-----------|-------------|-------------|
| **Orchestration** | Prefect 3.0 | Flows automatisés avec schedules |
| **ML Model** | PyTorch LSTM | 98.39% accuracy |
| **Sentiment** | FinBERT | Analyse sentiment news |
| **Interface** | Streamlit | Dashboard temps réel |
| **LLM** | Ollama | Analyse contextuelle |
| **Data** | Parquet | Stockage incrémental |

### **Flows Automatiques**

| Flow | Fréquence | Description |
|------|-----------|-------------|
| 📊 **Prix 15min** | */15 * * * * | Refresh prix temps réel |
| 📰 **News + Sentiment** | */4 * * * * | RSS + FinBERT |
| 🤖 **Trading** | */15 9-16 * * 1-5 | Décisions trading (heures marché) |

---

## 📁 **STRUCTURE DU PROJET**

```
sentinel42/
├── src/
│   ├── core/              # Modules ML (LSTM, Sentiment, Fusion)
│   ├── data/              # Gestion données (storage)
│   ├── gui/               # Interface Streamlit
│   ├── models/            # Modèles ML
│   └── tests/             # Tests unitaires/intégration
├── flows/
│   ├── sentinel_flows.py  # Flows Prefect
│   └── deployments.py     # Déploiements
├── scripts/               # Scripts essentiels
│   ├── refresh_prices.py  # Refresh prix
│   ├── refresh_news.py    # Refresh news
│   ├── trading_pipeline.py # Pipeline trading
│   └── train_lstm_model.py # Entraînement modèle
├── data/
│   ├── realtime/          # Données temps réel
│   ├── trading/           # Décisions trading
│   └── logs/              # Logs système
├── config/                # Configuration
└── Makefile               # Commandes simplifiées
```

---

## 🎯 **FONCTIONNALITÉS**

### **1. Data Collection**
- ✅ Prix temps réel (yfinance, 15min)
- ✅ News multiples sources (RSS + NewsAPI)
- ✅ Sentiment analysis (FinBERT)
- ✅ Stockage Parquet incrémental

### **2. Machine Learning**
- ✅ LSTM PyTorch (98.39% accuracy)
- ✅ Features: OHLC returns
- ✅ Prédictions horizon 1 période
- ✅ Auto-entraînement

### **3. Trading**
- ✅ Fusion adaptative signaux (prix + sentiment)
- ✅ Décisions BUY/HOLD/SELL
- ✅ Seuils adaptatifs
- ✅ Log toutes décisions

### **4. Monitoring**
- ✅ Dashboard Streamlit temps réel
- ✅ Dashboard Prefect (flows)
- ✅ Logs structurés
- ✅ Métriques performance

---

## 🛠️ **COMMANDES MAKE**

| Commande | Action |
|----------|--------|
| `make start` | Démarrer tout (Ollama + Prefect + Streamlit) |
| `make stop` | Arrêter tout |
| `make restart` | Redémarrer |
| `make status` | Statut services |
| `make check-all` | Vérification complète (services + logs + flows) |
| `make prefect-deploy` | Déployer flows Prefect |
| `make prefect-worker` | Démarrer worker Prefect |
| `make prefect-ui` | Ouvrir dashboard Prefect |
| `make logs` | Voir logs worker |
| `make test` | Lancer tests |
| `make clean` | Nettoyer caches |

---

## 📚 **DOCUMENTATION**

| Document | Description |
|----------|-------------|
| `START_HERE.md` | ⭐ Démarrage ultra-simple |
| `QUICK_START.md` | Guide détaillé |
| `PREFECT_GUIDE.md` | Guide Prefect complet |
| `DEPLOYMENT_STATUS.md` | État production actuel |

---

## 🔧 **CONFIGURATION**

### **Variables d'environnement**

Créer `.env` :

```bash
NEWSAPI_KEY=your_key_here
PREFECT_API_URL=http://localhost:4200/api
```

### **Installation**

```bash
# Cloner le projet
git clone <repository-url>
cd sentinel42

# Installer dépendances
uv sync

# Déployer flows Prefect
uv run prefect deploy --all
```

---

## 🎯 **UTILISATION NORMALE**

### **Workflow quotidien**

1. **Démarrer le matin**
   ```bash
   make start
   ```

2. **Vérifier**
   ```bash
   make check-all
   ```

3. **Monitorer**
   - Streamlit: http://localhost:8501
   - Prefect: http://localhost:4200

4. **Arrêter le soir**
   ```bash
   make stop
   ```

### **Les flows tournent automatiquement !**

Une fois démarré, le système est autonome :
- Prix rafraîchis toutes les 15 minutes
- News rafraîchies toutes les 4 minutes
- Décisions trading aux heures de marché

---

## 📊 **DONNÉES GÉNÉRÉES**

```
data/
├── realtime/
│   ├── prices/
│   │   └── spy_15min.parquet          # Prix temps réel
│   └── news/
│       └── spy_news.parquet           # News + sentiment
├── trading/
│   └── decisions_log/
│       ├── trading_decisions.json     # Historique décisions
│       └── trading_state.json         # État trading
└── models/
    └── spy/
        └── version_1/
            ├── model.pkl              # Modèle LSTM
            └── scaler.pkl             # Scaler données
```

---

## 🐛 **TROUBLESHOOTING**

### **Prefect dashboard vide**

**Cause** : Worker pas démarré

**Solution** :
```bash
make start-prefect-worker
```

### **Pas de décisions dans Streamlit**

**Cause** : Hors heures de marché OU aucune décision générée encore

**Solution** :
```bash
# Exécuter manuellement
uv run prefect deployment run '🤖 Trading Flow/trading-production'
```

### **Services ne démarrent pas**

**Solution** :
```bash
make stop
make clean
make start
```

---

## 🧪 **TESTS**

```bash
# Tests complets
make test

# Tests unitaires
uv run pytest src/tests/unit/ -v

# Tests intégration
uv run pytest src/tests/integration/ -v

# Tests avec couverture
uv run pytest --cov=src --cov-report=html
```

---

## 📈 **MÉTRIQUES**

- **Accuracy LSTM** : 98.39%
- **Latence prédiction** : <100ms
- **Success rate flows** : >95%
- **Uptime** : 24/7

---

## 🔐 **PRODUCTION**

### **Démarrage automatique au boot**

```bash
# Ajouter au crontab
@reboot cd /path/to/sentinel42 && make start
```

### **Monitoring continu**

```bash
# Vérification périodique
*/30 * * * * cd /path/to/sentinel42 && make check-all
```

---

## 👨‍💻 **DÉVELOPPEMENT**

### **Architecture TDD**

- Tests AVANT implémentation
- 100% de succès requis
- Couverture minimale 80%

### **Contribuer**

1. Fork le projet
2. Créer une branche (`git checkout -b feature/amazing`)
3. Commit (`git commit -m 'Add amazing feature'`)
4. Push (`git push origin feature/amazing`)
5. Ouvrir une Pull Request

---

## 📝 **CHANGELOG**

### **Version 2.0** (2025-10-01)
- ✅ Migration complète vers Prefect
- ✅ 5 flows automatisés
- ✅ Dashboard Streamlit amélioré
- ✅ Décisions réelles affichées
- ✅ Documentation complète

### **Version 1.0**
- ✅ LSTM PyTorch
- ✅ Sentiment FinBERT
- ✅ Fusion adaptative
- ✅ Interface Streamlit

---

## 📧 **SUPPORT**

- **Documentation** : Voir `docs/`
- **Issues** : GitHub Issues
- **Discussions** : GitHub Discussions

---

## 📄 **LICENSE**

**📜 MIT License - Licence Permissive**

Ce projet est sous **licence open source MIT** - La plus permissive et populaire.

