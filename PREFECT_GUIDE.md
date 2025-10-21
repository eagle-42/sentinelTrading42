# 🚀 Guide Prefect pour Sentinel42

## 📋 **Vue d'ensemble**

Prefect orchestre tous les workflows de Sentinel42 avec monitoring visuel, retry automatique, et alertes.

---

## 🏗️ **Architecture**

```
┌─────────────────────────────────────────┐
│       Prefect Server (Port 4200)        │
│         Dashboard & Orchestration        │
└─────────────────────────────────────────┘
                    │
                    ├─── Work Pool: sentinel-pool
                    │
        ┌───────────┼───────────┐
        │           │           │
    ┌───▼───┐   ┌──▼───┐   ┌──▼────┐
    │ Data  │   │Trading│   │Validate│
    │Refresh│   │ Flow  │   │  Flow  │
    └───────┘   └───────┘   └────────┘
```

---

## 🎯 **Flows Configurés**

### **1. Data Refresh Flow** 🔄
- **Fréquence** : Toutes les 4 minutes
- **Actions** :
  - Rafraîchit prix 15min (yfinance)
  - Rafraîchit news (RSS + NewsAPI)
  - Calcule sentiment (FinBERT)
- **Retry** : 2 fois, délai 30s

### **2. Trading Flow** 🤖
- **Fréquence** : Toutes les 15 minutes (9h-16h ET, Lun-Ven)
- **Actions** :
  - Exécute Data Refresh
  - Génère prédiction LSTM
  - Fusionne signaux
  - Crée décision (BUY/SELL/HOLD)
- **Retry** : 1 fois, délai 60s

### **3. Full System Flow** 📊
- **Fréquence** : Manuel (démarrage)
- **Actions** :
  - Refresh initial complet
  - Première décision forcée
  - Initialisation système

---

## 🚀 **Démarrage**

### **Méthode 1 : Makefile (recommandé)**
```bash
# Démarrer Prefect
make prefect-start

# Ouvrir dashboard
make prefect-ui

# Arrêter Prefect
make prefect-stop
```

### **Méthode 2 : Script direct**
```bash
# Démarrer
bash scripts/start_prefect.sh

# Dashboard
open http://localhost:4200
```

### **Méthode 3 : Commandes manuelles**
```bash
# 1. Serveur
prefect server start

# 2. Work pool (autre terminal)
prefect work-pool create sentinel-pool --type process

# 3. Déployer flows (autre terminal)
cd flows
uv run python deployments.py

# 4. Worker
prefect worker start --pool sentinel-pool
```

---

## 📊 **Dashboard Prefect**

### **Accès**
- URL : http://localhost:4200
- Aucun login requis (local)

### **Sections principales**

**1. Flows** : Liste des workflows
**2. Flow Runs** : Historique des exécutions
**3. Work Pools** : Pools de workers
**4. Deployments** : Déploiements actifs
**5. Logs** : Logs en temps réel

### **Visualisations**
- ✅ Timeline des exécutions
- ✅ Graphe de dépendances
- ✅ Métriques de performance
- ✅ Retry history
- ✅ Alertes d'échec

---

## 🔧 **Commandes Utiles**

### **Exécution manuelle**
```bash
# Exécuter un flow
prefect deployment run 'data-refresh-flow/data-refresh-production'
prefect deployment run 'trading-flow/trading-production'

# Forcer trading immédiat
prefect deployment run 'trading-flow/trading-production' --param force=true
```

### **Monitoring**
```bash
# Statut serveur
prefect server status

# Liste flows
prefect flow ls

# Liste deployments
prefect deployment ls

# Logs en direct
prefect flow-run logs --follow <run-id>
```

### **Debugging**
```bash
# Test local d'un flow
cd flows
uv run python sentinel_flows.py

# Vérifier configuration
prefect config view
```

---

## 🎨 **Avantages vs schedule.py**

| Feature | schedule.py | Prefect |
|---------|-------------|---------|
| **Dashboard visuel** | ❌ | ✅ |
| **Retry automatique** | ❌ | ✅ |
| **Graphe dépendances** | ❌ | ✅ |
| **Logs centralisés** | ❌ | ✅ |
| **Alertes** | ❌ | ✅ |
| **Parallélisation** | ❌ | ✅ |
| **Historique** | ❌ | ✅ |
| **API REST** | ❌ | ✅ |

---

## 📈 **Monitoring Production**

### **Métriques surveillées**
- ✅ Temps d'exécution par flow
- ✅ Taux de succès/échec
- ✅ Nombre de retry
- ✅ Latence moyenne
- ✅ Erreurs par type

### **Alertes configurées**
- 🔴 Échec après retries
- 🟡 Latence > 5 minutes
- 🟡 Aucune donnée récupérée

---

## 🔒 **Production**

### **Démarrage automatique**
Ajouter au crontab:
```bash
@reboot cd /path/to/sentinel42 && make prefect-start
```

### **Systemd Service** (Linux)
```ini
[Unit]
Description=Sentinel42 Prefect Server
After=network.target

[Service]
Type=simple
User=eagle
WorkingDirectory=/path/to/sentinel42
ExecStart=/path/to/sentinel42/scripts/start_prefect.sh
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## 🐛 **Troubleshooting**

### **Serveur ne démarre pas**
```bash
# Vérifier port 4200
lsof -i :4200

# Tuer processus existant
pkill -f "prefect server"
```

### **Worker ne se connecte pas**
```bash
# Vérifier work pool
prefect work-pool ls

# Recréer
prefect work-pool delete sentinel-pool
prefect work-pool create sentinel-pool --type process
```

### **Flow échoue**
```bash
# Voir logs détaillés
prefect flow-run logs <run-id>

# Test local
cd flows
uv run python sentinel_flows.py
```

---

## 📚 **Ressources**

- **Documentation** : https://docs.prefect.io
- **Communauté** : https://discourse.prefect.io
- **GitHub** : https://github.com/PrefectHQ/prefect

---

## 🎯 **Prochaines étapes**

1. ✅ Ajouter notifications (Slack/Email)
2. ✅ Métriques Prometheus
3. ✅ Alertes avancées
4. ✅ Backup automatique état

**Prefect est maintenant opérationnel !** 🚀
