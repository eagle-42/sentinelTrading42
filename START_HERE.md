# 🚀 START HERE - Sentinel42

> **Guide de démarrage ultra-rapide** - Lancer le système en 1 commande

## ⚡ **Démarrage en 1 commande**

```bash
make start
```

**Services démarrés automatiquement** :
- ✅ Ollama (LLM local)
- ✅ Prefect Server (orchestration)
- ✅ Prefect Worker (exécution flows)
- ✅ Streamlit (interface web)

**Attendre ~20 secondes** que tous les services démarrent.

---

## **✅ Vérifier que tout tourne**

```bash
make status
```

**Output attendu** :
```
📊 Statut de Sentinel42:

Prefect Server:
  ✅ En cours d'exécution
   Dashboard: http://localhost:4200

Prefect Worker:
  ✅ En cours d'exécution

Streamlit:
  ✅ En cours d'exécution
   URL: http://localhost:8501

Ollama:
  ✅ En cours d'exécution
   Port: 11434
```

---

## **📊 Accès aux interfaces**

### **Streamlit (Interface principale)**
```
http://localhost:8501
```

### **Prefect (Dashboard orchestration)**
```
http://localhost:4200
```

---

## **🔄 Flows automatiques**

Les flows Prefect s'exécutent automatiquement selon leurs schedules :

| Flow | Fréquence | Description |
|------|-----------|-------------|
| **📊 Prix 15min** | */15 minutes | Refresh prix SPY temps réel |
| **📰 News + Sentiment** | */4 minutes | RSS + analyse FinBERT |
| **🤖 Trading** | */15 min (9h-16h) | Décisions trading |
| **📈 Historical** | 1x/jour (16h30 ET) | Mise à jour historique |

---

## **🐛 Troubleshooting**

### **❌ Worker Prefect pas actif**
```bash
make status              # Vérifier statut
make start-prefect-worker  # Démarrer worker si besoin
```

### **❌ Pas de données/décisions**
```bash
# Les flows peuvent prendre quelques minutes
# Vérifier sur le dashboard Prefect: http://localhost:4200
```

### **❌ Erreur au démarrage**
```bash
make stop    # Tout arrêter
make start   # Redémarrer proprement
```

---

## **🛑 Arrêter l'application**

```bash
make stop
```

---

## **🔧 Commandes principales**

```bash
make start        # Démarrer tout
make stop         # Arrêter tout
make restart      # Redémarrer
make status       # Vérifier statut
make check-prod   # Check production
```

---

## **📚 Documentation**

- **[README.md](README.md)** → Architecture & Features complètes
- **[PREFECT_GUIDE.md](PREFECT_GUIDE.md)** → Guide Prefect détaillé
- **[ARCHITECTURE_PRINCIPLES.md](docs/ARCHITECTURE_PRINCIPLES.md)** → Principes de développement

---

**🎉 Sentinel42 est maintenant opérationnel !**
