# 🚀 Guide de Démarrage Détaillé - Sentinel42

## Prérequis

- Python 3.12+
- uv package manager
- Compte NewsAPI (optionnel, pour les news enrichies)

## Installation

### 1. Clonage et installation

```bash
# Cloner le projet
git clone <repository-url>
cd sentinel42

# Installer avec uv (recommandé)
uv sync

# Ou avec pip traditionnel
pip install -r requirements.txt
```

### 2. Configuration

Créer le fichier `.env` :

```bash
cp env.example .env
# Éditer .env avec vos clés API
```

Variables importantes :
- `NEWSAPI_KEY` : Pour les news (optionnel)
- `PREFECT_API_URL` : http://localhost:4200/api

## Démarrage

### Démarrage complet

```bash
# Tout démarrer d'un coup
make start

# Ou étape par étape :
make start-ollama       # LLM local
make start-prefect      # Orchestration
make start-streamlit    # Interface web
```

### Vérification

```bash
# Vérifier que tout fonctionne
make status
make check-all
```

## Utilisation quotidienne

### Monitoring

- **Streamlit** : http://localhost:8501 (données temps réel)
- **Prefect** : http://localhost:4200 (flows et logs)

### Logs

```bash
# Voir les logs en temps réel
make logs

# Ou directement
tail -f data/logs/worker.log
```

## Développement

### Tests

```bash
# Tous les tests
make test

# Tests avec couverture
make test-coverage

# Tests spécifiques
uv run pytest src/tests/unit/ -v      # Tests unitaires
uv run pytest src/tests/integration/ -v  # Tests d'intégration
```

### Code quality

```bash
# Formatage automatique
make format

# Linting
make lint

# Type checking
make mypy
```

## Troubleshooting

### Problèmes courants

**Ollama ne démarre pas :**
```bash
# Vérifier l'installation
ollama --version

# Redémarrer le service
make stop-ollama && make start-ollama
```

**Prefect dashboard vide :**
```bash
# Démarrer le worker
make start-prefect-worker

# Vérifier les déploiements
uv run prefect deployment ls
```

**Pas de données dans Streamlit :**
```bash
# Attendre un cycle complet (15min pour les prix)
# Ou déclencher manuellement
uv run prefect deployment run '📊 Prix 15min Flow/prix-production'
```

## Support

- 📚 [Documentation complète](README.md)
- 🐛 [Issues GitHub](https://github.com/sentinel/sentinel42/issues)
- 💬 [Discussions](https://github.com/sentinel/sentinel42/discussions)
