#!/usr/bin/env python3
"""
Script de lancement Streamlit optimisé
Conforme aux bonnes pratiques officielles
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Lance Streamlit avec la configuration optimale"""

    # Chemin vers l'application principale
    app_path = Path(__file__).parent / "main.py"

    # Configuration du serveur
    cmd = [
        "uv",
        "run",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        "8501",
        "--server.address",
        "0.0.0.0",
        "--browser.gatherUsageStats",
        "false",
        "--logger.level",
        "info",
    ]

    print("🚀 Lancement de Sentinel Trading Interface...")
    print(f"📱 Interface disponible sur: http://localhost:8501")
    print(f"🔧 Commande: {' '.join(cmd)}")
    print("=" * 50)

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Arrêt de l'application")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors du lancement: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
