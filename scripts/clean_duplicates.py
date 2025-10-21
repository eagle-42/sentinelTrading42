#!/usr/bin/env python3
"""
Script pour nettoyer les doublons dans les fichiers de données
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

def clean_trading_decisions():
    """Nettoie les doublons dans trading_decisions.json"""
    file_path = Path("data/trading/decisions_log/trading_decisions.json")

    if not file_path.exists():
        print("❌ Fichier trading_decisions.json non trouvé")
        return

    # Charger
    with open(file_path, 'r') as f:
        decisions = json.load(f)

    print(f"📊 Total décisions avant nettoyage: {len(decisions)}")

    # Détecter doublons (même timestamp à 30 secondes près)
    cleaned = []
    seen_timestamps = defaultdict(list)

    for dec in decisions:
        ts = datetime.fromisoformat(dec['timestamp'].replace('Z', '+00:00'))
        # Arrondir à la minute
        ts_rounded = ts.replace(second=0, microsecond=0)

        # Vérifier si on a déjà cette décision (même minute)
        duplicate = False
        for seen_ts in seen_timestamps[ts_rounded]:
            # Si différence < 30 secondes, c'est un doublon
            if abs((ts - seen_ts).total_seconds()) < 30:
                duplicate = True
                break

        if not duplicate:
            cleaned.append(dec)
            seen_timestamps[ts_rounded].append(ts)
        else:
            print(f"   🗑️ Doublon supprimé: {dec['timestamp'][:19]} | {dec['decision']}")

    print(f"✅ Total décisions après nettoyage: {len(cleaned)}")
    print(f"🗑️ Doublons supprimés: {len(decisions) - len(cleaned)}")

    # Sauvegarder
    if len(cleaned) < len(decisions):
        # Backup
        backup_path = file_path.with_suffix('.json.bak')
        with open(backup_path, 'w') as f:
            json.dump(decisions, f, indent=2)
        print(f"💾 Backup sauvegardé: {backup_path}")

        # Sauvegarder nettoyé
        with open(file_path, 'w') as f:
            json.dump(cleaned, f, indent=2)
        print(f"✅ Fichier nettoyé sauvegardé: {file_path}")
    else:
        print("✅ Aucun doublon trouvé")

def clean_validation_history():
    """Nettoie les doublons dans decision_validation_history.parquet"""
    file_path = Path("data/trading/validation_log/decision_validation_history.parquet")

    if not file_path.exists():
        print("\n❌ Fichier decision_validation_history.parquet non trouvé")
        return

    # Charger
    df = pd.read_parquet(file_path)
    print(f"\n📊 Total validations avant nettoyage: {len(df)}")

    # Supprimer doublons exacts
    df_cleaned = df.drop_duplicates(subset=['timestamp', 'ticker', 'decision'], keep='first')

    print(f"✅ Total validations après nettoyage: {len(df_cleaned)}")
    print(f"🗑️ Doublons supprimés: {len(df) - len(df_cleaned)}")

    if len(df_cleaned) < len(df):
        # Backup
        backup_path = file_path.with_suffix('.parquet.bak')
        df.to_parquet(backup_path)
        print(f"💾 Backup sauvegardé: {backup_path}")

        # Sauvegarder nettoyé
        df_cleaned.to_parquet(file_path)
        print(f"✅ Fichier nettoyé sauvegardé: {file_path}")
    else:
        print("✅ Aucun doublon trouvé")

def clean_logs():
    """Nettoie les anciens logs d'erreurs"""
    logs_to_clean = [
        "data/logs/errors.log",
        "data/logs/prefect_server.log",
        "data/logs/prefect_worker.log"
    ]

    print("\n🧹 NETTOYAGE DES LOGS")
    for log_path in logs_to_clean:
        path = Path(log_path)
        if path.exists():
            size_before = path.stat().st_size / 1024  # KB

            # Garder seulement les 1000 dernières lignes
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            if len(lines) > 1000:
                with open(path, 'w', encoding='utf-8') as f:
                    f.writelines(lines[-1000:])

                size_after = path.stat().st_size / 1024  # KB
                print(f"   ✂️ {path.name}: {size_before:.1f} KB → {size_after:.1f} KB ({len(lines)} → 1000 lignes)")
            else:
                print(f"   ✅ {path.name}: {len(lines)} lignes (OK)")

if __name__ == "__main__":
    print("🧹 NETTOYAGE DES DONNÉES\n")
    print("=" * 60)

    # 1. Nettoyer les décisions de trading
    print("\n1️⃣ TRADING DECISIONS")
    clean_trading_decisions()

    # 2. Nettoyer l'historique de validation
    print("\n2️⃣ VALIDATION HISTORY")
    clean_validation_history()

    # 3. Nettoyer les logs
    print("\n3️⃣ LOGS")
    clean_logs()

    print("\n" + "=" * 60)
    print("✅ NETTOYAGE TERMINÉ")
