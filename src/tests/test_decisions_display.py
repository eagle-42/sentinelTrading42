#!/usr/bin/env python3
"""
Test de vérification de l'affichage des décisions dans Streamlit
"""

import sys
from pathlib import Path
import json
import pandas as pd

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_decisions_file():
    """Vérifie que le fichier de décisions existe et est valide"""
    decisions_file = Path("data/trading/decisions_log/trading_decisions.json")
    
    print("🔍 VÉRIFICATION DES DÉCISIONS")
    print("=" * 60)
    
    # Test: fichier existe
    if not decisions_file.exists():
        print("⚠️ Fichier de décisions introuvable (normal si pas encore de trading)")
        # Ne pas fail si le fichier n'existe pas (OK pour tests)
        assert True
        return
    
    print(f"✅ Fichier trouvé: {decisions_file}")
    
    # Test: JSON valide
    with open(decisions_file, 'r') as f:
        decisions = json.load(f)
    
    print(f"✅ Fichier JSON valide")
    print(f"📊 Nombre de décisions: {len(decisions)}")
    print()
    
    # Vérifier structure
    assert isinstance(decisions, list), "Décisions doit être une liste"
    
    if decisions:
        print("📋 DERNIÈRES DÉCISIONS:")
        print("-" * 60)
        for i, decision in enumerate(decisions[-5:], 1):  # 5 dernières
            print(f"\n{i}. Décision {decision.get('ticker', 'N/A')}")
            print(f"   Heure: {decision.get('timestamp', 'N/A')}")
            print(f"   Décision: {decision.get('decision', 'N/A')}")
            print(f"   Confiance: {decision.get('confidence', 0)*100:.1f}%")
            print(f"   Signal fusionné: {decision.get('fused_signal', 0):.4f}")
            
            # Vérifier structure décision
            assert 'decision' in decision, "Clé 'decision' manquante"
    else:
        print("⚠️ Aucune décision enregistrée")

def test_service_loading():
    """Teste le service de validation des décisions"""
    print("\n" + "=" * 60)
    print("🔍 TEST SERVICE HISTORICAL VALIDATION")
    print("=" * 60)
    
    from gui.services.decision_validation_service import DecisionValidationService
    import pandas as pd
    
    service = DecisionValidationService()
    print("✅ Service initialisé")
    
    # Test: service existe
    assert service is not None, "Service non initialisé"
    assert hasattr(service, 'get_validation_history'), "Méthode get_validation_history manquante"
    
    # Test: récupérer l'historique (peut être vide)
    history_df = service.get_validation_history("SPY", days=7)
    
    # Test: retourne un DataFrame
    assert isinstance(history_df, pd.DataFrame), "Doit retourner un DataFrame"
    print(f"✅ Méthode get_validation_history fonctionne")
    
    if history_df.empty:
        print("⚠️ Historique vide (normal si pas encore de validations)")
    else:
        print(f"✅ Historique chargé: {len(history_df)} validations")
        print(f"📊 Colonnes: {list(history_df.columns)}")
    
    # Test: stats de validation
    stats = service.get_validation_stats("SPY", days=7)
    assert isinstance(stats, dict), "Stats doivent être un dict"
    print(f"✅ Méthode get_validation_stats fonctionne")
    print(f"📊 Accuracy globale: {stats.get('global_accuracy', 0)*100:.1f}%")

def main():
    """Exécute tous les tests"""
    print()
    print("🚀 TEST AFFICHAGE DÉCISIONS STREAMLIT")
    
    success = True
    
    # Test 1: Fichier de décisions
    try:
        test_decisions_file()
        print("✅ Test fichier: OK")
    except Exception as e:
        print(f"❌ Test fichier: ERREUR - {e}")
        success = False
    
    # Test 2: Service de chargement
    try:
        test_service_loading()
        print("✅ Test service: OK")
    except Exception as e:
        print(f"❌ Test service: ERREUR - {e}")
        success = False
    
    print()
    print("=" * 60)
    print("📊 RÉSUMÉ")
    print("=" * 60)
    
    if success:
        print("✅ TOUT EST OPÉRATIONNEL")
        print()
        print("💡 Pour voir les décisions dans Streamlit:")
        print("   1. Ouvrir http://localhost:8501")
        print("   2. Aller sur la page 'Production'")
        print("   3. Section 'Décisions Récentes - Synthèse'")
    else:
        print("❌ PROBLÈME DÉTECTÉ - Vérifier les logs ci-dessus")

if __name__ == "__main__":
    main()
