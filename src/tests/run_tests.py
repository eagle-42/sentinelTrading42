#!/usr/bin/env python3
"""
🧪 Script d'exécution des tests Sentinel42
Exécute tous les tests avec les bonnes pratiques TDD et pytest
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


class TestRunner:
    """Gestionnaire de tests Sentinel42 avec bonnes pratiques TDD"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.test_dir = self.project_root / "src" / "tests"
        self.coverage_dir = self.project_root / "htmlcov"
        self.coverage_xml = self.project_root / "coverage.xml"

    def setup_environment(self) -> bool:
        """Configure l'environnement de test"""
        print("🔧 Configuration de l'environnement de test...")

        # Variables d'environnement pour les tests
        test_env = {
            "FINBERT_MODE": "stub",
            "NEWSAPI_ENABLED": "false",
            "POLYGON_API_KEY": "test_key",
            "NEWSAPI_KEY": "test_key",
            "TICKERS": "SPY:S&P 500 ETF,NVDA:NVIDIA Corporation",
            "NEWS_FEEDS": "https://example.com/feed1.rss,https://example.com/feed2.rss",
            "PRICE_INTERVAL": "1min",
            "PRICE_PERIOD": "1d",
            "FUSION_MODE": "adaptive",
            "PYTHONPATH": str(self.project_root),
        }

        for key, value in test_env.items():
            os.environ[key] = value

        print("✅ Environnement configuré")
        return True

    def run_diagnostic_tests(self) -> bool:
        """Exécute les tests de diagnostic"""
        print("\n🔍 Tests de diagnostic")
        print("=" * 30)

        tests = [
            ("Imports", self._test_imports),
            ("Constantes", self._test_constants),
            ("Configuration", self._test_configuration),
            ("Modules Core", self._test_core_modules),
            ("Modules Data", self._test_data_modules),
        ]

        for test_name, test_func in tests:
            print(f"1. Test {test_name}...")
            try:
                if test_func():
                    print(f"✅ {test_name} OK")
                else:
                    print(f"❌ {test_name} échoué")
                    return False
            except Exception as e:
                print(f"❌ Erreur {test_name}: {e}")
                return False

        print("✅ Tous les tests de diagnostic réussis")
        return True

    def _test_imports(self) -> bool:
        """Test des imports"""
        try:
            import src.constants
            import src.core.fusion
            import src.core.prediction
            import src.core.sentiment
            import src.data.storage

            return True
        except ImportError:
            return False

    def _test_constants(self) -> bool:
        """Test des constantes"""
        try:
            from src.constants import CONSTANTS

            return CONSTANTS.PROJECT_ROOT.exists()
        except Exception:
            return False

    def _test_configuration(self) -> bool:
        """Test de la configuration"""
        try:
            from src.constants import CONSTANTS

            # Vérifier que les constantes essentielles existent
            return (
                CONSTANTS.DATA_ROOT.exists() and
                len(CONSTANTS.TICKERS) > 0 and
                CONSTANTS.LSTM_SEQUENCE_LENGTH > 0
            )
        except Exception:
            return False

    def _test_core_modules(self) -> bool:
        """Test des modules core"""
        try:
            from src.core.fusion import AdaptiveFusion
            from src.core.prediction import LSTMPredictor
            from src.core.sentiment import SentimentAnalyzer

            # Test d'initialisation
            fusion = AdaptiveFusion()
            predictor = LSTMPredictor("SPY")
            analyzer = SentimentAnalyzer()

            return all([fusion, predictor, analyzer])
        except Exception:
            return False

    def _test_data_modules(self) -> bool:
        """Test des modules data"""
        try:
            from src.data.storage import DataStorage

            # Test d'initialisation
            storage = DataStorage()

            return bool(storage)
        except Exception:
            return False

    def run_unit_tests(self) -> int:
        """Exécute les tests unitaires"""
        print("\n🧪 Tests unitaires")
        print("=" * 30)

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(self.test_dir),
            "-m",
            "unit",
            "-v",
            "--tb=short",
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml",
            "--cov-fail-under=80",
        ]

        return self._run_pytest_command(cmd, "Tests unitaires")

    def run_integration_tests(self) -> int:
        """Exécute les tests d'intégration"""
        print("\n🔗 Tests d'intégration")
        print("=" * 30)

        cmd = [sys.executable, "-m", "pytest", str(self.test_dir), "-m", "integration", "-v", "--tb=short"]

        return self._run_pytest_command(cmd, "Tests d'intégration")

    def run_all_tests(self) -> int:
        """Exécute tous les tests"""
        print("\n🚀 Tests complets")
        print("=" * 30)

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(self.test_dir),
            "-v",
            "--tb=short",
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml",
            "--cov-fail-under=80",
            "--maxfail=5",  # Arrêter après 5 échecs
        ]

        return self._run_pytest_command(cmd, "Tests complets")

    def _run_pytest_command(self, cmd: List[str], test_type: str) -> int:
        """Exécute une commande pytest"""
        print(f"🔧 Commande: {' '.join(cmd)}")
        print("=" * 50)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)

            print("STDOUT:")
            print(result.stdout)

            if result.stderr:
                print("\nSTDERR:")
                print(result.stderr)

            print(f"\n📊 {test_type} - Code de sortie: {result.returncode}")

            if result.returncode == 0:
                print(f"✅ {test_type} réussis!")
            else:
                print(f"❌ {test_type} échoués")

            return result.returncode

        except Exception as e:
            print(f"❌ Erreur lors de l'exécution des {test_type}: {e}")
            return 1

    def generate_report(self) -> None:
        """Génère un rapport de test"""
        print("\n📊 Rapport de test")
        print("=" * 30)

        if self.coverage_dir.exists():
            print(f"📁 Rapport HTML: {self.coverage_dir}/index.html")

        if self.coverage_xml.exists():
            print(f"📄 Rapport XML: {self.coverage_xml}")

    def run(self, test_type: str = "all") -> int:
        """Point d'entrée principal"""
        print("🧪 Sentinel42 - Tests TDD avec pytest")
        print("=" * 50)

        if not self.setup_environment():
            print("❌ Échec de la configuration")
            return 1

        # Tests de diagnostic
        if not self.run_diagnostic_tests():
            print("\n❌ Les tests de diagnostic ont échoué. Arrêt.")
            return 1

        # Exécution des tests selon le type
        if test_type == "unit":
            exit_code = self.run_unit_tests()
        elif test_type == "integration":
            exit_code = self.run_integration_tests()
        else:  # all
            exit_code = self.run_all_tests()

        # Rapport
        self.generate_report()

        if exit_code == 0:
            print("\n🎉 Tous les tests sont passés avec succès!")
        else:
            print("\n⚠️ Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")

        return exit_code


def main():
    """Fonction principale"""
    import argparse

    parser = argparse.ArgumentParser(description="Exécute les tests Sentinel42")
    parser.add_argument(
        "--type", choices=["unit", "integration", "all"], default="all", help="Type de tests à exécuter"
    )

    args = parser.parse_args()

    runner = TestRunner()
    exit_code = runner.run(args.type)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
