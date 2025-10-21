"""
Interface Streamlit principale - Sentinel Trading
Structure conforme aux bonnes pratiques officielles Streamlit
"""

from pathlib import Path

import streamlit as st

# Configuration des logs AVANT tout import
from src.gui.config.logging_config import setup_logging
from src.constants import CONSTANTS

setup_logging()

# Imports au niveau du module (bonnes pratiques officielles)
from src.gui.pages.analysis_page import show_analysis_page
from src.gui.pages.logs_page import show_logs_page
from src.gui.pages.production_page import show_production_page


def inject_css() -> None:
    """Injection contrôlée du CSS centralisé (bonnes pratiques officielles)"""
    css_path = Path(__file__).parent / "assets" / "custom.css"
    if css_path.exists():
        css_content = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def main():
    """Fonction principale selon les bonnes pratiques officielles Streamlit"""

    # Configuration de la page au tout début (Get started + API reference)
    st.set_page_config(
        page_title=CONSTANTS.STREAMLIT_PAGE_TITLE,
        page_icon=CONSTANTS.STREAMLIT_PAGE_ICON,
        layout=CONSTANTS.STREAMLIT_LAYOUT,
        initial_sidebar_state="expanded",
    )

    # Injection du CSS centralisé
    inject_css()

    # Header principal
    st.markdown(
        """
    <div class="main-header">
        <h1>🚀 Sentinel42</h1>
        <p>Système d'analyse et de prédiction des marchés financiers</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Onglets avec Default tab (bonnes pratiques officielles)
    tabs = st.tabs(["📊 Analysis", "🚀 Production", "📋 Logs"])

    with tabs[0]:
        show_analysis_page()

    with tabs[1]:
        show_production_page()

    with tabs[2]:
        show_logs_page()


# Appel explicite de la fonction principale (DCO recommandation 1)
if __name__ == "__main__":
    main()
