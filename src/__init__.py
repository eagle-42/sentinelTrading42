"""
Sentinel42 - Système de Trading Algorithmique

"""

from .constants import CONSTANTS, SentinelConstants

__version__ = "2.0.0"
__author__ = "Sentinel Team"
__description__ = "Trading Algorithmique TDD avec Fusion Adaptative Prix/Sentiment"

# Exports principaux
__all__ = ["CONSTANTS", "SentinelConstants", "__version__", "__author__", "__description__"]

# Configuration initiale
CONSTANTS.ensure_directories()
