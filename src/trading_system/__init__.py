"""
Trading System - Système de trading algorithmique hybride
Copyright (c) 2024 Votre Nom. Licence MIT.
"""
from pathlib import Path
import sys
if sys.version_info >= (3, 9):
    from importlib.resources import files
else:
    from importlib_resources import files

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"

# Méthode moderne (Python ≥3.9)
_module = __name__
_config_path = files(_module) / '../../config'
if not _config_path.is_dir():
    raise FileNotFoundError
config_path = Path(_config_path)


__all__ = ["__version__", "config_path"]