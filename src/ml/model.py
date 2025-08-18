# src/ml/model.py
import joblib
from typing import Tuple

def load_model(path: str) -> Tuple[any, any]:
    """Charge le modèle et le scaler depuis un fichier joblib"""
    try:
        loaded = joblib.load(path)
        return loaded['model'], loaded['scaler']
    except Exception as e:
        raise RuntimeError(f"Erreur de chargement du modèle {path}: {str(e)}")