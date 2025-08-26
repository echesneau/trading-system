# src/ml/model.py
import joblib
from typing import Dict

def load_model(path: str) -> Dict:
    """Charge le modèle et le scaler depuis un fichier joblib"""
    try:
        loaded = joblib.load(path)
        return loaded
    except Exception as e:
        raise RuntimeError(f"Erreur de chargement du modèle {path}: {str(e)}")