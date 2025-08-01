import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import load
from .core import BaseStrategy
from src.features.technical import calculate_indicators

class HybridStrategy(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.model = load(config['model_path'])
        self.scaler = load(config['scaler_path'])
    
    def generate_signal(self, data):
        # Calcul des indicateurs
        data = calculate_indicators(data)
        
        # Préparation des features
        features = data[[
            'RSI', 'MACD', 'Signal', 'UpperBand', 'LowerBand',
            'ATR', 'EMA50', 'EMA200', 'VolMA20'
        ]].iloc[[-1]]
        
        # Normalisation
        features_scaled = self.scaler.transform(features)
        
        # Prédiction ML
        proba = self.model.predict_proba(features_scaled)[0][1]
        prediction = 1 if proba > 0.65 else 0
        
        # Règles de décision
        if data['RSI'].iloc[-1] < 30 and prediction == 1:
            return 'BUY'
        elif data['RSI'].iloc[-1] > 70 or prediction == 0:
            return 'SELL'
        return 'HOLD'