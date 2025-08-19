# src/ml/trainer.py
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np
import joblib
from typing import Tuple, Dict


class ModelTrainer:
    def __init__(self, config: Dict):
        self.config = {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'target_horizon': 5,
            'technical_params': {  # Nouveau: paramètres pour calculate_indicators
                'ema_windows': [20, 50, 200],
                'bb_window': 20,
                'rsi_window': 14,
                'atr_window': 14
            },
            **config
        }
        # Stocker les paramètres techniques séparément
        self.technical_params = self.config['technical_params']

    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prépare les features et targets"""
        # Calcul des indicateurs
        from src.features.technical import calculate_indicators
        # Applique les paramètres techniques
        tech_params = self.config.get('technical_params', {})
        data = calculate_indicators(data, **tech_params)

        # Création de la target
        horizon = self.config['target_horizon']
        data['Target'] = (data['Close'].shift(-horizon) > data['Close']).astype(int)
        data.dropna(inplace=True)

        # Sélection des features
        features = data[[
            'RSI', 'MACD', 'MACD_Signal',
            'BB_Upper', 'BB_Lower', 'ATR',
            'EMA_20', 'EMA_50'
        ]]

        return features, data['Target']

    def train(self, data: pd.DataFrame) -> Dict:
        """Entraîne un modèle XGBoost avec validation temporelle"""
        features, target = self.prepare_features(data)

        # Split temporel
        tscv = TimeSeriesSplit(n_splits=3)
        scaler = StandardScaler()
        model = xgb.XGBClassifier(**{k: v for k, v in self.config.items()
                                     if k in xgb.XGBClassifier().get_params()})

        # Validation croisée
        scores = []
        for train_idx, test_idx in tscv.split(features):
            X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
            y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

            # Normalisation
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Entraînement
            model.fit(X_train_scaled, y_train,
                      eval_set=[(X_test_scaled, y_test)],
                      early_stopping_rounds=10,
                      verbose=False)

            scores.append(model.score(X_test_scaled, y_test))

        # Entraînement final sur toutes les données
        final_scaler = StandardScaler()
        X_final_scaled = final_scaler.fit_transform(features)
        model.fit(X_final_scaled, target)

        return {
            'model': model,
            'scaler': final_scaler,
            'feature_names': features.columns.tolist(),
            'test_score': np.mean(scores),
            'technical_params': self.technical_params,
            'config': self.config
        }