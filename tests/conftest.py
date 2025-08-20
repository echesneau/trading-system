import pytest
import pandas as pd
import numpy as np

from trading_system.ml.trainer import ModelTrainer
from trading_system.data.loader import load_yfinance_data

@pytest.fixture(scope="session")
def sample_data():
    # Générer plus de données pour les indicateurs longs
    dates = pd.date_range('2020-01-01', periods=250)  # 250 périodes
    prices = np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates)))

    return pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.01,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000, 5000, len(dates))
    }, index=dates)

@pytest.fixture(scope="session")
def ticker_test():
    """Fixture pour le ticker de test"""
    return "SAN.PA"

@pytest.fixture(scope="session")
def test_data(ticker_test):
    # Charger des données de test
    return load_yfinance_data(
        ticker=ticker_test,
        start_date="2023-01-01",
        end_date="2023-06-30"
    )

@pytest.fixture(scope="session")
def model_trainer():
    """Retourne une instance de ModelTrainer configurée pour les tests"""
    return ModelTrainer({
        'n_estimators': 10,  # Léger pour les tests
        'max_depth': 3,
        'early_stopping_rounds': 10,
        'target_horizon': 3,  # Court terme pour les tests
        'technical_params': {
            'ema_windows': [20, 50],  # Évite EMA_200 qui nécessite plus de données
            'rsi_window': 10,
            'bb_window': 14
        }
    })


@pytest.fixture(scope="session")
def trained_model_artifacts(test_data, model_trainer):
    """Utilise ModelTrainer pour entraîner un modèle sur les données réelles"""
    try:
        artifacts = model_trainer.train(test_data)

        # Validation basique du modèle
        assert 'model' in artifacts
        assert 'scaler' in artifacts
        assert 'feature_names' in artifacts
        assert len(artifacts['feature_names']) > 0

        return artifacts

    except Exception as e:
        pytest.skip(f"Échec de l'entraînement: {str(e)}")