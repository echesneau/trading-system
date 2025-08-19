# tests/unit/test_trainer.py
import pytest
from src.ml.trainer import ModelTrainer
import pandas as pd
import numpy as np


@pytest.fixture
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


def test_trainer_initialization():
    trainer = ModelTrainer({'n_estimators': 50})
    assert trainer.config['n_estimators'] == 50
    assert trainer.config['target_horizon'] == 5


def test_full_training(sample_data):
    # Réduire les fenêtres maximales pour les tests
    config = {
        'max_depth': 2,
        'ema_windows': [10, 20],  # Au lieu de [20, 50, 200]
        'bb_window': 10,  # Au lieu de 20
        'rsi_window': 10  # Au lieu de 14
    }
    trainer = ModelTrainer(config)
    results = trainer.train(sample_data)

    assert 'model' in results
    assert 'scaler' in results
    assert isinstance(results['feature_names'], list)
    assert len(results['feature_names']) > 0
    assert 0 <= results['test_score'] <= 1