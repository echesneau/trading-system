# tests/unit/test_trainer.py
import pytest
from src.ml.trainer import ModelTrainer
import pandas as pd
import numpy as np


@pytest.fixture
def sample_data():
    dates = pd.date_range('2020-01-01', periods=100)
    return pd.DataFrame({
        'Open': np.random.uniform(100, 200, 100),
        'High': np.random.uniform(105, 205, 100),
        'Low': np.random.uniform(95, 195, 100),
        'Close': np.random.uniform(100, 200, 100),
        'Volume': np.random.randint(1000, 5000, 100)
    }, index=dates)


def test_trainer_initialization():
    trainer = ModelTrainer({'n_estimators': 50})
    assert trainer.config['n_estimators'] == 50
    assert trainer.config['target_horizon'] == 5


def test_full_training(sample_data):
    trainer = ModelTrainer({'max_depth': 2})
    results = trainer.train(sample_data)

    assert 'model' in results
    assert 'scaler' in results
    assert isinstance(results['feature_names'], list)
    assert 0 <= results['test_score'] <= 1