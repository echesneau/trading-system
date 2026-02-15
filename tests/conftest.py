import pytest
import pandas as pd
import numpy as np

from trading_system.ml.trainer import ModelTrainer
from trading_system.data.loader import load_yfinance_data
from trading_system.database.tickers import TickersRepository
from trading_system.database.trading_params import BestStrategyRepository
from trading_system.database.validators import StrategyValidationRepository

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
            'bollinger_window': 14
        }
    })


@pytest.fixture(scope="session")
def trained_model_artifacts(test_data, model_trainer):
    """Utilise ModelTrainer pour entraîner un modèle sur les données réelles"""
    artifacts = model_trainer.train(test_data)
    # Validation basique du modèle
    assert 'model' in artifacts
    assert 'scaler' in artifacts
    assert 'feature_names' in artifacts
    assert len(artifacts['feature_names']) > 0

    return artifacts

@pytest.fixture()
def temp_db(tmp_path):
    return tmp_path / "test.db"

@pytest.fixture
def euronext_csv(tmp_path):
    csv_path = tmp_path / "euronext.csv"

    df = pd.DataFrame({
        "Company": [
            "Crédit Agricole",
            "Bitcoin Corp",
        ],
        "Ticker": [
            "ACA.PA",
            "BTCUSD"
        ],
        "Exchange": [
            "Euronext Paris",
            "Crypto"
        ],
        "Currency": [
            "EUR",
            "USD"
        ]
    })

    df.to_csv(csv_path, index=False)
    return csv_path

@pytest.fixture()
def repo_tickers(temp_db, euronext_csv):
    return TickersRepository(temp_db, euronext_csv_path=euronext_csv)

@pytest.fixture()
def repo_strategy(temp_db):
    return BestStrategyRepository(temp_db)

@pytest.fixture
def example_optim_results():
    return {
        "ticker": "ACA.PA",
        "date": "2026-02-02 00:06:11",
        "params": {
            "rsi_window": 7,
            "rsi_buy": 35,
            "rsi_sell": 75,
            "macd_fast": 16,
            "macd_slow": 26,
            "macd_signal": 13,
            "bollinger_window": 10,
            "bollinger_std": 1
        },
        "train_results": {
            "sharpe_ratio": 0.05,
            "total_return": 5.48,
            "max_drawdown": 0.29,
            "strategy_score": 0.23,
            "annualized_return": 0.14
        },
        "validation_results": {
            "total_return": 0.019,
            "sharpe_ratio": 0.026,
            "max_drawdown": 0.025,
            "strategy_score": 0.155,
            "annualized_return": 0.009
        }
    }

@pytest.fixture()
def repo_validation(temp_db):
    return StrategyValidationRepository(temp_db)
