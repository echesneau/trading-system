from datetime import datetime
import pandas as pd

from trading_system.database.tickers import TickersRepository
from trading_system.database.trading_params import BestStrategyRepository
from trading_system.database import euronext_csv_category, euronext_csv_growth_access
from trading_system.data.loader import load_yfinance_data
from scripts.profile_optimisation import backtest_wrapper
from trading_system.backtesting.engine import BacktestingEngine
from trading_system.strategies.classical import ClassicalStrategy
from trading_system.features.technical import calculate_indicators
from scripts.run_backtest_validator import is_valid

def test_integration_load_real_euronext_csv_categ(tmp_path):
    repo = TickersRepository(
        db_path=tmp_path / "test.db",
        euronext_csv_categ=euronext_csv_category
    )

    df = repo.load_euronext_csv(euronext_csv_category)

    # ---- Sanity checks ----
    assert not df.empty
    assert set(df.columns) == {"Ticker", "Company", "Market"}

    # ---- Invariants métier ----
    assert df["Ticker"].notna().all()
    assert df["Company"].notna().all()
    assert (df["Market"].isin(["Euronext_cat_A", "Euronext_cat_B", "Euronext_cat_C"])).all()

    # ---- Cohérence ----
    assert df["Ticker"].is_unique

def test_integration_load_real_euronext_csv_growth(tmp_path):
    repo = TickersRepository(
        db_path=tmp_path / "test.db",
        euronext_csv_categ=euronext_csv_growth_access
    )

    df = repo.load_euronext_csv(euronext_csv_growth_access)

    # ---- Sanity checks ----
    assert not df.empty
    assert set(df.columns) == {"Ticker", "Company", "Market"}

    # ---- Invariants métier ----
    assert df["Ticker"].notna().all()
    assert df["Company"].notna().all()
    assert (df["Market"].isin(["Euronext Growth", "Euronext Access"])).all()

    # ---- Cohérence ----
    assert df["Ticker"].is_unique
def test_integration_update_db_real_csv(tmp_path):
    db_path = tmp_path / "euronext.db"

    repo = TickersRepository(
        db_path=db_path,
        euronext_csv_categ=euronext_csv_category,
        euronext_csv_growth_access_path=euronext_csv_growth_access
    )

    repo.update_db()

    result = repo.fetch_all()

    assert not result.empty
    assert "ticker" in result.columns
    assert "market" in result.columns

    # Invariants
    assert result["ticker"].is_unique
    assert (result["market"].isin([
        "Euronext Growth", "Euronext Access", "Euronext_cat_A", "Euronext_cat_B", "Euronext_cat_C"]
    )).all()

def test_integration_update_params_db(temp_db, example_optim_results):
    repo = BestStrategyRepository(temp_db)
    repo.create_table()
    ticker = 'AIR.PA'
    params = {
        'rsi_window': 21,
        'rsi_buy': 30,
        'rsi_sell': 70,
        'macd_fast': 8,
        'macd_slow': 21,
        'macd_signal': 13,
        'bollinger_window': 10,
        'bollinger_std': 1
    }
    raw_data = load_yfinance_data(
        ticker=ticker,
        start_date="2010-01-01",
        end_date="2024-01-01",
        interval="1d"
    )
    result, cache = backtest_wrapper(params, raw_data, 1000, 0.05, cache={})
    best_strategy = pd.Series(result)
    metadata = {
        "ticker": ticker,
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'params': best_strategy['params'],
        'train_results': {
            'sharpe_ratio': best_strategy['sharpe_ratio'],
            'total_return': best_strategy['total_return'],
            'max_drawdown': best_strategy['max_drawdown'],
            "strategy_score": best_strategy['strategy_score'],
            "annualized_return": best_strategy['annualized_return'],
        }
    }
    validation_data = load_yfinance_data(
        ticker=ticker,
        start_date="2024-01-01",
        end_date="2026-02-01",
        interval="1d"
    )
    data = calculate_indicators(validation_data, **best_strategy['params'])
    best_strategy_engine = ClassicalStrategy(**best_strategy['params'])
    validation_engine = BacktestingEngine(
        strategy=best_strategy_engine,
        data=data,
        initial_capital=10000,
        transaction_fee=0.001,
        position_size=1
    )
    validation_result = validation_engine.run_numba()
    metadata['validation_results'] = {
        'total_return': validation_result['performance']['return'],
        'sharpe_ratio': validation_result['performance']['sharpe_ratio'],
        'max_drawdown': validation_result['performance']['max_drawdown'],
        "strategy_score": validation_result['performance']['strategy_score'],
        "annualized_return": validation_result['performance']['annualized_return'],
    }
    repo.upsert(metadata)
    result = repo.fetch_one('AIR.PA')
    assert result is not None
    assert result["ticker"] == 'AIR.PA'
    assert result["params_json"] == metadata["params"]
    assert result["train_total_return"] == metadata['train_results']["total_return"]
    assert result["val_total_return"] == metadata['validation_results']["total_return"]

def test_integration_validators_db(repo_validation):
    ticker = 'AIR.PA'
    params = {
        'rsi_window': 21,
        'rsi_buy': 30,
        'rsi_sell': 70,
        'macd_fast': 8,
        'macd_slow': 21,
        'macd_signal': 13,
        'bollinger_window': 10,
        'bollinger_std': 1
    }
    raw_data = load_yfinance_data(
        ticker=ticker,
        start_date="2023-01-01",
        end_date="2025-01-01",
        interval="1d"
    )
    data = calculate_indicators(raw_data, **params)
    strategy = ClassicalStrategy(**params)
    engine = BacktestingEngine(
        strategy=strategy,
        data=data,
        initial_capital=1000,
        transaction_fee=0.05,
        position_size=1
    )
    result = engine.run()
    valid = is_valid(result)
    repo_validation.upsert(ticker, valid["valid"], valid["reason"])
    fetched = repo_validation.fetch_all()
    assert len(fetched) == 1
    fetched = repo_validation.fetch_one(ticker)
    assert fetched["valid"] == valid["valid"]
