# optimize_parameters.py
import itertools
import warnings
import pandas as pd
import numpy as np
import math
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from trading_system.data.loader import load_ccxt_data
from trading_system.backtesting.engine import BacktestingEngine
from trading_system.strategies.classical import ClassicalStrategy
from trading_system.features.technical import calculate_indicators
from trading_system.database.trading_params import BestStrategyRepository
from trading_system.database import db_path, euronext_csv_category, euronext_csv_growth_access
from trading_system.database.tickers import TickersRepository

warnings.filterwarnings("ignore")


def update_cache(cache, data, params):
    # RSI
    if 'rsi_window' in params and f"RSI_{params['rsi_window']}" not in cache:
        cache[f"RSI_{params['rsi_window']}"] = data[f"RSI"]
    # MACD
    macd_key = f"MACD_{params['macd_fast']}_{params['macd_slow']}_{params['macd_signal']}"
    macd_signal_key = f"MACD_Signal_{params['macd_fast']}_{params['macd_slow']}_{params['macd_signal']}"
    if "macd_fast" in params and "macd_slow" in params and "macd_signal" in params and macd_key not in cache:
        cache[macd_key] = data[f"MACD"]
        cache[macd_signal_key] = data[f"MACD_Signal"]
    # ATR
    if 'atr_window' in params and f"ATR_{params['atr_window']}" not in cache:
        cache[f"ATR_{params['rsi_window']}"] = data[f"ATR"]
    # Bollinger Bands
    if "bollinger_window" in params and "bollinger_std" in params and f"BB_Upper_{params['bollinger_window']}_{params['bollinger_std']}" not in cache:
        cache[f"BB_Upper_{params['bollinger_window']}_{params['bollinger_std']}"] = data[f"BB_Upper"]
        cache[f"BB_Middle_{params['bollinger_window']}_{params['bollinger_std']}"] = data[f"BB_Middle"]
        cache[f"BB_Lower_{params['bollinger_window']}_{params['bollinger_std']}"] = data[f"BB_Lower"]
    # EMA
    if "ema_windows" in params:
        for window in params["ema_windows"]:
            ema_key = f"EMA_{window}"
            if ema_key not in cache:
                cache[ema_key] = data[f"EMA_{window}"]
    # ADX
    if "adx_window" in params and f"ADX_{params['adx_window']}" not in cache:
        cache[f"ADX_{params['adx_window']}"] = data[f"ADX"]
    # OBV
    if "balance_volume" in params and params['balance_volume'] and "OBV" not in cache:
        cache["OBV"] = data["OBV"]
    # VolMa
    if "volma_window" in params and f"VolMA_{params['volma_window']}" not in cache:
        cache[f"VolMA_{params['volma_window']}"] = data[f"VolMA"]
    # Price volume trend
    if "price_volume_trend" in params and params['price_volume_trend'] and "Price_Volume_Trend" not in cache:
        cache["Price_Volume_Trend"] = data["Price_Volume_Trend"]
    return cache

def backtest_wrapper(params, raw_data, initial_capital=10000, transaction_fee=0.0025, cache={}):
    """
    Fonction wrapper pour exécuter un backtest avec des paramètres spécifiques.
    Cette fonction doit être autonome pour être exécutée dans un processus séparé.
    """
    try:
        data = calculate_indicators(raw_data, cache=cache, **params)
        cache = update_cache(cache, data, params)
        strategy = ClassicalStrategy(**params)
        engine = BacktestingEngine(
            strategy=strategy,
            data=data,
            initial_capital=initial_capital,
            transaction_fee=transaction_fee,
            position_size=1
        )

        result = engine.run_numba()

        return {
            'params': params,
            'sharpe_ratio': result['performance']['sharpe_ratio'],
            'total_return': result['performance']['return'],
            'max_drawdown': result['performance']['max_drawdown'],
            'annualized_return': result['performance']['annualized_return'],
            'strategy_score': result['performance']['strategy_score'],
            'n_trades': len(result['trades']) if 'trades' in result and result['trades'] is not None else 0
        }, cache
    except Exception as e:
        print(f"Erreur avec les paramètres {params}: {str(e)}")
        return {
            'params': params,
            'sharpe_ratio': -np.inf,
            'total_return': -np.inf,
            'max_drawdown': np.inf,
            'annualized_return': -np.inf,
            'n_trades': 0,
            'strategy_score': 0,
            'error': str(e)
        }, cache


def optimize_parameters_parallel(raw_data, param_grid, initial_capital=10000,
                                 transaction_fee=0.0025):
    """
    Exécute une grid search parallélisée pour trouver les meilleurs paramètres.

    Args:
        data: Données de marché pour le backtesting
        param_grid: Dictionnaire des plages de paramètres à tester
        initial_capital: Capital initial pour le backtest
        transaction_fee: Frais de transaction

    Returns:
        DataFrame avec les résultats de tous les tests
    """
    results = []

    # Générer toutes les combinaisons de paramètres
    keys = param_grid.keys()
    values = param_grid.values()
    n_combinations = math.prod(len(v) for v in param_grid.values())

    print(f"Nombre total de combinaisons à tester: {n_combinations}")
    print()

    cache = {}
    for combination in itertools.product(*values):
        params = dict(zip(keys, combination))
        result, cache = backtest_wrapper(params, raw_data, initial_capital, transaction_fee, cache=cache)
        results.append(result)

    return pd.DataFrame(results)


    return pd.DataFrame(results)
def get_exchange_from_ticker(ticker: str) -> str:
    try :
        min_kraken = load_ccxt_data(pair=ticker, exchange_name="kraken",
                             start_date="2000-01-02",interval="1d").index.min()
    except :
        min_kraken = datetime.now()
    try:
        min_binance = load_ccxt_data(pair=ticker, exchange_name="binance",
                             start_date="2000-01-02",interval="1d").index.min()
    except :
        min_binance = datetime.now()
    if min_kraken<min_binance:
        loader = "kraken"
    else:
        loader = "binance"
    return loader

def optimize_one(ticker: str, grid: dict, database = BestStrategyRepository(db_path)):

    raw_data = load_ccxt_data(
        ticker,
        exchange_name=get_exchange_from_ticker(ticker),
        interval="1d",
        start_date="2000-01-02",
        end_date="2025-09-01",
        limit=None)
    results_df = optimize_parameters_parallel(
        raw_data=raw_data,
        param_grid=grid,
        initial_capital=10000,
        transaction_fee=0.0025,
    )
    # Trouver les meilleures combinaisons selon différentes métriques
    best_sharpe = results_df.loc[results_df['sharpe_ratio'].idxmax()]
    best_return = results_df.loc[results_df['total_return'].idxmax()]
    best_drawdown = results_df.loc[results_df['max_drawdown'].idxmin()]
    best_strategy = results_df.loc[results_df['strategy_score'].idxmax()]

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
    print("Meilleur Strategie:")
    print(f"Paramètres: {best_strategy['params']}")
    print(f"Sharpe: {best_strategy['sharpe_ratio']:.2f}")
    print(f"Total Return: {best_strategy['total_return']:.2%}")
    print(f"Annualized Return: {best_strategy['annualized_return']:.2%}")
    print(f"Strategy Score: {best_strategy['strategy_score']:.2f}")

    # Tester les meilleurs paramètres sur une période de validation différente
    validation_data = load_ccxt_data(
        ticker,
        exchange_name="kraken",
        interval="1d",
        start_date="2025-09-01",
        end_date="2026-06-01",
        limit=None)

    # Valider les meilleurs paramètres
    data = calculate_indicators(validation_data, **best_strategy['params'])
    best_strategy_engine = ClassicalStrategy(**best_strategy['params'])
    validation_engine = BacktestingEngine(
        strategy=best_strategy_engine,
        data=data,
        initial_capital=10000,
        transaction_fee=0.0025,
        position_size=1
    )

    validation_result = validation_engine.run()
    print(f"Performance en validation: {validation_result['performance']['return']:.2%}")
    metadata['validation_results'] = {
        'total_return': validation_result['performance']['return'],
        'sharpe_ratio': validation_result['performance']['sharpe_ratio'],
        'max_drawdown': validation_result['performance']['max_drawdown'],
        "strategy_score": validation_result['performance']['strategy_score'],
        "annualized_return": validation_result['performance']['annualized_return'],
    }
    database.upsert(metadata)

def run(ticker, param_grid, db):
    print("=" * 50)
    print(f"Optimisation pour le ticker: {ticker}")
    optimize_one(ticker, grid=param_grid, database=db)

if __name__ == "__main__":
    tickers_cac40_list = [
        "BTC/EUR", 'ETH/EUR', 'XRP/EUR', 'ADA/EUR', 'XLM/EUR', 'ATOM/EUR', #  'ETC/EUR'
        "BTC/USDT", 'ETH/USDT', 'XRP/USDT', 'ADA/USDT', 'XLM/USDT', 'ETC/USDT', 'ATOM/USDT'
    ]
    tickers_db = TickersRepository(db_path, euronext_csv_categ=euronext_csv_category,
                                   euronext_csv_growth_access_path=euronext_csv_growth_access)
    tickers_db.update_db()
    tickers_df = tickers_db.fetch_all()
    tickers_eur_df = tickers_df[tickers_df['market'].isin(["Crypto_EUR"])]
    tickers_usd_df = tickers_df[tickers_df['market'].isin(["Crypto_USDT"])]
    params_db = BestStrategyRepository(db_path)

    all_tickers = pd.concat([tickers_eur_df, tickers_usd_df])
    # Définir les plages de paramètres à tester
    param_grid = {
        'rsi_window': [7, 10, 14, 21],
        'rsi_buy': [25, 30, 35],
        'rsi_sell': [65, 70, 75],
        'macd_fast': [8, 12, 16, 20],
        'macd_slow': [21, 26, 30, 34],
        'macd_signal': [7, 9, 11, 13],
        'bollinger_window': [10, 15, 20, 25],
        'bollinger_std': [1, 1.5, 2.0],
        # "adx_min": [None, 15, 20, 25],  # faible 15-20, forte 25-40
        # "stock_min": [None, 20, 25],  # inf 25 environ
        # "stock_max": [None, 75, 80],  # sup 75 environ
        # "atr_max": [None, 0.01, 0.03, 0.1],  # entre 0 et 0.05
        # "stochastic_oscillator": [False, True],
    }
    max_workers = 6
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run, t, param_grid, params_db): t for t in all_tickers['ticker'].tolist()}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                future.result()
                print(f"✅ Terminé pour {ticker}")
            except Exception as e:
                print(f"Erreur lors de l'optimisation pour {ticker}: {str(e)}")






