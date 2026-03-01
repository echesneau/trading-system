# optimize_parameters.py
import itertools
import warnings
import pandas as pd
import numpy as np
import math
import json
from datetime import datetime
from trading_system.data.loader import load_yfinance_data
from trading_system.backtesting.engine import BacktestingEngine
from trading_system.strategies.classical import ClassicalStrategy
from trading_system.features.technical import calculate_indicators
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
    if "Daily_Return" not in cache:
        cache["Daily_Return"] = data["Daily_Return"]
    return cache

def backtest_wrapper(params, raw_data, initial_capital=10000, transaction_fee=0.005, cache={}):
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
            'strategy_score':0,
            'error': str(e)
        }, cache


def optimize_parameters_parallel(raw_data, param_grid, initial_capital=10000,
                                 transaction_fee=0.005):
    """
    Exécute une grid search parallélisée pour trouver les meilleurs paramètres.

    Args:
        data: Données de marché pour le backtesting
        param_grid: Dictionnaire des plages de paramètres à tester
        initial_capital: Capital initial pour le backtest
        transaction_fee: Frais de transaction
        max_workers: Nombre maximum de processus workers (None = auto-détection)

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
    best_result = {
        'params': {},
        'sharpe_ratio': -np.inf,
        'total_return': -np.inf,
        'max_drawdown': np.inf,
        'annualized_return': -np.inf,
        'strategy_score': -np.inf,
        'n_trades': 0
    }
    for combination in itertools.product(*values):
        params = dict(zip(keys, combination))
        result, cache = backtest_wrapper(params, raw_data, initial_capital, transaction_fee, cache=cache)
        if result['strategy_score'] > best_result['strategy_score']:
            best_result = result
    return best_result

def optimize_one(ticker: str, grid: dict, odir="./"):
    raw_data = load_yfinance_data(
        ticker=ticker,
        start_date="2010-01-01",
        end_date="2024-01-01",
        interval="1d"
    )
    best_result = optimize_parameters_parallel(
        raw_data=raw_data,
        param_grid=grid,
        initial_capital=10000,
        transaction_fee=0.005  # None = utilise tous les coeurs disponibles
    )
    metadata = {
        "ticker": ticker,
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'params': best_result['params'],
        'train_results': {
            'sharpe_ratio': best_result['sharpe_ratio'],
            'total_return': best_result['total_return'],
            'max_drawdown': best_result['max_drawdown'],
            "strategy_score": best_result['strategy_score'],
            "annualized_return": best_result['annualized_return'],
        }
    }
    # Tester les meilleurs paramètres sur une période de validation différente
    validation_data = load_yfinance_data(
        ticker=ticker,
        start_date="2024-01-01",
        end_date="2026-02-01",
        interval="1d"
    )

    # Valider les meilleurs paramètres
    data = calculate_indicators(validation_data, **best_result['params'])
    best_strategy_engine = ClassicalStrategy(**best_result['params'])
    validation_engine = BacktestingEngine(
        strategy=best_strategy_engine,
        data=data,
        initial_capital=10000,
        transaction_fee=0.001,
        position_size=1
    )

    validation_result = validation_engine.run_numba()
    print(f"Performance en validation: {validation_result['performance']['return']:.2%}")
    metadata['validation_results'] = {
        'total_return': validation_result['performance']['return'],
        'sharpe_ratio': validation_result['performance']['sharpe_ratio'],
        'max_drawdown': validation_result['performance']['max_drawdown'],
        "strategy_score": validation_result['performance']['strategy_score'],
        "annualized_return": validation_result['performance']['annualized_return'],
    }
    #with open(f'{odir}/metadata_opt_{ticker}.json', 'w', encoding='utf-8') as f:
    #    json.dump(metadata, f, ensure_ascii=False, indent=4)

def run(ticker, param_grid, odir):
    print("=" * 50)
    print(f"Optimisation pour le ticker: {ticker}")
    optimize_one(ticker, grid=param_grid, odir=odir)

if __name__ == "__main__":
    tickers_cac40_list = [
        "AIR.PA",  # Airbus
        # "MT.AS",  # ArcelorMittal (coté Amsterdam)
        #"CS.PA",  # AXA
        #"BNP.PA",  # BNP Paribas
    ]
    # Définir les plages de paramètres à tester
    param_grid = {
        'rsi_window': [7, 21],
        'rsi_buy': [None, 30, 35],
        'rsi_sell': [70, 75],
        'macd_fast': [8, 20],
        'macd_slow': [21, 34],
        'macd_signal': [7, 13],
        'bollinger_window': [None, 10, 20],
        'bollinger_std': [None, 1, 1.5],
        "adx_min": [None, 20],
        "stock_min": [None, 20],
        "stock_max": [None, 80],
        "atr_max": [None, 0.1],
        "stochastic_oscillator": [False, True]
    }

    odir = f"data_optim/tt/"
    t0 = datetime.now()
    for ticker in tickers_cac40_list:
        run(ticker, param_grid, odir)
    tend = datetime.now()
    t_tot = (tend - t0).total_seconds()
    result = {"nb_ticker": len(tickers_cac40_list),
              "params": param_grid,
              "total_time_seconds": t_tot}
    # with open(f'{odir}/summary_optimization.json', 'w', encoding='utf-8') as f:
    #     json.dump(result, f, ensure_ascii=False, indent=4)



