# optimize_parameters.py
import itertools
import warnings
import pandas as pd
import numpy as np
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    return cache

def backtest_wrapper(params, raw_data, initial_capital=10000, transaction_fee=0.005, cache={}):
    """
    Fonction wrapper pour exécuter un backtest avec des paramètres spécifiques.
    Cette fonction doit être autonome pour être exécutée dans un processus séparé.
    """
    try:
        data = calculate_indicators(raw_data, cache=cache,  **params)
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

    Returns:
        DataFrame avec les résultats de tous les tests
    """
    results = []

    # Générer toutes les combinaisons de paramètres
    keys = param_grid.keys()
    values = param_grid.values()
    all_combinations = [dict(zip(keys, combination))
                        for combination in itertools.product(*values)]

    print(f"Nombre total de combinaisons à tester: {len(all_combinations)}")
    print()
    cache = {}
    for params in all_combinations:
        result, cache = backtest_wrapper(params, raw_data, initial_capital, transaction_fee, cache=cache)
        results.append(result)

    # Utiliser ProcessPoolExecutor pour paralléliser les backtests
    # with ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     # Soumettre toutes les tâches
    #     future_to_params = {
    #         executor.submit(
    #             backtest_wrapper,
    #             params,
    #             raw_data,
    #             initial_capital,
    #             transaction_fee
    #         ): params for params in all_combinations
    #     }

    #     # Collecter les résultats au fur et à mesure
    #     for i, future in enumerate(as_completed(future_to_params)):
    #         params = future_to_params[future]
    #         try:
    #             result = future.result()
    #             results.append(result)

    #             if i % 100 == 0:
    #                 pass
                    # print(f"\rTest {i}/{len(all_combinations)} - strategy_score: {result['strategy_score']:.2f}", end='', flush=True)
                    #print(f"Test {i}/{len(all_combinations)} - Return: {result['total_return']:.2f}", end="\n")

    #         except Exception as e:
    #             print(f"Erreur avec les paramètres {params}: {str(e)}")
    #             results.append({
    #                 'params': params,
    #                 'sharpe_ratio': -np.inf,
    #                 'total_return': -np.inf,
    #                 'max_drawdown': np.inf,
    #                 'annualized_return': -np.inf,
    #                 'strategy_score': -np.inf,
    #                 'n_trades': 0,
    #                 'error': str(e)
    #             })

    return pd.DataFrame(results)


def optimize_one(ticker: str, grid: dict, odir="./"):
    raw_data = load_yfinance_data(
        ticker=ticker,
        start_date="2010-01-01",
        end_date="2024-01-01",
        interval="1d"
    )
    results_df = optimize_parameters_parallel(
        raw_data=raw_data,
        param_grid=grid,
        initial_capital=10000,
        transaction_fee=0.005,
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
    # print("Meilleur Strategie:")
    # print(f"Paramètres: {best_strategy['params']}")
    # print(f"Sharpe: {best_strategy['sharpe_ratio']:.2f}")
    # print(f"Total Return: {best_strategy['total_return']:.2%}")
    # print(f"Annualized Return: {best_strategy['annualized_return']:.2%}")
    # print(f"Strategy Score: {best_strategy['strategy_score']:.2f}")

    # Sauvegarder tous les résultats
    results_df.to_csv(f"{odir}/optimization_results_{ticker}.csv", index=False)
    # Tester les meilleurs paramètres sur une période de validation différente
    validation_data = load_yfinance_data(
        ticker=ticker,
        start_date="2024-01-01",
        end_date="2026-02-01",
        interval="1d"
    )

    # Valider les meilleurs paramètres
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
    print(f"Performance en validation: {validation_result['performance']['return']:.2%}")
    metadata['validation_results'] = {
        'total_return': validation_result['performance']['return'],
        'sharpe_ratio': validation_result['performance']['sharpe_ratio'],
        'max_drawdown': validation_result['performance']['max_drawdown'],
        "strategy_score": validation_result['performance']['strategy_score'],
        "annualized_return": validation_result['performance']['annualized_return'],
    }
    with open(f'{odir}/metadata_opt_{ticker}.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

def run(ticker, param_grid, odir):
    print("=" * 50)
    print(f"Optimisation pour le ticker: {ticker}")
    optimize_one(ticker, grid=param_grid, odir=odir)

if __name__ == "__main__":
    tickers_cac40_list = [
        "AIR.PA",  # Airbus
        "MT.AS",  # ArcelorMittal (coté Amsterdam)
        "CS.PA",  # AXA
        "BNP.PA",  # BNP Paribas
    ]
    # Définir les plages de paramètres à tester
    param_grid = {
        'rsi_window': [7, 21],
        'rsi_buy': [30, 35],
        'rsi_sell': [70, 75],
        'macd_fast': [8, 20],
        'macd_slow': [21, 34],
        'macd_signal': [7, 13],
        'bollinger_window': [10, 20],
        'bollinger_std': [1, 1.5],
        "adx_min": [None, 20], # faible 15-20, forte 25-40
        "stock_min": [None, 20], # inf 25 environ
        "stock_max": [None, 80], # sup 75 environ
        "atr_max": [None, 0.03], # entre 0 et 0.05
        "stochastic_oscillator": [False, True],
    }

    odir = f"data_optim/test_param/"
    t0 = datetime.now()
    max_workers = 4
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run, t, param_grid, odir): t for t in tickers_cac40_list}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                future.result()
                print(f"✅ Terminé pour {ticker}")
            except Exception as e:
                print(f"Erreur lors de l'optimisation pour {ticker}: {str(e)}")

    tend = datetime.now()
    t_tot = (tend - t0).total_seconds()
    result = {"nb_ticker": len(tickers_cac40_list),
              "params": param_grid,
              "total_time_seconds": t_tot}
    with open(f'{odir}/summary_optimization.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)






