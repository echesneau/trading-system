# optimize_parameters.py
import itertools
import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from trading_system import config_path
from trading_system.data.loader import load_yfinance_data
from trading_system.backtesting.engine import BacktestingEngine
from trading_system.strategies.classical import ClassicalStrategy
from trading_system.features.technical import calculate_indicators


def backtest_wrapper(params, raw_data, initial_capital=10000, transaction_fee=0.005):
    """
    Fonction wrapper pour exécuter un backtest avec des paramètres spécifiques.
    Cette fonction doit être autonome pour être exécutée dans un processus séparé.
    """
    try:
        data = calculate_indicators(raw_data, **params)
        strategy = ClassicalStrategy(**params)
        engine = BacktestingEngine(
            strategy=strategy,
            data=data,
            initial_capital=initial_capital,
            transaction_fee=transaction_fee,
            position_size=1
        )

        result = engine.run()

        return {
            'params': params,
            'sharpe_ratio': result['performance']['sharpe_ratio'],
            'total_return': result['performance']['return'],
            'max_drawdown': result['performance']['max_drawdown'],
            'annualized_return': result['performance']['annualized_return'],
            'strategy_score': result['performance']['strategy_score'],
            'n_trades': len(result['trades']) if 'trades' in result and result['trades'] is not None else 0
        }
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
        }


def optimize_parameters_parallel(raw_data, param_grid, initial_capital=10000,
                                 transaction_fee=0.005, max_workers=None):
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
    all_combinations = [dict(zip(keys, combination))
                        for combination in itertools.product(*values)]

    print(f"Nombre total de combinaisons à tester: {len(all_combinations)}")
    print()

    # Utiliser ProcessPoolExecutor pour paralléliser les backtests
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Soumettre toutes les tâches
        future_to_params = {
            executor.submit(
                backtest_wrapper,
                params,
                raw_data,
                initial_capital,
                transaction_fee
            ): params for params in all_combinations
        }

        # Collecter les résultats au fur et à mesure
        for i, future in enumerate(as_completed(future_to_params)):
            params = future_to_params[future]
            try:
                result = future.result()
                results.append(result)

                if i % 100 == 0:
                    print(f"\rTest {i}/{len(all_combinations)} - strategy_score: {result['strategy_score']:.2f}", end='', flush=True)
                    #print(f"Test {i}/{len(all_combinations)} - Return: {result['total_return']:.2f}", end="\n")

            except Exception as e:
                print(f"Erreur avec les paramètres {params}: {str(e)}")
                results.append({
                    'params': params,
                    'sharpe_ratio': -np.inf,
                    'total_return': -np.inf,
                    'max_drawdown': np.inf,
                    'annualized_return': -np.inf,
                    'strategy_score': -np.inf,
                    'n_trades': 0,
                    'error': str(e)
                })

    return pd.DataFrame(results)


    return pd.DataFrame(results)

def optimize_one(ticker: str, grid: dict, odir="./"):
    raw_data = load_yfinance_data(
        ticker=ticker,
        start_date="2010-01-01",
        end_date="2023-01-01",
        interval="1d"
    )
    results_df = optimize_parameters_parallel(
        raw_data=raw_data,
        param_grid=grid,
        initial_capital=10000,
        transaction_fee=0.005,
        max_workers=None  # None = utilise tous les coeurs disponibles
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

    # Sauvegarder tous les résultats
    results_df.to_csv(f"{odir}/optimization_results_{ticker}.csv", index=False)
    # Tester les meilleurs paramètres sur une période de validation différente
    validation_data = load_yfinance_data(
        ticker=ticker,
        start_date="2023-01-01",
        end_date="2025-09-01",
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

    validation_result = validation_engine.run()
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

if __name__ == "__main__":
    tickers_cac40_list = [
        "AIR.PA",  # Airbus
        "MT.AS",  # ArcelorMittal (coté Amsterdam)
        "CS.PA",  # AXA
        "BNP.PA",  # BNP Paribas
        "EN.PA",  # Bouygues
        "CAP.PA",  # Capgemini
        "CA.PA",  # Carrefour
        "ACA.PA",  # Crédit Agricole
        "BN.PA",  # Danone
        "DSY.PA",  # Dassault Systèmes
        "ENGI.PA",  # Engie
        "EL.PA",  # EssilorLuxottica
        "ERF.PA",  # Eurofins Scientific
        "RMS.PA",  # Hermès
        "KER.PA",  # Kering
        "LR.PA",  # Legrand
        "OR.PA",  # L’Oréal
        "MC.PA",  # LVMH
        "ML.PA",  # Michelin
        "ORA.PA",  # Orange
        "RI.PA",  # Pernod Ricard
        "RNO.PA",  # Renault
        "SAF.PA",  # Safran
        "SGO.PA",  # Saint-Gobain
        "SAN.PA",  # Sanofi
        "SU.PA",  # Schneider Electric
        "GLE.PA",  # Société Générale
        "STMPA.PA",  # STMicroelectronics
        "TEP.PA",  # Teleperformance
        "HO.PA",  # Thales
        "TTE.PA",  # TotalEnergies
        # "URW.PA",  # Unibail-Rodamco-Westfield --> problème de données
        "VIE.PA",  # Veolia
        "DG.PA",  # Vinci
        "VIV.PA",  # Vivendi
        "WLN.PA"  # Worldline
    ]
    # Définir les plages de paramètres à tester
    param_grid = {
        'rsi_window': [7, 10, 14, 21],
        'rsi_buy': [25, 30, 35],
        'rsi_sell': [65, 70, 75],
        'macd_fast': [8, 12, 16, 20],
        'macd_slow': [21, 26, 30, 34],
        'macd_signal': [7, 9, 11, 13],
        'bollinger_window': [10, 15, 20, 25],
        'bollinger_std': [1, 1.5, 2.0]
    }

    odir = f"{config_path}/classical_strategy/"
    for ticker in tickers_cac40_list:
        print("=" * 50)
        print(f"Optimisation pour le ticker: {ticker}")
        optimize_one(ticker, grid=param_grid, odir=odir)






