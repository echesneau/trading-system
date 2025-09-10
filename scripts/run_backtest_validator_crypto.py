import json
from datetime import datetime, timedelta

from trading_system import config_path
from trading_system.data.loader import get_all_ticker_parameters_from_config, load_ccxt_data
from trading_system.strategies.classical import ClassicalStrategy
from trading_system.features.technical import calculate_indicators
from trading_system.backtesting.engine import BacktestingEngine

def is_valid(result, min_performance=0.02, max_drawdown=-20.0, min_trades=2, min_trades_earn_rate=0.75):
    valid = True
    reason = []
    if result['performance']['return'] < min_performance:
        valid = False
        reason += ['Low Return']
    if result['performance']['max_drawdown'] < max_drawdown:
        valid = False
        reason += ['High Drawdown']
    if len(result['trades']) < min_trades:
        valid = False
        reason += ['Not enough trades']

    reason = ', '.join(reason) if len(reason) > 0 else 'OK'
    return {'valid': valid, 'reason': reason}

if __name__ == "__main__":
    # get all metadata files
    output_path = f"{config_path}/validation_classical_strategy_crypto.json"
    config_path = f"{config_path}/classical_strategy_crypto/"
    config = get_all_ticker_parameters_from_config(config_path)
    validation_period = 365  # 1 an de validation
    initial_capital = 10000
    transaction_fee = 0.0025  # 0.5% par transaction
    end_date = datetime.now().date()
    start_date = (end_date - timedelta(days=validation_period + 50)).strftime('%Y-%m-%d')  # +50 pour les indicateurs
    end_date = end_date.strftime('%Y-%m-%d')
    valid = {}
    for ticker, params in config.items():
        raw_data = load_ccxt_data(
            ticker,
            exchange_name="binance",
            interval="1d",
            start_date=start_date,
            end_date=end_date,
            limit=None
        )
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
        valid[ticker] = is_valid(result)
    # Sauvegarder le fichier
    with open(output_path, 'w') as f:
        json.dump(valid, f, indent=4)