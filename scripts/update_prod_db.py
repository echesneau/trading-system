from trading_system.database.trading_params import BestStrategyRepository
from trading_system.database.tickers import TickersRepository
from trading_system.database import db_path_dev, db_path, euronext_csv_category, euronext_csv_growth_access

if __name__ == '__main__':
    tickers_db_prod = TickersRepository(db_path, euronext_csv_categ=euronext_csv_category,
                                        euronext_csv_growth_access_path=euronext_csv_growth_access)
    tickers_db_dev = TickersRepository(db_path_dev, euronext_csv_categ=euronext_csv_category,
                                        euronext_csv_growth_access_path=euronext_csv_growth_access)
    params_db_prod = BestStrategyRepository(db_path)
    params_db_dev = BestStrategyRepository(db_path_dev)

    for _, dev_values in params_db_dev.fetch_all().iterrows():
        ticker = dev_values['ticker']
        prod_values = params_db_prod.fetch_one(ticker)
        if prod_values is None:
            dev_values['date'] = dev_values['updated_at']
            dev_values['params'] = dev_values['params_json']
            dev_values['train_results'] = {}
            for par in ['total_return', 'annualized_return', "sharpe_ratio", "max_drawdown", "strategy_score"]:
                dev_values['train_results'][par] = dev_values[f"train_{par}"]
            dev_values['validation_results'] = {}
            for par in ['total_return', 'annualized_return', "sharpe_ratio", "max_drawdown", "strategy_score"]:
                dev_values['validation_results'][par] = dev_values[f"val_{par}"]
            params_db_prod.upsert(dev_values)
            print(f"{ticker} updated in prod db")
        elif (
                (dev_values['val_strategy_score'] is not None and
                 (prod_values['val_strategy_score'] is None or
                  dev_values['val_strategy_score'] > prod_values['val_strategy_score']))
                and
                (dev_values['val_annualized_return'] is not None and
                 (prod_values['val_annualized_return'] is None or
                  dev_values['val_annualized_return'] > prod_values['val_annualized_return']))
        ):
            dev_values['date'] = dev_values['updated_at']
            dev_values['params'] = dev_values['params_json']
            dev_values['train_results'] = {}
            for par in ['total_return', 'annualized_return', "sharpe_ratio", "max_drawdown", "strategy_score"]:
                dev_values['train_results'][par] = dev_values[f"train_{par}"]
            dev_values['validation_results'] = {}
            for par in ['total_return', 'annualized_return', "sharpe_ratio", "max_drawdown", "strategy_score"]:
                dev_values['validation_results'][par] = dev_values[f"val_{par}"]
            params_db_prod.upsert(dev_values)
            print(f"{ticker} updated in prod db")
        else:
            print(f"{ticker} is not updated in prod db")