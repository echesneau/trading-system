from trading_system.database.tickers import TickersRepository
from trading_system.database.trading_params import BestStrategyRepository
from trading_system.database import db_path_dev, validator_db_path, euronext_csv_category, euronext_csv_growth_access

if __name__ == "__main__":
    # Open DB
    tickers_db = TickersRepository(db_path_dev, euronext_csv_categ=euronext_csv_category,
                                        euronext_csv_growth_access_path=euronext_csv_growth_access)
    params_db = BestStrategyRepository(db_path_dev)

    # Validate DB
    tickers_db.validate_existing_tickers(confirm=True)
    params_db.validate_existing_tickers(tickers_db)

