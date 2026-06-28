from trading_system.database.tickers import TickersRepository
from trading_system.database.trading_params import BestStrategyRepository
from trading_system.database.validators import StrategyValidationRepository
from trading_system.database import db_path, validator_db_path, euronext_csv_category, euronext_csv_growth_access

if __name__ == "__main__":
    # Open DB
    tickers_db = TickersRepository(db_path, euronext_csv_categ=euronext_csv_category,
                                        euronext_csv_growth_access_path=euronext_csv_growth_access)
    params_db = BestStrategyRepository(db_path)
    validators_db = StrategyValidationRepository(validator_db_path)

    # Validate DB
    tickers_db.validate_existing_tickers(confirm=False)
    params_db.validate_existing_tickers(tickers_db, confirm=False)
    validators_db.validate_existing_tickers(tickers_db, confirm=False)

