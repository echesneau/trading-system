import pandas as pd
import datetime as dt
from ccxt.base.errors import BadSymbol

from trading_system import config_path
from trading_system.data.loader import load_ccxt_data, load_yfinance_data
from trading_system.features.fundamental import get_previous_date
from trading_system.data.loader import DataLoadingError

def sparql_to_dataframe(results: dict) -> pd.DataFrame:
    """
    Function to transform SPARQL query results into a pandas DataFrame.
    """
    rows = []
    for b in results["results"]["bindings"]:
        row = {}
        for key, val in b.items():
            row[key] = val["value"]
        rows.append(row)
    return pd.DataFrame(rows)

def convert_exhange_wikidata_to_yahoo(df: pd.DataFrame,
                                      mapper_path: str = f"{config_path}/wikidata_yahoo_mapping.csv") -> pd.DataFrame:
    """
    Function to rename exchangeLabel from wikidata to yahoo.

    Parameters:
        df: pd.DataFrame
        mapper_path: str

    Returns:
        pd.DataFrame
    """
    # Load Mapper
    mapper = pd.read_csv(mapper_path, sep=",")
    # merged
    df = df.merge(mapper, on="exchangeLabel", how='left')
    return df

def add_yahoo_suffix(df: pd.DataFrame, suffix_table_path: str = f"{config_path}/yahoo_suffix.csv") -> pd.DataFrame:
    """
    Function to add yahoo suffix to ticker.

    Parameters:
        df: pd.DataFrame
        suffix_table_path: str

    Returns:
        pd.DataFrame
    """
    # Load Mapper
    suffix_table = pd.read_csv(suffix_table_path, sep=",", comment="#")
    # merged
    df = df.merge(suffix_table, left_on="yahoo_market", right_on="Marché ou indice", how='left')
    df["Ticker_Yahoo"] = df["ticker"] + df["Suffixe"].fillna("")
    return df

def check_crypto(ticker: str) -> bool:
    """
    Fonction pour vérifier que le ticker est valide et disponible.
    Parameters
    ----------
    ticker: str
        ticker à valider

    Returns
    -------
    bool
    """
    start_date = get_previous_date(7, today=None)
    end_date = (dt.datetime.now().date() + dt.timedelta(days=1)).strftime('%Y-%m-%d')
    try :
        data = load_ccxt_data(ticker,
                              start_date=start_date,
                              end_date=end_date)
        return len(data) > 0
    except BadSymbol :
        return False
    except ValueError:
        return False

def check_yahoo(ticker: str) -> bool:
    """
    Fonction pour vérifier que le ticker est valide et disponible.

    Parameters
    ----------
    ticker: str
        ticker à valider

    Returns
    -------
    bool
    """
    start_date = get_previous_date(7, today=None)
    end_date = (dt.datetime.now().date() + dt.timedelta(days=1)).strftime('%Y-%m-%d')
    try:
        data = load_yfinance_data(ticker,
                                  start_date=start_date,
                                  end_date=end_date)
        return len(data) > 0
    except DataLoadingError:
        return False
