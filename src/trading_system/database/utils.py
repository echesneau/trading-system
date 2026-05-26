import pandas as pd

from trading_system import config_path

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