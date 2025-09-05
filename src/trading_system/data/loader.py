# src/data/loader.py
import os
import json
import yfinance as yf
import pandas as pd
from typing import Optional, Union, Dict
import logging
import cachetools.func

logger = logging.getLogger(__name__)


class DataLoadingError(Exception):
    """Exception personnalisée pour les erreurs de chargement"""
    pass


def _normalize_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalise les colonnes (gère le MultiIndex de yfinance)"""
    if isinstance(df.columns, pd.MultiIndex):
        # Cas multi-tickers : on sélectionne le ticker demandé
        if ticker not in df.columns.levels[0]:
            raise DataLoadingError(f"Ticker {ticker} non trouvé dans les données")

        # Flattening des colonnes
        df = df[ticker].copy()
    else:
        # Cas single ticker
        df = df.copy()

    return df


@cachetools.func.ttl_cache(maxsize=10, ttl=3600)
def load_yfinance_data(
        ticker: str,
        start_date: Union[str, pd.Timestamp],
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        interval: str = "1d",
        progress: bool = False,
        **kwargs
) -> pd.DataFrame:
    """
    Charge les données depuis Yahoo Finance avec gestion du MultiIndex.

    Args:
        ticker: Symbole Yahoo Finance (ex: "AIR.PA")
        start_date: Date de début (incluse)
        end_date: Date de fin (exclue)
        interval: "1d", "1h", etc.

    Returns:
        DataFrame avec colonnes: Open, High, Low, Close, Volume

    Raises:
        DataLoadingError: Si le chargement échoue
    """
    try:
        logger.info(f"Chargement {ticker} du {start_date} au {end_date}")

        # Validation du ticker
        if not isinstance(ticker, str) or "." not in ticker:
            raise DataLoadingError(f"Format de ticker invalide: {ticker}")

        # Téléchargement
        data = yf.download(
            tickers=ticker,
            start=pd.to_datetime(start_date),
            end=pd.to_datetime(end_date) if end_date else None,
            interval=interval,
            progress=progress,
            group_by='ticker',  # Important pour la cohérence
            **kwargs
        )
        # 1. Vérification des données vides en premier
        if data.empty:
            raise DataLoadingError(f"Aucune donnée disponible pour {ticker}")
        # Normalisation des colonnes
        data = _normalize_columns(data, ticker)

        # Validation
        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        missing_cols = required_cols - set(data.columns)
        if missing_cols:
            raise DataLoadingError(f"Colonnes manquantes: {missing_cols}")

        return data[list(required_cols)]

    except Exception as e:
        logger.exception("Erreur de chargement")
        raise DataLoadingError(f"Erreur avec Yahoo Finance pour {ticker}: {str(e)}")


def load_multiple_tickers(
        tickers: list,  # Format: {"AIR.PA": "Airbus"}
        **kwargs
) -> Dict[str, pd.DataFrame]:
    """Charge plusieurs tickers en parallèle"""
    from concurrent.futures import ThreadPoolExecutor

    def _load_single(ticker):
        return ticker, load_yfinance_data(ticker, **kwargs)
        #try:
        #    return ticker, load_yfinance_data(ticker, **kwargs)
        #except Exception as e:
        #    logger.warning(f"Échec sur {ticker}: {str(e)}")
        #    return ticker, None

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = dict(executor.map(_load_single, tickers))

    return {ticker: df for ticker, df in results.items() if df is not None}

def get_all_ticker_parameters_from_config(config_path: str) -> dict:
    """Extrait tous les tickers metadata du répertoire de configuration."""
    params = {}
    for file in os.listdir(config_path):
        if file.endswith('.json'):
            with open(f'{config_path}/{file}') as f:
                tmp = json.load(f)
            ticker = tmp['ticker']
            params[ticker] = tmp['params']
    return params

def load_validation_results(file_path: str) -> dict:
    """Charge les résultats de validation depuis un fichier JSON."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier non trouvé: {file_path}")
    with open(file_path, 'r') as f:
        return json.load(f)
