# src/data/loader.py
import os
import json
import yfinance as yf
import krakenex
import ccxt
import pandas as pd
from typing import Optional, Union, Dict
import logging
import cachetools.func
import time

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

@cachetools.func.ttl_cache(maxsize=10, ttl=3600)
def load_kraken_data(
        pair: str,
        start_date: Union[str, pd.Timestamp],
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        interval: str = "1d",
        **kwargs
) -> pd.DataFrame:
    """
    Charge les données depuis kraken à intervalle régulier.

    Args:
        pair: paire kraken (ex: "XBTEUR")
        start_date: Date de début (incluse)
        end_date: Date de fin (exclue)
        interval: "1d", "1h", etc.

    Returns:
        DataFrame avec colonnes: Open, High, Low, Close, Volume

    Raises:
        DataLoadingError: Si le chargement échoue
    """
    interval_minutes = {
        '1m': 1, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '4h': 240, '1d': 1440, '1w': 10080
    }[interval]

    # Conversion des dates
    start_ts = int(pd.Timestamp(start_date).timestamp())
    end_ts = int(pd.Timestamp(end_date).timestamp()) if end_date else None
    # Téléchargement
    api = krakenex.API()
    all_data = []
    since = start_ts


    while True:
        resp = api.query_public("OHLC", {
            "pair": pair,
            "interval": interval_minutes,
            "since": since
        })
        if resp.get("error"):
            raise DataLoadingError(resp["error"])
        data = resp["result"].get(pair, [])
        last = resp["result"]["last"]
        if not data:
            break
        df = pd.DataFrame(data, columns=[
            "time", "Open", "High", "Low", "Close", "vwap", "Volume", "count"
        ])
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.set_index("time")
        # filtrage par date de fin
        if end_ts:
            df = df[df.index < pd.to_datetime(end_date)]
        if df.empty:
            break
        all_data.append(df[["Open", "High", "Low", "Close", "Volume"]].astype(float))

        # Avancer "since" pour la boucle suivante
        since = last
        # Si la dernière bougie dépasse end_date, on arrête
        if end_ts and since >= end_ts:
            break
    if not all_data:
        raise DataLoadingError(f"Aucune donnée disponible pour {pair}")

    return pd.concat(all_data).sort_index()




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

@cachetools.func.ttl_cache(maxsize=10, ttl=3600)
def load_ccxt_data(
    pair: str = "BTC/USDT",
    exchange_name: str = "binance",
    interval: str = "1d",
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    limit: int = 10000,
    pause: float = 1.2
) -> pd.DataFrame:
    """
    Charge les données OHLCV depuis un exchange via ccxt.

    Args:
        pair: paire de marché (ex: "BTC/USDT", "ETH/EUR")
        exchange_name: nom de l'exchange (ex: "binance", "kraken", "coinbase")
        interval: granularité ("1m","5m","15m","1h","1d","1w", etc.)
        start_date: date de début (str ou Timestamp), optionnel
        end_date: date de fin (exclue, str ou Timestamp), optionnel
        limit: nombre max de bougies par appel (souvent 1000)
        pause: délai entre appels API (pour éviter le rate limit)

    Returns:
        DataFrame avec colonnes: Open, High, Low, Close, Volume
    """
    # Initialisation exchange
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class()

    # Conversion des dates
    since_ts = int(pd.Timestamp(start_date).timestamp() * 1000) if start_date else None
    end_ts = int(pd.Timestamp(end_date).timestamp() * 1000) if end_date else None

    all_data = []
    while True:
        ohlcv = exchange.fetch_ohlcv(pair, timeframe=interval, since=since_ts, limit=limit)
        if not ohlcv:
            break

        df = pd.DataFrame(ohlcv, columns=["time","Open","High","Low","Close","Volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")

        # filtrer si end_date défini
        if end_ts:
            df = df[df["time"] <= pd.to_datetime(end_date)]

        if df.empty:
            break

        all_data.append(df)

        # Avancer since
        since_ts = int(df["time"].iloc[-1].timestamp() * 1000)

        # Conditions d’arrêt
        if len(ohlcv) < limit:
            break
        if end_ts and since_ts >= end_ts:
            break

        time.sleep(pause)

    if not all_data:
        raise ValueError(f"Aucune donnée récupérée pour {pair} sur {exchange_name}")

    return pd.concat(all_data).drop_duplicates("time").set_index("time").sort_index()