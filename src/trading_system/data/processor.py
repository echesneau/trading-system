# src/data/processor.py
import pandas as pd
from typing import Optional
from src.trading_system.features.technical import calculate_indicators


def process_market_data(
        raw_data: pd.DataFrame,
        indicators_config: Optional[dict] = None
) -> pd.DataFrame:
    """
    Transforme les données brutes en données prêtes pour l'analyse.

    Args:
        raw_data: DataFrame de yfinance (doit contenir OHLCV)
        indicators_config: Configuration pour les indicateurs techniques

    Returns:
        DataFrame enrichi avec les indicateurs
    """
    default_config = {
        "rsi_window": 14,
        "ema_windows": [5, 10, 20],
        "bb_window": 20
    }
    config = {**default_config, **(indicators_config or {})}

    # Validation des données d'entrée
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing_cols = required_cols - set(raw_data.columns)
    if missing_cols:
        raise ValueError(f"Données manquantes: {missing_cols}")

    # Calcul des indicateurs
    processed_data = calculate_indicators(raw_data, **config)

    # Nettoyage des NaN (dûs aux indicateurs nécessitant une période)
    processed_data = processed_data.dropna()

    return processed_data