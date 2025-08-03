# src/strategies/core.py
from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):
    """
    Classe de base abstraite pour toutes les stratégies de trading.
    Toutes les stratégies doivent implémenter ces méthodes.
    """

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Méthode principale pour générer les signaux de trading.

        Args:
            data: DataFrame contenant les données de marché (OHLCV + indicateurs)

        Returns:
            Series avec les signaux ('BUY', 'SELL', 'HOLD') indexés par date
        """
        pass

    @abstractmethod
    def get_parameters(self) -> dict:
        """
        Retourne les paramètres actuels de la stratégie.

        Returns:
            Dictionnaire des paramètres
        """
        pass

    @abstractmethod
    def set_parameters(self, params: dict):
        """
        Met à jour les paramètres de la stratégie.

        Args:
            params: Dictionnaire des nouveaux paramètres
        """
        pass