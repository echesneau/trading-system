from abc import ABC, abstractmethod
import logging

class BrokerBase(ABC):
    """
    Interface de base pour tous les brokers (Kraken, Binance, etc.).

    Cette classe définit le contrat minimal que chaque broker doit respecter.
    """

    def __init__(self, dry_run=True, base_currency="EUR", max_position_size=100, min_position_size = 1):
        self.dry_run = dry_run
        self.base_currency = base_currency
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.logger = logging.getLogger(self.__class__.__name__)
        self.name = self.__class__.__name__

    # --- Méthodes obligatoires ---------------------------------------------

    @abstractmethod
    def get_balance(self, currency: str):
        """Retourne le solde disponible pour une devise."""
        pass

    @abstractmethod
    def place_market_order(self, symbol: str, side: str, amount: float):
        """Passe un ordre au marché."""
        pass

    @abstractmethod
    def place_stop_loss(self, symbol: str, amount: float, stop_price: float):
        """Place un stop-loss."""
        pass

    @abstractmethod
    def get_open_positions(self):
        """Retourne les positions ouvertes (si applicable)."""
        pass

    # --- Méthodes utilitaires mutualisées ----------------------------------

    def log_order(self, message: str):
        """Méthode utilitaire commune pour logguer les ordres."""
        self.logger.info(f"[{self.name}] {message}")

    def is_dry_run(self) -> bool:
        """Retourne True si le broker est en mode simulation."""
        return self.dry_run

    def compute_order_amount(self, price: float) -> float:
        """
        Calcule la quantité à acheter en fonction du prix et du solde disponible.

        amount = min(max_position_size, balance) / price
        """
        balance = self.get_balance(self.base_currency)
        budget = min(self.max_position_size, balance)
        if budget < self.min_position_size:
            return 0

        if price <= 0:
            raise ValueError("Le prix doit être positif.")

        return budget / price

