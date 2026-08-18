import os
from datetime import datetime
import pandas as pd
import ccxt
from ccxt import InvalidOrder, InsufficientFunds

from .broker import BrokerBase


class KrakenBroker(BrokerBase):
    """
    Implémentation du broker Kraken via ccxt.
    Gère les ordres au marché, stop-loss, balances et positions.
    """

    def __init__(self, dry_run=True, base_currency="EUR", max_position_size=100, min_position_size=1):
        super().__init__(
            dry_run=dry_run,
            base_currency=base_currency,
            max_position_size=max_position_size,
            min_position_size=min_position_size
        )

        self.exchange = ccxt.kraken({
            "apiKey": os.getenv("KRAKEN_API_KEY"),
            "secret": os.getenv("KRAKEN_API_SECRET"),
            "enableRateLimit": True,
        })

        self.log_order(f"Broker initialisé (dry_run={self.dry_run})")

    # ----------------------------------------------------------------------
    # BALANCE
    # ----------------------------------------------------------------------

    def get_balance(self, currency=None):
        """
        Retourne le solde disponible pour une devise.
        """
        currency = currency or self.base_currency
        balance = self.exchange.fetch_balance()
        currency_balance =  balance.get(currency, {})
        return currency_balance.get("free", 0)


    # ----------------------------------------------------------------------
    # POSITIONS
    # ----------------------------------------------------------------------

    def get_open_positions(self):
        """
        Retourne les positions ouvertes (spot uniquement).
        """
        positions = self.exchange.fetch_positions()
        return positions

    # ----------------------------------------------------------------------
    # Price
    # ----------------------------------------------------------------------
    def get_price(self, ticker):
        """
        Retourne le prix actuel pour un ticker donné.
        """
        ohlcv = self.exchange.fetch_ohlcv(ticker, timeframe="1m", limit=10)
        df = pd.DataFrame(ohlcv, columns=["time", "Open", "High", "Low", "Close", "Volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        return df.iloc[-1]["Close"]

    # ----------------------------------------------------------------------
    # ORDRES
    # ----------------------------------------------------------------------

    def place_market_order(self, symbol, side, amount):
        """
        Passe un ordre au marché.
        symbol: 'BTC/EUR'
        side: 'buy' ou 'sell'
        amount: quantité (float)
        """
        self.log_order(f"Market order demandé: {side.upper()} {amount} {symbol}")

        if self.is_dry_run():
            self.log_order(f"[DRY RUN] Market order ignoré.")
            return {"status": "dry_run", "symbol": symbol, "side": side, "amount": amount}
        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type="market",
                side=side,
                amount=amount,
                params={"validate": False}
            )
            self.log_order(f"Ordre exécuté: {order}")
            return order
        except InvalidOrder as e:
            self.log_order(f"Erreur market order: {e}")
            raise InvalidOrder(e)
        except InsufficientFunds as e:
            self.log_order(f"Erreur market order: {e}")
            raise InsufficientFunds(e)

    def place_stop_loss(self, symbol, amount, stop_price,limit_price=None):
        """
        Place un stop-loss SPOT sur Kraken via un ordre STOP-LIMIT.
        stop_price : prix de déclenchement
        limit_price : prix limite (obligatoire pour Kraken)
        """
        if limit_price is None:
            # Par défaut, on place un prix limite légèrement sous le stop
            limit_price = stop_price * 0.999
        self.log_order(f"Stop-loss demandé: SELL {amount} {symbol} @ {stop_price}, limit={limit_price}")

        if self.is_dry_run():
            self.log_order(f"[DRY RUN] Stop-loss ignoré.")
            return {
                "status": "dry_run",
                "symbol": symbol,
                "amount": amount,
                "stop_price": stop_price,
                "limit_price": limit_price
            }
        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type="stop-loss-limit",
                side="sell",
                amount=amount,
                price=stop_price,
                params={
                    "price2": limit_price,
                    "trigger": "last",
                    "validate": False
                }
            )
            self.log_order(f"Stop-loss placé: {order}")
            return order
        except InvalidOrder as e:
            self.log_order(f"Erreur stop-loss: {e}")
            raise InvalidOrder(e)
        except InsufficientFunds as e:
            self.log_order(f"Erreur stop-loss: {e}")
            raise InsufficientFunds(e)

    def cancel_order(self, order_id: str):
        """
        Annule un ordre ouvert sur Kraken.
        """
        self.log_order(f"Annulation de l'ordre {order_id}")

        if self.is_dry_run():
            return {"status": "dry_run", "order_id": order_id}

        result = self.exchange.cancel_order(order_id)
        self.log_order(f"Ordre annulé: {result}")
        return result

    def get_open_orders(self):
        """
        Retourne la liste des ordres ouverts sur Kraken.
        """
        return self.exchange.fetch_open_orders()

    def find_order_id(self, symbol: str, type: str = None, side: str = None):
        """
        Retourne l'order_id du premier ordre ouvert correspondant aux critères.
        symbol : obligatoire
        type   : optionnel (market, limit, stop-loss, etc.)
        side   : optionnel (buy, sell)
        """
        open_orders = self.get_open_orders()
        orders = []
        for order in open_orders:
            if order.get("symbol") == symbol:
                if type is None or order.get("type") == type:
                    if side is None or order.get("side") == side:
                        orders.append(order.get('id'))
        return orders
