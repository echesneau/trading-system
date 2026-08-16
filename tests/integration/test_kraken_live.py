import os

import pandas as pd
import pytest
from ccxt import InvalidOrder, InsufficientFunds

from trading_system.execution.kraken_broker import KrakenBroker

def test_kraken_initialization():
    broker = KrakenBroker(dry_run=True)
    assert broker.exchange is not None

def test_kraken_get_balance():
    broker = KrakenBroker(dry_run=True)
    balance = broker.get_balance("EUR")

    assert isinstance(balance, (int, float))
    assert balance > 0

def test_kraken_get_positions():
    broker = KrakenBroker(dry_run=True)
    positions = broker.get_open_positions()

    assert isinstance(positions, list)

def test_kraken_invalid_order():
    broker = KrakenBroker(dry_run=False)

    # quantité volontairement trop petite
    amount = 1e-20
    with pytest.raises(InvalidOrder):
        result = broker.place_market_order("BTC/EUR", "buy", amount)

    # Pas assez de fonds pour acheter 100 BTC
    with pytest.raises(InsufficientFunds):
        result = broker.place_market_order("BTC/EUR", "buy", 100)


def test_kraken_invalid_stop_loss():
    broker = KrakenBroker(dry_run=False)

    # stop-loss volontairement absurde
    with pytest.raises(InsufficientFunds):
        result = broker.place_stop_loss("BTC/EUR", 0.01, -100)


def test_kraken_get_price():
    ticker = "BTC/EUR"
    broker = KrakenBroker(dry_run=False)
    price = broker.get_price(ticker)
    assert price >= 0

def test_kraken_cancel_order():
    broker = KrakenBroker(dry_run=False)
    actual_price = broker.get_price("ADA/EUR")
    price_order = actual_price / 10
    qtt = 2 / price_order
    order = broker.exchange.create_order(
        symbol="ADA/EUR",
        type="limit",
        side="buy",
        amount=qtt,
        price=price_order)
    assert "id" in order
    open_orders = broker.get_open_orders()
    assert len(open_orders) > 0
    assert order["id"] in broker.find_order_id("ADA/EUR", side="buy")
    broker.cancel_order(order["id"])
    assert order["id"] not in broker.find_order_id("ADA/EUR", side="buy")

