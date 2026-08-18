import pytest
from trading_system.execution.kraken_broker import KrakenBroker

class FakeExchange:
    def __init__(self):
        self.balance = {"EUR": 100}
        self.positions = []
        self.last_order = None
        self.open_orders = []
        self.cancelled = []

    def fetch_balance(self):
        return self.balance

    def fetch_positions(self):
        return self.positions

    def create_order(self, symbol, type, side, amount, params=None, price=None):
        self.last_order = {
            "symbol": symbol,
            "type": type,
            "side": side,
            "amount": amount,
            "params": params,
            "price": price,
        }
        return self.last_order

    def fetch_open_orders(self):
        return self.open_orders

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        return {"id": order_id, "status": "canceled"}

def test_initialization(monkeypatch):
    fake = FakeExchange()
    monkeypatch.setattr("ccxt.kraken", lambda config: fake)

    broker = KrakenBroker(dry_run=True, base_currency="EUR", max_position_size=100)

    assert broker.exchange is fake
    assert broker.dry_run is True
    assert broker.base_currency == "EUR"
    assert broker.max_position_size == 100

def test_get_balance(monkeypatch):
    fake = FakeExchange()
    fake.balance = {"EUR": {"free": 123}}

    monkeypatch.setattr("ccxt.kraken", lambda config: fake)

    broker = KrakenBroker()
    assert broker.get_balance("EUR") == 123

def test_get_balance_missing_currency(monkeypatch):
    fake = FakeExchange()
    fake.balance = {"USD": 50}

    monkeypatch.setattr("ccxt.kraken", lambda config: fake)

    broker = KrakenBroker()
    assert broker.get_balance("EUR") == 0

def test_get_balance_error(monkeypatch):
    class BrokenExchange(FakeExchange):
        def fetch_balance(self):
            raise Exception("network error")

    monkeypatch.setattr("ccxt.kraken", lambda config: BrokenExchange())

    broker = KrakenBroker()
    with pytest.raises(Exception):
        broker.get_balance("EUR")

def test_market_order_dry_run(monkeypatch):
    fake = FakeExchange()
    monkeypatch.setattr("ccxt.kraken", lambda config: fake)

    broker = KrakenBroker(dry_run=True)

    result = broker.place_market_order("BTC/EUR", "buy", 0.01)

    assert result["status"] == "dry_run"
    assert fake.last_order is None

def test_market_order_real(monkeypatch):
    fake = FakeExchange()
    monkeypatch.setattr("ccxt.kraken", lambda config: fake)

    broker = KrakenBroker(dry_run=False)

    order = broker.place_market_order("BTC/EUR", "buy", 0.01)

    assert order["symbol"] == "BTC/EUR"
    assert order["type"] == "market"
    assert order["side"] == "buy"
    assert order["amount"] == 0.01

def test_stop_loss(monkeypatch):
    fake = FakeExchange()
    monkeypatch.setattr("ccxt.kraken", lambda config: fake)

    broker = KrakenBroker(dry_run=False)

    order = broker.place_stop_loss("BTC/EUR", 0.01, 25000)

    assert order["type"] == "stop-loss-limit"
    assert order["params"]["stopPrice"] == 25000

def test_get_open_positions(monkeypatch):
    fake = FakeExchange()
    fake.positions = [{"symbol": "BTC/EUR", "amount": 0.01}]

    monkeypatch.setattr("ccxt.kraken", lambda config: fake)

    broker = KrakenBroker()
    positions = broker.get_open_positions()

    assert positions == fake.positions

def test_get_open_orders(monkeypatch):
    fake = FakeExchange()
    fake.open_orders = [
        {"id": "A1", "symbol": "BTC/EUR"},
        {"id": "B2", "symbol": "ETH/EUR"},
    ]

    monkeypatch.setattr("ccxt.kraken", lambda config: fake)

    broker = KrakenBroker()
    assert broker.get_open_orders() == fake.open_orders

def test_cancel_order_dry_run(monkeypatch):
    fake = FakeExchange()
    monkeypatch.setattr("ccxt.kraken", lambda config: fake)

    broker = KrakenBroker(dry_run=True)

    result = broker.cancel_order("A1")

    assert result["status"] == "dry_run"
    assert result["order_id"] == "A1"
    assert fake.cancelled == []  # rien ne doit être envoyé

def test_cancel_order_real(monkeypatch):
    fake = FakeExchange()
    monkeypatch.setattr("ccxt.kraken", lambda config: fake)

    broker = KrakenBroker(dry_run=False)

    result = broker.cancel_order("A1")

    assert result["id"] == "A1"
    assert result["status"] == "canceled"
    assert fake.cancelled == ["A1"]

def test_find_order_id_symbol_only(monkeypatch):
    fake = FakeExchange()
    fake.open_orders = [
        {"id": "A1", "symbol": "BTC/EUR", "type": "limit", "side": "buy"},
        {"id": "B2", "symbol": "BTC/EUR", "type": "stop-loss", "side": "sell"},
    ]

    monkeypatch.setattr("ccxt.kraken", lambda config: fake)

    broker = KrakenBroker()

    # Avec ton code actuel, side=None => aucun match
    assert broker.find_order_id("BTC/EUR") == ["A1", "B2"]

def test_find_order_id_full_filters(monkeypatch):
    fake = FakeExchange()
    fake.open_orders = [
        {"id": "A1", "symbol": "BTC/EUR", "type": "limit", "side": "buy"},
        {"id": "B2", "symbol": "BTC/EUR", "type": "stop-loss", "side": "sell"},
    ]

    monkeypatch.setattr("ccxt.kraken", lambda config: fake)

    broker = KrakenBroker()

    result = broker.find_order_id("BTC/EUR", type="stop-loss", side="sell")
    assert result == ["B2"]

def test_find_order_id_type_only(monkeypatch):
    fake = FakeExchange()
    fake.open_orders = [
        {"id": "A1", "symbol": "BTC/EUR", "type": "limit", "side": "buy"},
    ]

    monkeypatch.setattr("ccxt.kraken", lambda config: fake)

    broker = KrakenBroker()

    assert broker.find_order_id("BTC/EUR", type="limit") == ["A1"]
