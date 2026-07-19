import pytest

from trading_system.execution.broker import BrokerBase

class FakeBroker(BrokerBase):
    def get_balance(self, currency):
        return 50

    def place_market_order(self, symbol, side, amount):
        pass

    def place_stop_loss(self, symbol, amount, stop_price):
        pass

    def get_open_positions(self):
        return []

def test_brokerbase_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BrokerBase()

def test_initialization():
    broker = FakeBroker(dry_run=True, base_currency="EUR", max_position_size=100)

    assert broker.dry_run is True
    assert broker.base_currency == "EUR"
    assert broker.max_position_size == 100
    assert broker.name == "FakeBroker"
    assert broker.logger is not None

def test_is_dry_run():
    assert FakeBroker(dry_run=True).is_dry_run() is True
    assert FakeBroker(dry_run=False).is_dry_run() is False

def test_log_order(monkeypatch):
    broker = FakeBroker()
    logs = []

    def fake_log(msg):
        logs.append(msg)

    monkeypatch.setattr(broker.logger, "info", fake_log)

    broker.log_order("test message")
    assert "test message" in logs[0]
    assert "FakeBroker" in logs[0]

def test_compute_order_amount_balance_above_max(monkeypatch):
    broker = FakeBroker(max_position_size=100)

    monkeypatch.setattr(broker, "get_balance", lambda _: 200)

    amount = broker.compute_order_amount(price=10)
    assert amount == 10  # 100 / 10

def test_compute_order_amount_balance_below_max(monkeypatch):
    broker = FakeBroker(max_position_size=100)

    monkeypatch.setattr(broker, "get_balance", lambda _: 50)

    amount = broker.compute_order_amount(price=10)
    assert amount == 5  # 50 / 10

def test_compute_order_amount_zero_balance(monkeypatch):
    broker = FakeBroker(max_position_size=100)

    monkeypatch.setattr(broker, "get_balance", lambda _: 0)

    amount = broker.compute_order_amount(price=10)
    assert amount == 0

def test_compute_order_amount_invalid_price():
    broker = FakeBroker()

    with pytest.raises(ValueError):
        broker.compute_order_amount(price=0)

