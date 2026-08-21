import datetime as dt
from trading_system.execution.auto_executor import AutoExecutor

class FakeBroker:
    def __init__(self):
        self.market_orders = []
        self.stop_losses = []
        self.cancelled_orders = []
        self.balance = {"BTC": 0.5, "ETH": 2.0}
        self.prices = {"BTC/EUR": 60000, "ETH/EUR": 3000, "ETC/EUR": 20}
        self.open_orders = {"BTC/EUR": ["ORD1", "ORD2"]}

    def get_price(self, symbol):
        return self.prices[symbol]

    def compute_order_amount(self, price):
        return 100 / price  # simulate: invest 100€

    def get_balance(self, symbol):
        return self.balance.get(symbol, 0)

    def place_market_order(self, symbol, side, amount):
        order = {"id": f"ORDER_{side.upper()}", "symbol": symbol, "side": side, "amount": amount}
        self.market_orders.append(order)
        return order

    def place_stop_loss(self, symbol, amount, sl_price):
        sl = {"symbol": symbol, "amount": amount, "stop_price": sl_price}
        self.stop_losses.append(sl)
        return sl

    def find_order_id(self, symbol, side=None):
        return self.open_orders.get(symbol, [])

    def cancel_order(self, order_id):
        self.cancelled_orders.append(order_id)
        return {"id": order_id, "status": "canceled"}

def test_execute_buy():
    broker = FakeBroker()
    executor = AutoExecutor(broker, risk_manager=0.1)

    sig = {
        "ticker": "ETC/EUR",
        "signal": "BUY",
        "price": 60000,
        "date": dt.datetime(2024, 1, 1)
    }

    result = executor.execute_buy(sig)

    assert result["type"] == "BUY"
    assert broker.market_orders[0]["side"] == "buy"
    assert len(broker.stop_losses) == 1  # risk_manager active

def test_execute_sell():
    broker = FakeBroker()
    executor = AutoExecutor(broker)

    sig = {
        "ticker": "BTC/EUR",
        "signal": "SELL",
        "price": 60000,
        "date": dt.datetime(2024, 1, 1)
    }

    result = executor.execute_sell(sig)

    assert result["type"] == "SELL"
    assert broker.cancelled_orders == ["ORD1", "ORD2"]
    assert broker.market_orders[0]["side"] == "sell"

def test_execute_from_report():
    broker = FakeBroker()
    executor = AutoExecutor(broker)

    report = {
        "buy_signals": [
            {"ticker": "ETC/EUR", "signal": "BUY", "price": 60000, "date": dt.datetime(2024, 1, 1)}
        ],
        "sell_signals": [
            {"ticker": "ETH/EUR", "signal": "SELL", "price": 3000, "date": dt.datetime(2024, 1, 1)}
        ],
        "errors": []
    }

    executions = executor.execute_from_report(report)

    assert len(executions["executed"]) == 2
    assert broker.market_orders[1]["side"] == "buy" # Sell first
    assert broker.market_orders[0]["side"] == "sell"

def test_compute_buy_amount():
    broker = FakeBroker()
    executor = AutoExecutor(broker)

    amount = executor.compute_buy_amount("ETC/EUR", price=20)

    assert amount == 100 / 20

def test_compute_sell_amount():
    broker = FakeBroker()
    executor = AutoExecutor(broker)

    amount = executor.compute_sell_amount("BTC/EUR")

    assert amount == broker.balance["BTC"]

def test_format_execution_report():
    broker = FakeBroker()
    executor = AutoExecutor(broker)

    report = {
        "buy_signals": [
            {"ticker": "BTC/EUR", "signal": "BUY", "price": 60000, "date": dt.datetime(2024, 1, 1)}
        ],
        "sell_signals": [
            {"ticker": "ETH/EUR", "signal": "SELL", "price": 3000, "date": dt.datetime(2024, 1, 1)}
        ],
        "errors": []
    }

    executions = {
        "executed": [
            {
                "type": "BUY",
                "ticker": "BTC/EUR",
                "symbol": "BTC/EUR",
                "amount": 0.001,
                "order": {"id": "ORDER_BUY"}
            }
        ],
        "errors": [
            {
                "signal": {"ticker": "ETH/EUR"},
                "error": "InsufficientFunds"
            }
        ]
    }

    html = executor.format_execution_report(report, executions)

    assert "BTC/EUR" in html
    assert "InsufficientFunds" in html
