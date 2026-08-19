import datetime as dt
import random

class AutoExecutor:
    """
    Exécute les signaux générés par le SignalReporter via un Broker.
    """

    def __init__(self, broker, risk_manager:float=None):
        """
        Initialisation

        Parameters
        ----------
        broker
        risk_manager: float
            ratio entre 0 et 1 de risque de perte. Si 0.1, stop_loss à 1-0.1=0.9 du prix d'achat
        """
        self.broker = broker
        self.risk_manager = risk_manager

    def execute_from_report(self, report):
        """
        Exécute les signaux BUY / SELL du rapport.
        """
        executions = {
            "executed": [],
            "errors": []
        }

        # 1. Exécuter les BUY
        for sig in random.sample(report["buy_signals"], len(report["buy_signals"])):
            if not sig['ticker'].endswith("EUR"):
                executions["errors"].append({"signal": sig, "error": "Transaction en € uniquement."})
            else:
                try:
                    result = self.execute_buy(sig)
                    executions["executed"].append(result)
                except Exception as e:
                    executions["errors"].append({"signal": sig, "error": str(e)})

        # 2. Exécuter les SELL
        for sig in random.sample(report["sell_signals"], len(report["sell_signals"])):
            if not sig['ticker'].endswith("EUR"):
                executions["errors"].append({"signal": sig, "error": "Transaction en € uniquement."})
            else:
                try:
                    result = self.execute_sell(sig)
                    if result is not None:
                        executions["executed"].append(result)
                except Exception as e:
                    executions["errors"].append({"signal": sig, "error": str(e)})

        return executions

    def execute_buy(self, sig):
        """
        Exécute un signal d'achat.
        sig = {ticker, signal, price, date}
        """
        ticker = sig['ticker']
        price = self.broker.get_price(ticker)
        amount = self.compute_buy_amount(ticker, price=price)
        if amount > 0:
            order = self.broker.place_market_order(ticker, "buy", amount)

            # Stop-loss automatique si risk manager
            if not self.risk_manager is None and 0 < self.risk_manager < 1:
                sl_price = price * (1 - self.risk_manager)
                sl_order = self.broker.place_stop_loss(ticker, amount, sl_price)

            return {
                "type": "BUY",
                "ticker": sig["ticker"],
                "symbol": ticker,
                "amount": amount,
                "order": order
            }
        else:
            raise ValueError(f"Déjà présent dans le portefeuille")

    def execute_sell(self, sig):
        """
        Exécute un signal de vente.
        """
        ticker = sig['ticker']
        # delete older order
        order_ids = self.broker.find_order_id(ticker, side="sell")
        for order in order_ids:
            self.broker.cancel_order(order)

        amount = self.compute_sell_amount(sig["ticker"])
        if amount > 0:
            order = self.broker.place_market_order(ticker, "sell", amount)
            return {
                "type": "SELL",
                "ticker": sig["ticker"],
                "symbol": ticker,
                "amount": amount,
                "order": order
            }
        else:
            return None

    def compute_buy_amount(self, symbol, price=None):
        """
        Détermine la taille de position à acheter.
        """
        # récupère le prix actuel
        if price is None:
            price = self.broker.get_price(symbol)
        qtt = self.broker.compute_order_amount(price)
        # check if ticker in balance
        ticker = symbol.split("/")[0]
        qtt_balance = self.broker.get_balance(ticker)
        if qtt < qtt_balance:
            return 0
        else:
            return qtt - qtt_balance

    def compute_sell_amount(self, ticker):
        """
        Détermine la taille de position à acheter.
        """
        symbol = ticker.split("/")[0]
        qtt = self.broker.get_balance(symbol)
        return qtt

    def format_execution_report(self, report: dict, executions: dict) -> str:
        """
        Fusionne le rapport de signaux + le rapport d'exécution
        avec tri : exécutés en premier, puis erreurs triées par type.
        """

        html = f"""
        <h2>⚡ Rapport d'Exécution Automatique</h2>
        <p>Date : <strong>{dt.datetime.now().strftime('%Y-%m-%d %H:%M')}</strong></p>
        """

        executed = executions.get("executed", [])
        exec_errors = executions.get("errors", [])
        signal_errors = report.get("errors", [])

        # Indexation
        exec_index = {e["ticker"]: e for e in executed}
        error_index = {err["signal"]["ticker"]: err["error"] for err in exec_errors}

        # Tri : exécutés → erreurs
        all_signals = report["buy_signals"] + report["sell_signals"]

        executed_signals = [s for s in all_signals if s["ticker"] in exec_index]
        error_signals = [s for s in all_signals if s["ticker"] in error_index]

        # tri des erreurs par message
        error_signals.sort(key=lambda s: error_index[s["ticker"]])

        html += "<h3>🟢 Signaux exécutés</h3>"
        if executed_signals:
            html += "<ul>"
            for sig in executed_signals:
                ticker = sig["ticker"]
                price = sig["price"]
                exec_tmp = exec_index[ticker]
                amount = exec_tmp["amount"]
                type_order = exec_tmp["type"]
                date = sig["date"].strftime('%Y-%m-%d')
                # order_id = exec_index[ticker]["order"].get("id", "N/A")
                html += f"""
                                <li>
                                    <strong>{type_order} {ticker}</strong> — Prix: {price:.2f} — Quantity: {amount} — Date: {date}<br>
                                </li>
                                """
                # html += f"""
                # <li>
                #     <strong>{ticker}</strong> — Prix: {price:.2f} — Date: {date}<br>
                #     ✔ Ordre exécuté (ID: {order_id})
                # </li>
                # """
            html += "</ul>"
        else:
            html += "<p>Aucun ordre exécuté.</p>"

        html += "<h3>🔴 Signaux en erreur (triés par type)</h3>"
        if error_signals:
            html += "<ul>"
            for sig in error_signals:
                ticker = sig["ticker"]
                price = sig["price"]
                date = sig["date"].strftime('%Y-%m-%d')
                err_msg = error_index[ticker]

                html += f"""
                <li>
                    <strong>{ticker}</strong> — Prix: {price:.2f} — Date: {date}<br>
                    ❌ Erreur : {err_msg}
                </li>
                """
            html += "</ul>"
        else:
            html += "<p>Aucune erreur.</p>"

        # Erreurs globales du SignalReporter
        if signal_errors:
            html += "<h3>⚠️ Erreurs du SignalReporter</h3><ul>"
            for err in signal_errors:
                html += f"<li><strong>{err['ticker']}</strong> — {err['error']}</li>"
            html += "</ul>"

        return html
