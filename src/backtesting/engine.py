# src/backtesting/engine.py
import pandas as pd
from typing import Dict, Any


class BacktestingEngine:
    """Moteur de backtesting pour évaluer les stratégies de trading."""

    def __init__(self, strategy, data: pd.DataFrame, initial_capital: float = 10000,
                 transaction_fee: float = 0.001, position_size: float = 0.1,
                 stop_loss: float = None, take_profit: float = None):
        """
        Initialise le moteur de backtesting.

        Args:
            strategy: Instance de la stratégie de trading à tester
            data: DataFrame contenant les données de marché
            initial_capital: Capital initial pour le backtest
            transaction_fee: Frais de transaction en pourcentage
            position_size: Taille de position en pourcentage du capital
            stop_loss: Niveau de stop-loss en pourcentage
            take_profit: Niveau de take-profit en pourcentage
        """
        self.strategy = strategy
        self.data = data
        self.initial_capital = initial_capital
        self.transaction_fee = transaction_fee
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def run(self) -> Dict[str, Any]:
        """Exécute le backtest et retourne les résultats."""
        # Initialisation des variables
        capital = self.initial_capital
        position = 0
        entry_price = 0
        portfolio_values = []
        trades = []

        # Pré-calcul des signaux
        signals = self.strategy.generate_signals(self.data)

        for i, (date, row) in enumerate(self.data.iterrows()):
            price = row['Close']
            signal = signals.iloc[i]

            # Gestion des positions existantes
            if position > 0:
                # Vérifier stop-loss
                if self.stop_loss and price <= entry_price * (1 - self.stop_loss):
                    trades.append({
                        'date': date,
                        'action': 'SELL',
                        'price': price,
                        'shares': position,
                        'reason': 'stop_loss'
                    })
                    capital += position * price * (1 - self.transaction_fee)
                    position = 0

                # Vérifier take-profit
                elif self.take_profit and price >= entry_price * (1 + self.take_profit):
                    trades.append({
                        'date': date,
                        'action': 'SELL',
                        'price': price,
                        'shares': position,
                        'reason': 'take_profit'
                    })
                    capital += position * price * (1 - self.transaction_fee)
                    position = 0

                # Vérifier signal de vente
                elif signal == 'SELL':
                    trades.append({
                        'date': date,
                        'action': 'SELL',
                        'price': price,
                        'shares': position,
                        'reason': 'signal'
                    })
                    capital += position * price * (1 - self.transaction_fee)
                    position = 0

            # Gestion des nouveaux signaux d'achat
            if signal == 'BUY' and position == 0 and capital > 0:
                max_invest = capital * self.position_size
                shares = max_invest // price

                if shares > 0:
                    cost = shares * price * (1 + self.transaction_fee)
                    capital -= cost
                    position = shares
                    entry_price = price

                    trades.append({
                        'date': date,
                        'action': 'BUY',
                        'price': price,
                        'shares': shares,
                        'reason': 'signal'
                    })

            # Calcul de la valeur du portefeuille
            portfolio_value = capital + (position * price)
            portfolio_values.append({
                'date': date,
                'value': portfolio_value,
                'position': position,
                'price': price
            })

        # Préparation des résultats
        results = {
            'portfolio': pd.DataFrame(portfolio_values).set_index('date'),
            'trades': pd.DataFrame(trades),
            'performance': self._calculate_performance(portfolio_values)
        }

        return results

    def _calculate_performance(self, portfolio_values: list) -> Dict[str, float]:
        """Calcule les métriques de performance."""
        if not portfolio_values:
            return {}

        start_value = portfolio_values[0]['value']
        end_value = portfolio_values[-1]['value']

        return {
            'return': (end_value - start_value) / start_value,
            'annualized_return': self._annualized_return(portfolio_values),
            'max_drawdown': self._max_drawdown(portfolio_values),
            'sharpe_ratio': self._sharpe_ratio(portfolio_values)
        }

    def _annualized_return(self, values: list) -> float:
        """Calcule le rendement annualisé."""
        # Implémentation simplifiée
        days = (values[-1]['date'] - values[0]['date']).days
        total_return = (values[-1]['value'] - values[0]['value']) / values[0]['value']
        return (1 + total_return) ** (365 / days) - 1

    def _max_drawdown(self, values: list) -> float:
        """Calcule le drawdown maximum."""
        peak = values[0]['value']
        max_dd = 0.0

        for val in values:
            if val['value'] > peak:
                peak = val['value']
            dd = (peak - val['value']) / peak
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def _sharpe_ratio(self, values: list) -> float:
        """Calcule le ratio de Sharpe simplifié."""
        returns = []
        for i in range(1, len(values)):
            ret = (values[i]['value'] - values[i - 1]['value']) / values[i - 1]['value']
            returns.append(ret)

        if not returns:
            return 0.0

        avg_return = sum(returns) / len(returns)
        std_dev = (sum((x - avg_return) ** 2 for x in returns) / len(returns)) ** 0.5

        return avg_return / std_dev if std_dev != 0 else 0.0