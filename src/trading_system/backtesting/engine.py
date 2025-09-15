# src/backtesting/engine.py
import pandas as pd
import numpy as np
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
                # Vérifier signal de vente
                if signal == 'SELL':
                    trades.append({
                        'date': date,
                        'action': 'SELL',
                        'price': price,
                        'shares': position,
                        'reason': 'signal'
                    })
                    capital += position * price * (1 - self.transaction_fee)
                    position = 0
                # Vérifier stop-loss
                elif self.stop_loss and price <= entry_price *  self.stop_loss:
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
                elif self.take_profit and price >= entry_price * self.take_profit:
                    trades.append({
                        'date': date,
                        'action': 'SELL',
                        'price': price,
                        'shares': position,
                        'reason': 'take_profit'
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
        if len(portfolio_values) < 2:
            return {
                'return': 0.0,
                'annualized_return': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }

        # Convertir en DataFrame pour faciliter les calculs
        df = pd.DataFrame(portfolio_values).set_index('date')
        # Calcul du rendement total
        start_val = df['value'].iloc[0]
        end_val = df['value'].iloc[-1]
        total_return = (end_val - start_val) / start_val if start_val != 0 else 0.0

        # Calcul du drawdown maximum
        df['peak'] = df['value'].cummax()
        df['drawdown'] = (df['peak'] - df['value']) / df['peak']
        max_drawdown = df['drawdown'].max()

        # Calcul du ratio de Sharpe (simplifié)
        returns = df['value'].pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() if returns.std() != 0 else 0.0

        # Calcul du rendement annualisé
        days = (df.index[-1] - df.index[0]).days
        annualized_return = ((1 + total_return) ** (365 / days) - 1) if days > 0 else 0.0

        return {
            'return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio if not np.isnan(sharpe_ratio) else 0.0
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

    def strategy_score(self, return_pct, drawdown_pct, n_trades, win_rate,
                       w_return=0.4, w_drawdown=0.3, w_trades=0.1, w_winrate=0.2,
                       max_trades_ref=200):
        """
        Calcule un score composite pour une stratégie de trading.

        Args:
            return_pct: rendement total en %
            drawdown_pct: drawdown max en %
            n_trades: nombre de trades
            win_rate: ratio trades gagnants (0-1)
            w_return, w_drawdown, w_trades, w_winrate: poids des critères
            max_trades_ref: valeur de référence pour normaliser le nombre de trades

        Returns:
            score (float) : plus il est élevé, meilleure est la stratégie
        """

        # Normalisation
        return_norm = np.tanh(return_pct / 100)  # borné [-1,1], stabilise gros gains
        drawdown_norm = np.tanh(drawdown_pct / 100)  # borné [0,1]
        trades_norm = min(n_trades / max_trades_ref, 1.0)  # max 1
        winrate_norm = win_rate  # déjà entre 0 et 1

        # Score composite
        score = (
                w_return * return_norm
                - w_drawdown * drawdown_norm
                + w_trades * trades_norm
                + w_winrate * winrate_norm
        )

        return score

    def _compute_trade_metrics(self, trades: pd.DataFrame, fee_rate: float = 0.001):
        """
        Calcule les profits, win/loss ratio et win rate avec frais inclus.

        Args:
            trades: DataFrame avec colonnes ["type", "price"] et alternance buy/sell
            fee_rate: frais de transaction par ordre (ex: 0.001 = 0.1%)

        Returns:
            dict avec profits, nb_gagnants, nb_perdants, win_rate, win_loss_ratio
        """
        # Séparer buy et sell
        buy_prices = trades.loc[trades["type"] == "buy", "price"].reset_index(drop=True)
        sell_prices = trades.loc[trades["type"] == "sell", "price"].reset_index(drop=True)

        # Appliquer les frais :
        # - le buy coûte plus cher
        # - le sell rapporte moins
        buy_adj = buy_prices * (1 + fee_rate)
        sell_adj = sell_prices * (1 - fee_rate)

        # Profit net par trade
        profits = sell_adj.values - buy_adj.values

        # Statistiques
        n_wins = (profits > 0).sum()
        n_losses = (profits <  0).sum()
        win_loss_ratio = n_wins / n_losses if n_losses > 0 else float("inf")
        win_rate = n_wins / (n_wins + n_losses) if (n_wins + n_losses) > 0 else 0

        return {
            "profits": profits,
            "n_wins": int(n_wins),
            "n_losses": int(n_losses),
            "win_rate": win_rate,
            "win_loss_ratio": win_loss_ratio,
            "total_profit": profits.sum()
        }