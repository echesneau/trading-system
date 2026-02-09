# src/backtesting/engine.py
import pandas as pd
import numpy as np
from numba import njit
from typing import Dict, Any

@njit(parallel=True)
def backtest_core(prices, signals, initial_capital, position_size, fee, stop_loss, take_profit):
    """
    Exécute un backtest vectorisé d'une stratégie de trading long-only.

    La fonction simule l'évolution du capital en fonction de signaux d'achat
    et de vente, avec gestion des frais de transaction, du stop-loss et du
    take-profit. Une seule position longue peut être ouverte à la fois.

    Parameters
    ----------
    prices : np.ndarray
        Tableau 1D des prix (ex: prix de clôture).
        Shape : (n,)

    signals : np.ndarray
        Tableau 1D des signaux de trading.
        -  1 : signal d'achat
        - -1 : signal de vente
        -  0 : aucun signal
        Shape : (n,)

    initial_capital : float
        Capital initial disponible au début du backtest.

    position_size : float
        Fraction du capital utilisée pour chaque entrée en position
        (ex: 0.1 pour investir 10 % du capital disponible).

    fee : float
        Frais de transaction proportionnels (ex: 0.001 pour 0.1 % par trade).

    stop_loss : float
        Seuil de stop-loss multiplicatif appliqué au prix d'entrée.
        Exemple : 0.95 signifie une sortie si le prix baisse de 5 %.
        Si <= 0, le stop-loss est désactivé.

    take_profit : float
        Seuil de take-profit multiplicatif appliqué au prix d'entrée.
        Exemple : 1.10 signifie une sortie si le prix augmente de 10 %.
        Si <= 0, le take-profit est désactivé.

    Returns
    -------
    portfolio_values : np.ndarray
        Valeur totale du portefeuille (capital + position valorisée)
        à chaque pas de temps.
        Shape : (n,)

    positions : np.ndarray
        Nombre d'unités détenues à chaque pas de temps.
        Shape : (n,)

    trades : np.ndarray
        Historique des trades exécutés.
        Shape : (n, 4)

        Colonnes :
        - trades[:, 0] : action
            *  1 : achat
            * -1 : vente
            *  0 : aucun trade
        - trades[:, 1] : prix d'exécution
        - trades[:, 2] : taille de la position (nombre d'unités)
        - trades[:, 3] : raison de la sortie / entrée
            * 0 : signal
            * 1 : stop-loss
            * 2 : take-profit

    Notes
    -----
    - La stratégie est long-only (pas de short).
    - Une seule position peut être ouverte à la fois.
    - Les ordres sont exécutés au prix courant (pas de slippage).
    - La fonction est compilée avec Numba (`@njit`) pour des performances élevées
      et est compatible avec une exécution parallèle.
    """
    n = len(prices)
    capital = initial_capital
    position = 0.0
    entry_price = 0.0

    portfolio_values = np.zeros(n)
    positions = np.zeros(n)
    trades = np.zeros((n, 4))
    for i in range(n):
        price = prices[i]
        signal = signals[i]

        # --- GESTION POSITION EXISTANTE ---
        if position > 0:
            if signal == -1:
                trades[i] = [-1, price, position, 0] # Action, price, position, reason (0=signal)
                capital += position * price * (1 - fee)
                position = 0

            elif stop_loss > 0 and price <= entry_price * stop_loss:
                trades[i] = [-1, price, position, 1]  # Action, price, position, reason (1=stop_loss)
                capital += position * price * (1 - fee)
                position = 0

            elif take_profit > 0 and price >= entry_price * take_profit:
                trades[i] = [-1, price, position, 2]  # Action, price, position, reason (2=take_profit)
                capital += position * price * (1 - fee)
                position = 0

        # --- NOUVEL ACHAT ---
        if signal == 1 and position == 0 and capital > 0:
            max_invest = capital * position_size
            shares = np.floor(max_invest / price)

            if shares > 0:
                cost = shares * price * (1 + fee)
                capital -= cost
                position = shares
                entry_price = price
                trades[i] = [1, price, position, 0]  # Action, price, position, reason (0=signal)

        portfolio_values[i] = capital + position * price
        positions[i] = position
    return portfolio_values, positions, trades

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

    def run_numba(self) -> Dict[str, Any]:
        """Exécute le backtest optimisé avec numba et retourne les résultats."""
        prices = self.data["Close"].values
        signals = self.strategy.generate_signals(self.data).values
        dates = self.data.index

        portfolio_values, positions, trades = backtest_core(
            prices,
            signals,
            self.initial_capital,
            self.position_size,
            self.transaction_fee,
            self.stop_loss if self.stop_loss else 0.0,
            self.take_profit if self.take_profit else 0.0,
        )

        portfolio_df = pd.DataFrame({
            "date": dates,
            "value": portfolio_values,
            "position": positions,
            "price": prices
        }, index=dates)

        trades_df = pd.DataFrame(trades, columns=["action", "price", "shares", "reason"])
        trades_df['date'] = dates
        trades_df = trades_df[trades_df["action"] != 0]

        results = {
            "portfolio": portfolio_df,
            "trades": trades_df,
            "performance": self._calculate_performance(portfolio_df, trades_df)
        }

        return results

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
            'performance': self._calculate_performance(portfolio_values, pd.DataFrame(trades))
        }

        return results

    def _calculate_performance(self, portfolio_values: list, trades: pd.DataFrame) -> Dict[str, float]:
        """Calcule les métriques de performance."""
        if len(portfolio_values) < 2:
            return {
                'return': 0.0,
                'annualized_return': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }

        # Convertir en DataFrame pour faciliter les calculs
        if isinstance(portfolio_values, pd.DataFrame):
            df = portfolio_values
        else:
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

        # calcum des métriques de trades
        trade_metrics = self._compute_trade_metrics(trades, fee_rate=self.transaction_fee)

        # custom strategy score
        strategy_score = self.strategy_score(annualized_return * 100, total_return * 100, max_drawdown * 100,
                                             len(trades), trade_metrics['win_rate'])

        return {
            'return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio if not np.isnan(sharpe_ratio) else 0.0,
            'trade_metrics': trade_metrics,
            'strategy_score': strategy_score
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

    def strategy_score(self, annualized_return, total_return, drawdown_pct, n_trades, win_rate,
                       w_cagr=0.35, w_total_return=0.1, w_drawdown=0.25, w_trades=0.15, w_winrate=0.15,
                       max_trades_ref_per_year=20):
        """
        Calcule un score composite pour une stratégie de trading.

        Args:
            annualized_return: rendement annualisé en % (CAGR)
            total_return: rendement total en %
            drawdown_pct: drawdown max en %
            n_trades: nombre de trades
            win_rate: ratio trades gagnants (0-1)
            w_cagr, w_total_return, w_drawdown, w_trades, w_winrate: poids des critères
            max_trades_ref_per_year: valeur de référence pour normaliser le nombre de trades

        Returns:
            score (float) : plus il est élevé, meilleure est la stratégie
        """

        # Normalisation
        cagr_norm = np.tanh(annualized_return / 100)
        return_norm = np.tanh(total_return / 100)  # borné [-1,1], stabilise gros gains
        drawdown_norm = np.tanh(drawdown_pct / 100)  # borné [0,1]
        # Durée du test
        test_years = self._get_test_years()
        # Fréquence de trading par an
        trades_per_year = n_trades / test_years
        trades_norm = min(trades_per_year / max_trades_ref_per_year, 1.0)
        winrate_norm = win_rate  # déjà entre 0 et 1

        # Score composite
        score = (
            w_cagr * cagr_norm
                + w_total_return * return_norm
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
        # Cas vide ou None
        if trades is None or trades.empty:
            return {
                "profits_by_trades": np.array([], dtype=float),
                "n_wins": 0,
                "n_losses": 0,
                "win_rate": 0.0,
                "win_loss_ratio": 0.0,
                "total_profit_by_trades": 0.0
            }
        actions = trades["action"].values
        prices = trades["price"].values
        # Séparer buy et sell
        # avec frais
        # - le buy coûte plus cher
        # - le sell rapporte moins
        buy_prices = prices[actions == 1] * (1 + fee_rate)
        sell_prices = prices[actions == -1] * (1 - fee_rate)

        # Profit net par trade
        n = min(len(buy_prices), len(sell_prices))
        profits = sell_prices[:n] - buy_prices[:n]

        # Statistiques
        n_wins = np.sum(profits > 0)
        n_losses = np.sum(profits < 0)
        win_loss_ratio = n_wins / n_losses if n_losses > 0 else float("inf")
        win_rate = n_wins / (n_wins + n_losses)

        return {
            "profits_by_trades": profits,
            "n_wins": int(n_wins),
            "n_losses": int(n_losses),
            "win_rate": win_rate,
            "win_loss_ratio": win_loss_ratio,
            "total_profit_by_trades": profits.sum()
        }

    def _get_test_years(self):
        """Calcule la durée du backtest en années."""
        if len(self.data.index) < 2:
            return 1e-6  # évite division par zéro
        delta_days = (self.data.index[-1] - self.data.index[0]).days
        return max(delta_days / 365, 1e-6)