from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional, Union
import pandas as pd

from trading_system.database.tickers import TickersRepository


class BestStrategyRepository:
    """
    Repository SQLite pour stocker la meilleure stratégie par ticker.

    Cette table contient exactement une ligne par ticker :
    - les meilleurs paramètres trouvés
    - les métriques associées (train / validation)
    - une date de mise à jour

    La table est volontairement simple et idempotente.
    """

    def __init__(self, db_path: Union[str, Path]):
        """
        Initialise le repository.

        Parameters
        ----------
        db_path : str | Path
            Chemin vers la base SQLite.
        """
        self.db_path = str(db_path)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def create_table(self) -> None:
        """
        Crée la table best_strategy_params si elle n'existe pas.
        """
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS best_strategy_params (
                    ticker TEXT PRIMARY KEY,
                    updated_at TEXT NOT NULL,
                    params_json TEXT NOT NULL,

                    -- TRAIN
                    train_total_return REAL,
                    train_annualized_return REAL,
                    train_sharpe_ratio REAL,
                    train_max_drawdown REAL,
                    train_strategy_score REAL,

                    -- VALIDATION
                    val_total_return REAL,
                    val_annualized_return REAL,
                    val_sharpe_ratio REAL,
                    val_max_drawdown REAL,
                    val_strategy_score REAL
                )
            """)

    def upsert(self, result: Dict[str, Any]) -> None:
        """
        Insère ou met à jour la meilleure stratégie pour un ticker.

        Parameters
        ----------
        result : dict
            Résultat final de l'optimisation, avec les clés :
            - ticker
            - date
            - params
            - train_results
            - validation_results
        """
        self.create_table()

        with self._connect() as conn:
            conn.execute("""
                INSERT INTO best_strategy_params (
                    ticker, updated_at, params_json,

                    train_total_return,
                    train_annualized_return,
                    train_sharpe_ratio,
                    train_max_drawdown,
                    train_strategy_score,

                    val_total_return,
                    val_annualized_return,
                    val_sharpe_ratio,
                    val_max_drawdown,
                    val_strategy_score
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    params_json = excluded.params_json,

                    train_total_return = excluded.train_total_return,
                    train_annualized_return = excluded.train_annualized_return,
                    train_sharpe_ratio = excluded.train_sharpe_ratio,
                    train_max_drawdown = excluded.train_max_drawdown,
                    train_strategy_score = excluded.train_strategy_score,

                    val_total_return = excluded.val_total_return,
                    val_annualized_return = excluded.val_annualized_return,
                    val_sharpe_ratio = excluded.val_sharpe_ratio,
                    val_max_drawdown = excluded.val_max_drawdown,
                    val_strategy_score = excluded.val_strategy_score
            """, (
                result["ticker"],
                result["date"],
                json.dumps(result["params"], sort_keys=True),

                result["train_results"]["total_return"],
                result["train_results"]["annualized_return"],
                result["train_results"]["sharpe_ratio"],
                result["train_results"]["max_drawdown"],
                result["train_results"]["strategy_score"],

                result["validation_results"]["total_return"],
                result["validation_results"]["annualized_return"],
                result["validation_results"]["sharpe_ratio"],
                result["validation_results"]["max_drawdown"],
                result["validation_results"]["strategy_score"],
            ))

    def fetch_all(self) -> pd.DataFrame:
        """
        Retourne toutes les meilleures stratégies.

        Returns
        -------
        pd.DataFrame
        """
        with self._connect() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM best_strategy_params",
                conn
            )
            df['params_json'] = df['params_json'].apply(json.loads)
            return df

    def fetch_one(self, ticker: str) -> Optional[pd.Series]:
        """
        Retourne la stratégie associée à un ticker.

        Parameters
        ----------
        ticker : str

        Returns
        -------
        pd.Series | None
        """
        with self._connect() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM best_strategy_params WHERE ticker = ?",
                conn,
                params=(ticker,)
            )
        if df.empty:
            return None
        df['params_json'] = df['params_json'].apply(json.loads)
        return df.iloc[0]

    def delete_ticker(self, ticker: str, confirm: bool = True) -> None:
        """
        Méthode pour supprimer un ticker de la db.

        Parameters
        ----------
        ticker: str
            ticker à supprimer
        confirm : bool, optional
            Si True, demande une confirmation manuelle dans le terminal.
            Si False, supprime directement (utile pour les tests ou le CI).
        Returns
        -------
        None
        """
        if confirm:
            user_input = input(f"Supprimer le ticker '{ticker}' ? (y/yes pour confirmer) : ").strip().lower()
            if user_input.lower() not in ("y", "yes"):
                print("Suppression annulée.")
                return
        with self._connect() as conn:
            conn.execute("DELETE FROM best_strategy_params WHERE ticker = ?", (ticker,))

    def validate_existing_tickers(self, tickers_db: TickersRepository, confirm: bool = True)  -> None:
        """
        Vérifie que les tickers en base sont toujours valides, sinon le supprime

        Parameters
        ----------
        tickers_db: TickersRepository
            Repository with available tickers
        confirm : bool, optional
            Si True, demande une confirmation manuelle dans le terminal.
            Si False, supprime directement (utile pour les tests ou le CI).
        Returns
        -------
        None
        """
        tickers_df = tickers_db.fetch_all()
        tickers_available = tickers_df["ticker"].unique()

        df = self.fetch_all()

        to_delete = []

        for _, row in df.iterrows():
            ticker = row["ticker"]
            if ticker not in tickers_available:
                to_delete.append(ticker)

        for ticker in to_delete:
            self.delete_ticker(ticker, confirm=confirm)
