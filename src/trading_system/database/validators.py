import sqlite3
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import pandas as pd

class StrategyValidationRepository:
    """
    Repository pour la validation ex-post des stratégies de trading.

    Cette table indique si la meilleure stratégie pour un ticker
    est jugée exploitable sur la dernière période de validation.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def create_table(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_validation (
                    ticker TEXT PRIMARY KEY,
                    valid INTEGER NOT NULL,
                    reason TEXT,
                    validated_at TEXT NOT NULL
                )
            """)

    def upsert(
        self,
        ticker: str,
        valid: bool,
        reason: Optional[str] = None
    ) -> None:
        """
        Insère ou met à jour la validation d'une stratégie.
        """
        self.create_table()

        with self._connect() as conn:
            conn.execute("""
                INSERT INTO strategy_validation (
                    ticker, valid, reason, validated_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    valid = excluded.valid,
                    reason = excluded.reason,
                    validated_at = excluded.validated_at
            """, (
                ticker,
                int(valid),
                reason,
                datetime.utcnow().isoformat()
            ))

    def fetch_one(self, ticker: str) -> Optional[Dict]:
        self.create_table()

        query = """
            SELECT *
            FROM strategy_validation
            WHERE ticker = ?
        """
        df = pd.read_sql(query, self._connect(), params=(ticker,))

        if df.empty:
            return None

        row = df.iloc[0].to_dict()
        return {
            "ticker": row["ticker"],
            "valid": bool(row["valid"]),
            "reason": row["reason"],
            "validated_at": row["validated_at"],
        }

    def fetch_all(self) -> List[Dict]:
        self.create_table()

        df = pd.read_sql(
            "SELECT * FROM strategy_validation",
            self._connect()
        )

        return [
            {
                "ticker": row["ticker"],
                "valid": bool(row["valid"]),
                "reason": row["reason"],
                "validated_at": row["validated_at"],
            }
            for row in df.to_dict(orient="records")
        ]
