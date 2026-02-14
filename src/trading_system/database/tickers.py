import pandas as pd
import sqlite3
from typing import Optional
from pathlib import Path
from trading_system.database import db_path, config_path


class TickersRepository:
    """
    Repository SQLite pour la gestion de la table `tickers`.

    Cette classe encapsule toute la logique d'accès à la base de données
    concernant les tickers (actions, crypto, etc.).

    Responsabilités :
    - Création de la table si elle n'existe pas
    - Insertion / mise à jour (UPSERT) de tickers
    - Mise à jour en masse à partir d'un DataFrame

    La classe est conçue pour être :
    - idempotente (appelable plusieurs fois sans effet de bord)
    - compatible CI
    - simple à tester
    """

    def __init__(self, db_path: str | Path):
        """
        Initialise le repository.

        Parameters
        ----------
        db_path : str | Path
            Chemin vers la base SQLite.
        """
        self.db_path = str(db_path)

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def create_table(self) -> None:
        """
        Crée la table `tickers` si elle n'existe pas.

        Schéma :
        - ticker : identifiant unique du ticker (clé primaire)
        - company : nom de l'entreprise ou de l'actif
        - market : marché associé (Paris, Crypto, US, etc.)
        - updated_at : timestamp de dernière mise à jour
        """
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tickers (
                    ticker TEXT PRIMARY KEY,
                    company TEXT NOT NULL,
                    market TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def upsert(
        self,
        ticker: str,
        company: str,
        market: str
    ) -> None:
        """
        Insère ou met à jour un ticker.

        Si le ticker existe déjà, les champs `company`, `market`
        et `updated_at` sont mis à jour.

        Parameters
        ----------
        ticker : str
            Identifiant du ticker (ex: "ACA.PA")
        company : str
            Nom de l'entreprise
        market : str
            Marché associé
        """
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO tickers (ticker, company, market, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(ticker) DO UPDATE SET
                    company = excluded.company,
                    market = excluded.market,
                    updated_at = CURRENT_TIMESTAMP
            """, (ticker, company, market))

    def bulk_upsert(self, df: pd.DataFrame) -> None:
        """
        Insère ou met à jour plusieurs tickers à partir d'un DataFrame.

        Le DataFrame doit contenir les colonnes :
        - 'Ticker'
        - 'Company'
        - 'Market'

        Parameters
        ----------
        df : pd.DataFrame
            Données à insérer ou mettre à jour.
        """
        required_cols = {"Ticker", "Company", "Market"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes : {missing}")

        rows = df[["Ticker", "Company", "Market"]].values.tolist()

        with self._connect() as conn:
            conn.executemany("""
                INSERT INTO tickers (ticker, company, market, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(ticker) DO UPDATE SET
                    company = excluded.company,
                    market = excluded.market,
                    updated_at = CURRENT_TIMESTAMP
            """, rows)

    def update_db(self, df: pd.DataFrame) -> None:
        """
        Met à jour la base des tickers à partir d'un DataFrame.

        Cette méthode est un point d'entrée haut niveau :
        - vérifie que la table existe
        - applique les mises à jour nécessaires

        Elle est volontairement simple et idempotente.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame contenant les tickers à synchroniser.
        """
        self.create_table()
        self.bulk_upsert(df)


    def fetch_all(self) -> pd.DataFrame:
        """
        Récupère l'ensemble des tickers sous forme de DataFrame.

        Returns
        -------
        pd.DataFrame
            Contenu de la table `tickers`.
        """
        with self._connect() as conn:
            return pd.read_sql("SELECT * FROM tickers", conn)
