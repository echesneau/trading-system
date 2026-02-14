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

    def __init__(self, db_path: str | Path,
                 euronext_csv_path: Optional[str | Path] = None):
        """
        Initialise le repository.

        Parameters
        ----------
        db_path : str | Path
            Chemin vers la base SQLite.
        euronext_csv_path : str | Path, optional
            Chemin vers le CSV Euronext.
        """
        self.db_path = str(db_path)
        self.euronext_csv_path = euronext_csv_path

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

    def update_db(self) -> None:
        """
        Met à jour la base des tickers à partir d'un DataFrame.

        Cette méthode est un point d'entrée haut niveau :
        - vérifie que la table existe
        - applique les mises à jour nécessaires

        Elle est volontairement simple et idempotente.

        """
        self.create_table()
        self.bulk_upsert(self.load_euronext_csv(self.euronext_csv_path))


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

    @staticmethod
    def load_euronext_csv(
            csv_path: str | Path,
            allowed_exchanges: tuple[str, ...] = (
                    "Euronext Paris",
                    "Euronext Access Paris",
            )
    ) -> pd.DataFrame:
        """
        Charge et nettoie un fichier CSV Euronext pour insertion en base.

        Le CSV doit contenir les colonnes :
        - Company
        - Ticker
        - Exchange
        - Currency

        Les étapes effectuées :
        - filtrage sur les places boursières autorisées
        - suppression des lignes incomplètes
        - renommage et normalisation des colonnes
        - suppression des doublons sur le ticker

        Parameters
        ----------
        csv_path : str or Path
            Chemin vers le fichier CSV Euronext.
        allowed_exchanges : tuple of str, optional
            Liste des places boursières à conserver.

        Returns
        -------
        pd.DataFrame
            DataFrame avec les colonnes normalisées :
            ["Ticker", "Company", "Market"]

        Raises
        ------
        ValueError
            Si une colonne requise est manquante.
        """
        csv_path = Path(csv_path)

        df = pd.read_csv(csv_path)

        required_cols = {"Company", "Ticker", "Exchange"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")

        # Filtrage des marchés
        df = df[df["Exchange"].isin(allowed_exchanges)]

        # Nettoyage
        df = df[["Ticker", "Company", "Exchange"]].dropna()

        # Normalisation
        df = df.rename(columns={
            "Exchange": "Market"
        })

        # Nettoyage des chaînes
        df["Ticker"] = df["Ticker"].str.strip()
        df["Company"] = df["Company"].str.strip()
        df["Market"] = df["Market"].str.strip()

        # Suppression des doublons
        df = df.drop_duplicates(subset="Ticker", keep="last")

        df = df.reset_index(drop=True)

        return df
