import ccxt
import pandas as pd
import sqlite3
from typing import Optional, Union
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

    def __init__(self, db_path: Union[str, Path],
                 euronext_csv_categ: Optional[Union[str, Path]] = None,
                 euronext_csv_growth_access_path: Optional[Union[str, Path]] = None):
        """
        Initialise le repository.

        Parameters
        ----------
        db_path : str | Path
            Chemin vers la base SQLite.
        euronext_csv_categ_path : str | Path, optional
            Chemin vers le CSV Euronext.
        """
        self.db_path = str(db_path)
        self.euronext_csv_categ_path = euronext_csv_categ
        self.euronext_csv_growth_access_path = euronext_csv_growth_access_path

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

    def update_db(self, crypto=True) -> None:
        """
        Met à jour la base des tickers à partir d'un DataFrame.

        Cette méthode est un point d'entrée haut niveau :
        - vérifie que la table existe
        - applique les mises à jour nécessaires

        Elle est volontairement simple et idempotente.

        """
        self.create_table()
        if self.euronext_csv_categ_path is not None:
            self.bulk_upsert(self.load_euronext_csv(self.euronext_csv_categ_path))
        if self.euronext_csv_growth_access_path is not None:
            self.bulk_upsert(self.load_euronext_csv(self.euronext_csv_growth_access_path))
        if crypto:
            self.bulk_upsert(self.load_crypto_tickers_ccxt())

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

    def get_all_euronext_tickers(self) -> list:
        """
        Récupère tous les tickers Euronext.

        Returns
        -------
        list
            liste des tickers filtrés pour les marchés Euronext.
        """
        market = [
        "Euronext Growth", "Euronext Access", "Euronext_cat_A", "Euronext_cat_B", "Euronext_cat_C"]
        all_tickers = self.fetch_all()
        mask = all_tickers["market"].isin(market)
        return all_tickers.loc[mask, 'ticker'].tolist()

    def get_all_crypto_tickers(self) -> list:
        """
        Récupère tous les tickers Crypto.

        Returns
        -------
        list
            liste des tickers filtrés pour les marchés Crypto.
        """
        market = ["Crypto_EUR", "Crypto_USDT"]
        all_tickers = self.fetch_all()
        mask = all_tickers["market"].isin(market)
        return all_tickers.loc[mask, 'ticker'].tolist()

    @staticmethod
    def load_euronext_csv(
            csv_path: Union[str, Path],
            allowed_exchanges: tuple[str, ...] = (
                    "A", "B", "C",
                    "Euronext Growth",
                    "Euronext Access",
            )
    ) -> pd.DataFrame:
        """
        Charge et nettoie un fichier CSV Euronext pour insertion en base.

        Le CSV doit contenir les colonnes :
        - Nom de l'entreprise
        - Code court
        - Compartiment

        Les étapes effectuées :
        - filtrage sur les places boursières autorisées
        - suppression des lignes incomplètes
        - renommage et normalisation des colonnes
        - renommage des Comportiments
        - renommage des tickers
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

        df = pd.read_csv(csv_path, sep=";", encoding="latin-1")

        required_cols = {"Nom de l'entreprise", "Code court", "Compartiment"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes dans le CSV : {missing}\n Existing columns: {set(df.columns)}")

        # Filtrage des marchés
        df = df[df["Compartiment"].isin(allowed_exchanges)]

        # Nettoyage
        df = df[["Nom de l'entreprise", "Code court", "Compartiment"]].dropna()

        # Normalisation
        df = df.rename(columns={
            "Compartiment": "Market",
            "Nom de l'entreprise": "Company",
            "Code court": "Ticker"
        })

        # Nettoyage des chaînes
        df["Ticker"] = df["Ticker"].str.strip()
        df["Company"] = df["Company"].str.strip()
        df["Market"] = df["Market"].str.strip()

        # Ajouter .PA aux tickers
        df["Ticker"] = df["Ticker"].astype(str) + ".PA"
        df["Market"] = df["Market"].replace({
            "A": "Euronext_cat_A",
            "B": "Euronext_cat_B",
            "C": "Euronext_cat_C"
        })

        # Suppression des doublons
        df = df.drop_duplicates(subset="Ticker", keep="last")

        df = df.reset_index(drop=True)

        return df

    @staticmethod
    def load_crypto_tickers_ccxt():
        exchange_class = getattr(ccxt, 'kraken')
        exchange = exchange_class()
        markets = exchange.load_markets()
        # Liste des symboles dispo
        pairs = list(markets.keys())
        pairs_euro = [pair for pair in pairs if pair.endswith("/EUR")]
        pair_usd = [pair for pair in pairs if pair.endswith("/USDT")]
        df = pd.DataFrame({
            'Ticker': pairs_euro + pair_usd,
            "Company": pairs_euro + pair_usd,
            "Market": ["Crypto_EUR"] * len(pairs_euro) + ["Crypto_USDT"] * len(pair_usd)
        })
        return df
