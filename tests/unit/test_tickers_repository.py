import pandas as pd
import pytest

from trading_system.database.tickers import TickersRepository
from trading_system.database.utils import check_crypto, check_yahoo

def test_create_table(repo_tickers):
    repo_tickers.create_table()

    with repo_tickers._connect() as conn:
        tables = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='tickers'
        """).fetchall()

    assert tables == [("tickers",)]

def test_upsert_insert(repo_tickers):
    repo_tickers.create_table()

    repo_tickers.upsert(
        ticker="ACA.PA",
        company="Crédit Agricole",
        market="Paris"
    )

    df = repo_tickers.fetch_all()

    assert len(df) == 1
    assert df.loc[0, "ticker"] == "ACA.PA"
    assert df.loc[0, "company"] == "Crédit Agricole"
    assert df.loc[0, "market"] == "Paris"

def test_delete_ticker(repo_tickers):
    repo_tickers.create_table()

    repo_tickers.upsert(
        ticker="ACA.PA",
        company="Crédit Agricole",
        market="Paris"
    )
    repo_tickers.delete_ticker("ACA.PA", confirm=False)
    df = repo_tickers.fetch_all()
    assert len(df) == 0

    repo_tickers.upsert(
        ticker="ACA.PA",
        company="Crédit Agricole",
        market="Paris"
    )
    repo_tickers.upsert(
        ticker="TOTO.PA",
        company="test",
        market="Paris"
    )
    repo_tickers.delete_ticker("ACA.PA", confirm=False)
    df = repo_tickers.fetch_all()
    assert len(df) == 1
    assert df.loc[0, "ticker"] == "TOTO.PA"

    repo_tickers.delete_ticker("ACA.PA", confirm=False)


def test_upsert_update_existing(repo_tickers):
    repo_tickers.create_table()

    repo_tickers.upsert("ACA.PA", "Crédit Agricole", "Paris")
    repo_tickers.upsert("ACA.PA", "Crédit Agricole SA", "Euronext Paris")

    df = repo_tickers.fetch_all()

    assert len(df) == 1
    assert df.loc[0, "company"] == "Crédit Agricole SA"
    assert df.loc[0, "market"] == "Euronext Paris"

def test_bulk_upsert(repo_tickers):
    repo_tickers.create_table()

    df = pd.DataFrame({
        "Ticker": ["ACA.PA", "BNP.PA"],
        "Company": ["Crédit Agricole", "BNP Paribas"],
        "Market": ["Paris", "Paris"]
    })

    repo_tickers.bulk_upsert(df)

    result = repo_tickers.fetch_all()

    assert len(result) == 2
    assert set(result["ticker"]) == {"ACA.PA", "BNP.PA"}


def test_update_db_is_idempotent(repo_tickers, tmp_path):
    repo_tickers.update_db(crypto=False, wikidata=False)
    repo_tickers.update_db(crypto=False, wikidata=False)  # appel multiple volontaire

    result = repo_tickers.fetch_all()

    assert len(result) == 1
    assert result.loc[0, "ticker"] == "ACA.PA"


def test_bulk_upsert_missing_columns_raises(repo_tickers):
    repo_tickers.create_table()

    df = pd.DataFrame({
        "Ticker": ["ACA.PA"],
        "Market": ["Paris"]
    })

    with pytest.raises(ValueError):
        repo_tickers.bulk_upsert(df)

def test_load_euronext_csv_filters_and_normalizes(euronext_csv, tmp_path):
    repo = TickersRepository(
        db_path=tmp_path / "test.db",
        euronext_csv_categ=euronext_csv
    )

    df = repo.load_euronext_csv(euronext_csv)

    # ---- Assertions structure ----
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"Ticker", "Company", "Market"}

    # ---- Assertions contenu ----
    assert len(df) == 1

    row = df.iloc[0]
    assert row["Ticker"] == "ACA.PA"
    assert row["Company"] == "Credit Agricole"
    assert row["Market"] == "Euronext_cat_A"

def test_load_euronext_csv_drops_unused_columns(euronext_csv, tmp_path):
    repo = TickersRepository(
        db_path=tmp_path / "test.db",
        euronext_csv_categ=euronext_csv
    )

    df = repo.load_euronext_csv(euronext_csv)

    assert "Exchange" not in df.columns
    assert "Currency" not in df.columns

def test_load_euronext_csv_missing_columns(tmp_path):
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame({"Ticker": ["ACA.PA"]}).to_csv(csv_path, index=False)

    repo = TickersRepository(db_path=tmp_path / "test.db")

    with pytest.raises(ValueError):
        repo.load_euronext_csv(csv_path)

def test_get_all_european_stock_exchange(tmp_path, euronext_csv):
    repo = TickersRepository(
        db_path=tmp_path / "test.db",
        euronext_csv_categ=euronext_csv
    )
    df = repo._get_all_european_stock_exchange()
    for col in ['exchange', 'exchangeLabel', 'countryLabel']:
        assert col in df.columns

    for country in df['countryLabel'].unique():
        assert country in ['Allemagne', 'Belgique', 'Espagne', 'Estonie', 'Finlande',
                           'France', 'Grèce', 'Irlande', 'Italie', 'Lettonie', 'Lituanie',
                           'Luxembourg', 'Norvège', 'Pays-Bas', 'Portugal', 'Royaume-Uni',
                           'Suisse']

    for exchange in ['Euronext', 'bourse de Bruxelles', 'Bourse des valeurs de Madrid',
                     'Euronext Paris', 'CAC Small', 'Euronext Growth Paris', 'Euronext Dublin', "Bourse d'Italie",
                     "bourse d'Oslo", 'Euronext Growth Oslo', 'Euronext Amsterdam', 'Euronext Growth',
                     'Euronext Lisbon', 'bourse de Londres']:
        assert exchange in df['exchangeLabel'].unique()

def test__get_all_european_stock_exchange_wikidata_code(tmp_path, euronext_csv):
    repo = TickersRepository(
        db_path=tmp_path / "test.db",
        euronext_csv_categ=euronext_csv
    )
    result = repo._get_all_european_stock_exchange_wikidata_code()
    assert len(result) > 0
    for code in ['Q842108', 'Q617426', 'Q2385849', 'Q107188657', 'Q107188622', 'Q478720']:
        assert code in result

def test_load_european_tickers_wikidata(tmp_path, euronext_csv):
    repo = TickersRepository(
        db_path=tmp_path / "test.db",
        euronext_csv_categ=euronext_csv
    )
    result = repo.load_european_tickers_wikidata()
    # ---- Assertions structure ----
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"Ticker", "Company", "Market"}
    assert len(result) > 0

    assert len(result['Ticker'].unique()) == len(result)

def test_check_crypto():
    valid_ticker = "BTC/EUR"
    invalid_ticker = "FAKE/TOTO"
    assert check_crypto(valid_ticker)
    assert not check_crypto(invalid_ticker)

def test_check_yahoo():
    valid_ticker = "ACA.PA"
    invalid_ticker = "FAKE.TOTO"
    assert check_yahoo(valid_ticker)
    assert not check_yahoo(invalid_ticker)
