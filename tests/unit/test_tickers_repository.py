import pandas as pd
import pytest

from trading_system.database.tickers import TickersRepository

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
    repo_tickers.update_db()
    repo_tickers.update_db()  # appel multiple volontaire

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
        euronext_csv_path=euronext_csv
    )

    df = repo.load_euronext_csv(euronext_csv)

    # ---- Assertions structure ----
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"Ticker", "Company", "Market"}

    # ---- Assertions contenu ----
    assert len(df) == 1

    row = df.iloc[0]
    assert row["Ticker"] == "ACA.PA"
    assert row["Company"] == "Crédit Agricole"
    assert row["Market"] == "Euronext Paris"

def test_load_euronext_csv_drops_unused_columns(euronext_csv, tmp_path):
    repo = TickersRepository(
        db_path=tmp_path / "test.db",
        euronext_csv_path=euronext_csv
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