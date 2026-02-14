import pandas as pd
import pytest

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


def test_update_db_is_idempotent(repo_tickers):
    df = pd.DataFrame({
        "Ticker": ["ACA.PA"],
        "Company": ["Crédit Agricole"],
        "Market": ["Paris"]
    })

    repo_tickers.update_db(df)
    repo_tickers.update_db(df)  # appel multiple volontaire

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