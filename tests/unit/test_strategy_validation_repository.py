import sqlite3

def test_create_table(repo_validation):
    repo_validation.create_table()

    with repo_validation._connect() as conn:
        cursor = conn.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type='table' AND name='strategy_validation'
        """)
        assert cursor.fetchone() is not None

def test_fetch_one_returns_none_if_not_found(repo_validation):
    result = repo_validation.fetch_one("ACA.PA")
    assert result is None

def test_upsert_insert(repo_validation):
    repo_validation.upsert(
        ticker="ACA.PA",
        valid=True,
        reason="Stratégie rentable"
    )

    result = repo_validation.fetch_one("ACA.PA")

    assert result["ticker"] == "ACA.PA"
    assert result["valid"] is True
    assert result["reason"] == "Stratégie rentable"
    assert result["validated_at"] is not None

def test_upsert_update_existing(repo_validation):
    repo_validation.upsert("ACA.PA", True, "OK")
    repo_validation.upsert("ACA.PA", False, "Drawdown trop élevé")

    result = repo_validation.fetch_one("ACA.PA")

    assert result["valid"] is False
    assert result["reason"] == "Drawdown trop élevé"

def test_upsert_is_idempotent(repo_validation):
    repo_validation.upsert("ACA.PA", True, "OK")
    repo_validation.upsert("ACA.PA", True, "OK")
    repo_validation.upsert("ACA.PA", True, "OK")

    all_rows = repo_validation.fetch_all()
    assert len(all_rows) == 1

def test_fetch_all(repo_validation):
    repo_validation.upsert("ACA.PA", True, "OK")
    repo_validation.upsert("BNP.PA", False, "Sharpe trop faible")

    results = repo_validation.fetch_all()

    assert len(results) == 2

    tickers = set(results['ticker'].tolist())
    assert tickers == {"ACA.PA", "BNP.PA"}

def test_valid_is_boolean(repo_validation):
    repo_validation.upsert("ACA.PA", True)
    result = repo_validation.fetch_one("ACA.PA")

    assert isinstance(result["valid"], bool)

def test_reason_can_be_none(repo_validation):
    repo_validation.upsert("ACA.PA", True)

    result = repo_validation.fetch_one("ACA.PA")

    assert result["reason"] is None
