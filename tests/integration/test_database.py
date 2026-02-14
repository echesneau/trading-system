from trading_system.database.tickers import TickersRepository
from trading_system.database import euronext_csv

def test_integration_load_real_euronext_csv(tmp_path):
    repo = TickersRepository(
        db_path=tmp_path / "test.db",
        euronext_csv_path=euronext_csv
    )

    df = repo.load_euronext_csv(euronext_csv)

    # ---- Sanity checks ----
    assert not df.empty
    assert set(df.columns) == {"Ticker", "Company", "Market"}

    # ---- Invariants métier ----
    assert df["Ticker"].notna().all()
    assert df["Company"].notna().all()
    assert (df["Market"].isin(["Euronext Paris", "Euronext Access Paris"])).all()

    # ---- Cohérence ----
    assert df["Ticker"].is_unique

def test_integration_update_db_real_csv(tmp_path):
    db_path = tmp_path / "euronext.db"

    repo = TickersRepository(
        db_path=db_path,
        euronext_csv_path=euronext_csv
    )

    repo.update_db()

    result = repo.fetch_all()

    assert not result.empty
    assert "ticker" in result.columns
    assert "market" in result.columns

    # Invariants
    assert result["ticker"].is_unique
    assert (result["market"].isin(["Euronext Paris", "Euronext Access Paris"])).all()