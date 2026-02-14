import json

def test_create_table(repo_strategy):
    repo_strategy.create_table()

    df = repo_strategy.fetch_all()
    assert df.empty

def test_upsert_inserts_row(repo_strategy, example_optim_results):
    repo_strategy.upsert(example_optim_results)

    df = repo_strategy.fetch_all()

    assert len(df) == 1
    assert df.loc[0, "ticker"] == "ACA.PA"

def test_upsert_is_idempotent(repo_strategy, example_optim_results):
    repo_strategy.upsert(example_optim_results)
    repo_strategy.upsert(example_optim_results)

    df = repo_strategy.fetch_all()
    assert len(df) == 1

def test_upsert_updates_existing_row(repo_strategy, example_optim_results):
    repo_strategy.upsert(example_optim_results)

    updated = example_optim_results.copy()
    updated["train_results"] = updated["train_results"].copy()
    updated["train_results"]["strategy_score"] = 0.99

    repo_strategy.upsert(updated)

    df = repo_strategy.fetch_all()

    assert len(df) == 1
    assert df.loc[0, "train_strategy_score"] == 0.99

def test_params_are_stored_as_json(repo_strategy, example_optim_results):
    repo_strategy.upsert(example_optim_results)

    row = repo_strategy.fetch_one("ACA.PA")

    params = row["params_json"]
    assert params["rsi_window"] == 7
    assert params["bollinger_std"] == 1

def test_fetch_one_returns_none_if_not_found(repo_strategy):
    repo_strategy.create_table()
    assert repo_strategy.fetch_one("UNKNOWN") is None

def test_multiple_tickers(repo_strategy, example_optim_results):
    r1 = example_optim_results

    r2 = example_optim_results.copy()
    r2["ticker"] = "BNP.PA"

    repo_strategy.upsert(r1)
    repo_strategy.upsert(r2)

    df = repo_strategy.fetch_all()

    assert len(df) == 2
    assert set(df["ticker"]) == {"ACA.PA", "BNP.PA"}

def test_schema_columns(repo_strategy, example_optim_results):
    repo_strategy.upsert(example_optim_results)
    df = repo_strategy.fetch_all()

    expected_columns = {
        "ticker",
        "updated_at",
        "params_json",
        "train_strategy_score",
        "val_strategy_score"
    }

    assert expected_columns.issubset(df.columns)
