import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from trading_system.database.trading_params import BestStrategyRepository
from trading_system.database import db_path

if __name__ == "__main__":
    params_db = BestStrategyRepository(db_path)

    data = params_db.fetch_all()
    params_df = pd.DataFrame()
    for index, row in data.iterrows():
        ticker = row['ticker']
        params = pd.Series(row['params_json'], name=ticker)
        params_df = pd.concat([params_df, params.to_frame().T])

    n_cols = 2
    n_rows = int(np.ceil(len(params_df.columns) / n_cols))
    fig,axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, n_rows*3))
    for i, column in enumerate(params_df.columns):
        i_row = i // n_cols
        i_col = i % n_cols
        ax = axs[i_row, i_col]
        params_df[column].value_counts(dropna=False).plot(kind='bar', ax=ax)
        ax.set_title(column)
    plt.tight_layout()
    plt.show()
    print(params_df.value_counts(dropna=False))
