import pandas as pd

def sparql_to_dataframe(results: dict) -> pd.DataFrame:
    """
    Function to transform SPARQL query results into a pandas DataFrame.
    """
    rows = []
    for b in results["results"]["bindings"]:
        row = {}
        for key, val in b.items():
            row[key] = val["value"]
        rows.append(row)
    return pd.DataFrame(rows)

