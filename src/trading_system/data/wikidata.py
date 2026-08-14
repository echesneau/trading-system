import requests
import time
import random

def wikidata_query(query, endpoint="https://query.wikidata.org/sparql",
                   max_retries=5, timeout=30):
    headers = {
        "User-Agent": "euronext-universe-builder/1.0"
    }
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(
                endpoint,
                params={"query": query, "format": "json"},
                headers=headers,
                timeout=timeout
            )
            r.raise_for_status()
            return r.json()

        except Exception as e:
            last_error = e
            print(f"⚠️ Wikidata error (attempt {attempt}/{max_retries}): {e}")

            # Backoff exponentiel + jitter
            sleep_time = (2 ** attempt) + random.random()
            time.sleep(sleep_time)

    raise RuntimeError(f"❌ Wikidata failed after {max_retries} attempts. Last error: {last_error}")
