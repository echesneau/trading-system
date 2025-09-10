import ccxt

if __name__ == "__main__":
    exchange = ccxt.binance()
    markets = exchange.load_markets()

    # Liste des symboles dispo
    pairs = list(markets.keys())
    pairs_euro = [pair for pair in pairs if pair.endswith("/EUR")]
    pair_usd = [pair for pair in pairs if pair.endswith("/USDT")]
    print(pairs_euro)  # affiche les 20 premières
    print(f"\nNombre total de paires en Euro: {len(pairs_euro)}")
    print(pair_usd)  # affiche les 20 premières
    print(f"\nNombre total de paires en USD: {len(pair_usd)}")