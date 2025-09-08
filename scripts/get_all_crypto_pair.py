import ccxt

if __name__ == "__main__":
    exchange = ccxt.binance()
    markets = exchange.load_markets()

    # Liste des symboles dispo
    pairs = list(markets.keys())
    pairs_euro = [pair for pair in pairs if pair.endswith("/EUR")]
    print(pairs_euro)  # affiche les 20 premières
    print(f"\nNombre total de paires en Euro: {len(pairs_euro)}")