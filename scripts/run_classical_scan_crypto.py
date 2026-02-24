import os
import json

from trading_system.notifications.email_sender import EmailSender
from trading_system.notifications.reporter import SignalReporter
from trading_system.strategies.classical import ClassicalStrategy
from trading_system.data.loader import load_ccxt_data
from trading_system.database import db_path, validator_db_path
from trading_system.database.trading_params import BestStrategyRepository
from trading_system.database.validators import StrategyValidationRepository
from trading_system.database.tickers import TickersRepository


TO_EMAILS = json.loads(os.getenv("REAL_EMAIL_TO_PROD", {}))
REAL_EMAIL_LOGIN = os.getenv("REAL_EMAIL_LOGIN")
REAL_EMAIL_PASSWORD = os.getenv("REAL_EMAIL_PASSWORD")
REAL_EMAIL_SMTP_SERVER = os.getenv("REAL_EMAIL_SMTP_SERVER")
REAL_EMAIL_SMTP_PORT = int(os.getenv("REAL_EMAIL_SMTP_PORT", "465"))

for var in [TO_EMAILS, REAL_EMAIL_LOGIN, REAL_EMAIL_PASSWORD, REAL_EMAIL_SMTP_SERVER]:
    if var is None or var == "":
        raise ValueError(f"❌ La variable d'environnement {var} doit être définie pour envoyer les résultats.")

email_sender = EmailSender(
        smtp_server=REAL_EMAIL_SMTP_SERVER,
        smtp_port=REAL_EMAIL_SMTP_PORT,
        login=REAL_EMAIL_LOGIN,
        password=REAL_EMAIL_PASSWORD
    )

if __name__ == "__main__":
    print("🔎 Début du scan quotidien...")
    tickers_db = TickersRepository(db_path)
    params_db = BestStrategyRepository(db_path)
    validators_db = StrategyValidationRepository(validator_db_path)
    tickers = tickers_db.get_all_crypto_tickers()
    valid_tickers = validators_db.fetch_all()
    mask = valid_tickers['ticker'].isin(tickers)
    valid_tickers = valid_tickers.loc[mask]
    valid_tickers = valid_tickers[valid_tickers['valid']]['ticker'].tolist()
    config = {ticker: params_db.fetch_one(ticker)["params_json"]
              for ticker in valid_tickers}
    # Générer le rapport
    reporter = SignalReporter(strategy=ClassicalStrategy, data_loader=load_ccxt_data)
    report = reporter.generate_daily_report(list(config.keys()), config, max_window_range=100)
    html_report = reporter.format_report_to_html(report)
    success = email_sender.send_email(
        to_emails=TO_EMAILS,
        subject="Rapport Trading Quotidien Stratégie Classique",
        html_body=html_report
    )
    if success:
        print("✅ Processus terminé avec succès.")
    else:
        print("❌ Le processus s'est terminé avec une erreur.")