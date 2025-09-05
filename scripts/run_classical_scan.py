import os
import json

from trading_system import config_path
from trading_system.data.loader import get_all_ticker_parameters_from_config, load_validation_results
from trading_system.notifications.email_sender import EmailSender
from trading_system.notifications.reporter import SignalReporter
from trading_system.strategies.classical import ClassicalStrategy
from trading_system.data import load_yfinance_data

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
    classical_config_path = f"{config_path}/classical_strategy/"
    validator_path = f"{config_path}/validation_classical_strategy.json"
    # read all parameters
    config = get_all_ticker_parameters_from_config(classical_config_path)
    # read status: if parameters are validated or not
    validator = load_validation_results(validator_path)
    # filter only validated parameters
    config = {k: v for k, v in config.items() if k in validator and validator[k]['valid'] == True}
    # Générer le rapport
    reporter = SignalReporter(strategy=ClassicalStrategy, data_loader=load_yfinance_data)
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