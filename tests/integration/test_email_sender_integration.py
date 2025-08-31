# tests/integration/test_email_integration.py
import pytest
import os
import json
from trading_system.notifications.email_sender import EmailSender


def test_smtp_connection_only():
    """Test seulement la connexion SMTP sans envoyer d'email."""
    smtp_server = os.getenv("REAL_EMAIL_SMTP_SERVER")
    smtp_port = int(os.getenv("REAL_EMAIL_SMTP_PORT", "465"))
    login = os.getenv("REAL_EMAIL_LOGIN")
    password = os.getenv("REAL_EMAIL_PASSWORD")

    # Test simple de connexion
    try:
        import smtplib
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(login, password)
            # Si on arrive ici, la connexion et l'authentification sont OK
            assert True
    except Exception as e:
        pytest.fail(f"La connexion SMTP a échoué: {e}")


def test_send_real_email():
    """
    Test d'intégration qui envoie un VRAI email de test.

    Ce test vérifie que:
    1. La configuration SMTP est correcte
    2. Les identifiants sont valides
    3. L'envoi d'email fonctionne en environnement réel

    Pour exécuter ce test:
    export REAL_EMAIL_SMTP_SERVER="smtp.gmail.com"
    export REAL_EMAIL_LOGIN="votre.email@gmail.com"
    export REAL_EMAIL_PASSWORD="votre-mot-de-passe-application"
    export REAL_EMAIL_TO=["email1@test.com", "email2@test.com"]
    pytest tests/integration/test_email_integration.py -v -m integration
    """
    # Récupération des paramètres depuis les variables d'environnement
    smtp_server = os.getenv("REAL_EMAIL_SMTP_SERVER")
    smtp_port = int(os.getenv("REAL_EMAIL_SMTP_PORT", "465"))  # Port par défaut SSL
    login = os.getenv("REAL_EMAIL_LOGIN")
    password = os.getenv("REAL_EMAIL_PASSWORD")
    to_emails = json.loads(os.getenv("REAL_EMAIL_TO"))

    # Initialisation du sender avec les VRAIS identifiants

    email_sender = EmailSender(
        smtp_server=smtp_server,
        smtp_port=smtp_port,
        login=login,
        password=password
    )

    # Création d'un email de test
    subject = "✅ Test d'Intégration - Système de Trading"
    html_body = """
    <h2>Test d'Intégration Réussi !</h2>
    <p>Ceci est un email de test automatique envoyé par votre système de trading.</p>
    <p>Si vous recevez ce message, cela signifie que:</p>
    <ul>
        <li>Votre configuration SMTP est correcte ✅</li>
        <li>Vos identifiants sont valides ✅</li>
        <li>Le module EmailSender fonctionne parfaitement ✅</li>
    </ul>
    <p><strong>Date d'envoi:</strong> {date}</p>
    <p><strong>Environnement:</strong> Test d'intégration</p>
    """.format(date=os.getenv("GITHUB_SHA", "Local"))

    # Tentative d'envoi
    success = email_sender.send_email(
        to_emails=to_emails,
        subject=subject,
        html_body=html_body
    )

    # Vérification
    assert success, "L'envoi de l'email réel a échoué. Vérifiez la configuration SMTP."