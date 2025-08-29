# tests/unit/test_email_sender.py
import pytest
from unittest.mock import Mock, patch, MagicMock
from trading_system.notifications.email_sender import EmailSender
import smtplib

# Fixture pour une instance d'EmailSender avec des paramètres fictifs
@pytest.fixture
def email_sender():
    """Retourne une instance d'EmailSender configurée pour les tests."""
    # Utilisation de valeurs fictives mais réalistes
    return EmailSender(
        smtp_server="smtp.test.com",
        smtp_port=587,
        login="test@test.com",
        password="fake_password"
    )

# Test pour une connexion et un envoi réussis
def test_send_email_success(email_sender):
    """Test que l'email est envoyé avec succès quand le serveur SMTP fonctionne."""
    # Données de test
    to_emails = ["recipient1@test.com", "recipient2@test.com"]
    subject = "Test Subject"
    html_body = "<p>This is a <b>test</b> email.</p>"

    mock_server = Mock()
    mock_server.login = Mock()
    mock_server.send_message = Mock()

    mock_smtp = MagicMock()
    mock_smtp.__enter__ = Mock(return_value=mock_server)
    mock_smtp.__exit__ = Mock(return_value=None)

    with patch('trading_system.notifications.email_sender.smtplib.SMTP_SSL',
               return_value=mock_smtp) as mock_smtp_constructor:

        result = email_sender.send_email(to_emails, subject, html_body)

    # Assertions
    assert result is True, "La méthode devrait retourner True en cas de succès"

    # Vérifie que le login a été tenté avec les bons identifiants
    mock_smtp_constructor.assert_called_once_with("smtp.test.com", 587)
    mock_server.login.assert_called_once_with("test@test.com", "fake_password")

    # Vérifie que send_message a été appelé une fois.
    # On pourrait vérifier le contenu du message, mais c'est plus complexe.
    assert mock_server.send_message.call_count == 1

    # Récupère l'argument passé à send_message (le message MIME)
    call_args = mock_server.send_message.call_args
    sent_message = call_args[0][0] # Premier argument du premier appel

    # Vérifications basiques sur l'objet message
    assert sent_message['From'] == "test@test.com"
    assert sent_message['To'] == "recipient1@test.com, recipient2@test.com"
    assert sent_message['Subject'] == "Test Subject"
    # Le corps du message est encapsulé, difficile à tester simplement sans un parsing complexe.
    # Le fait que send_message ait été appelé sans erreur est souvent suffisant.

# Test pour une erreur de connexion/login
def test_send_email_login_failure(email_sender):
    """Test que la fonction gère correctement une erreur d'authentification SMTP."""
    to_emails = ["recipient@test.com"]
    subject = "Test Subject"
    html_body = "<p>Test</p>"

    # Simuler une exception lors du login
    mock_server = Mock()
    mock_server.login.side_effect = smtplib.SMTPAuthenticationError(535, b'Authentication failed')

    mock_smtp = MagicMock()
    mock_smtp.__enter__ = Mock(return_value=mock_server)
    mock_smtp.__exit__ = Mock(return_value=None)

    # --- CORRECTION : Même chemin de patch ---
    with patch('trading_system.notifications.email_sender.smtplib.SMTP_SSL', return_value=mock_smtp):
        result = email_sender.send_email(to_emails, subject, html_body)

    # La fonction doit retourner False et ne pas propager l'exception
    assert result is False
    mock_server.login.assert_called_once()

# Test pour une erreur générale du serveur SMTP
def test_send_email_connection_error(email_sender):
    """Test que la fonction gère correctement une erreur de connexion au serveur."""
    to_emails = ["recipient@test.com"]
    subject = "Test Subject"
    html_body = "<p>Test</p>"

    # Simuler une exception lors de la création de la connexion
    with patch('trading_system.notifications.email_sender.smtplib.SMTP_SSL', side_effect=smtplib.SMTPConnectError(421, b'Service not available')):
        result = email_sender.send_email(to_emails, subject, html_body)

    # La fonction doit retourner False
    assert result is False
    # SMTP_SSL a été appelé, mais login et send_message n'ont pas pu être appelés

# Test pour vérifier la construction de l'email avec une seule adresse
def test_send_email_single_recipient(email_sender):
    """Test le format de l'en-tête 'To' avec un seul destinataire."""
    to_emails = ["sole_recipient@test.com"] # Liste d'un seul élément
    subject = "Single Recipient"
    html_body = "<p>Test</p>"

    mock_server = Mock()
    mock_smtp = MagicMock()
    mock_smtp.__enter__ = Mock(return_value=mock_server)
    mock_smtp.__exit__ = Mock(return_value=None)
    with patch('trading_system.notifications.email_sender.smtplib.SMTP_SSL', return_value=mock_smtp):
        email_sender.send_email(to_emails, subject, html_body)

    # Récupère le message envoyé
    sent_message = mock_server.send_message.call_args[0][0]
    # Vérifie que l'en-tête 'To' est une string simple, pas une liste
    assert sent_message['To'] == "sole_recipient@test.com"