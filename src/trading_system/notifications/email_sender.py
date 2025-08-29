# src/notifications/email_sender.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from typing import List

class EmailSender:
    """Client simple pour envoyer des emails via SMTP."""

    def __init__(self, smtp_server: str, smtp_port: int, login: str, password: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.login = login
        self.password = password

    def send_email(self, to_emails: List[str], subject: str, html_body: str) -> bool:
        """
        Envoie un email HTML.

        Args:
            to_emails: Liste des adresses destinataires.
            subject: Sujet de l'email.
            html_body: Corps du message en HTML.

        Returns:
            bool: True si l'envoi a réussi, False sinon.
        """
        # Création du message
        msg = MIMEMultipart()
        msg['From'] = self.login
        msg['To'] = ", ".join(to_emails)
        msg['Subject'] = subject

        # Attache le corps du message en HTML
        msg.attach(MIMEText(html_body, 'html'))

        try:
            # Connexion et envoi sécurisé
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
                server.login(self.login, self.password)
                server.send_message(msg)
            print("✅ Email envoyé avec succès !")
            return True
        except Exception as e:
            print(f"❌ Erreur lors de l'envoi de l'email: {e}")
            return False