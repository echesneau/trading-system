# src/notifications/reporter.py
import pandas as pd
from typing import List, Dict, Any
import datetime as dt
import logging

logger = logging.getLogger(__name__)

from ..features.fundamental import get_previous_date
from trading_system.features import calculate_indicators
from trading_system.strategies.hybrid import HybridStrategy
from trading_system.ml.model import load_model

class SignalReporter:
    """
    Collecte et agrège les signaux de trading pour plusieurs tickers
    afin de générer un rapport consolidé.
    """

    def __init__(self, strategy, data_loader, debug=False, debug_date: str = None):
        """
        debug_date: str
            date of report in format YYYY-MM-DD
        """
        self.strategy_cls = strategy
        self.data_loader = data_loader
        self.debug = debug
        self.debug_date = debug_date

    def generate_daily_report(self, tickers: List[str],
                              ticker_params: Dict[str, Dict],
                              max_window_range: int =50, **kwargs) -> Dict[str, List[Dict]]:
        """
        Génère un rapport des signaux pour la dernière journée de données disponibles
        pour chaque ticker.

        Args:
            tickers: Liste des symboles (ex: ['AIR.PA', 'SAN.PA'])
            ticker_params: Dictionnaire {ticker: paramètres} pour chaque ticker
            max_window_range: Nombre maximum de jours requis pour les indicateurs

        Returns:
            Un dictionnaire regroupant les signaux d'achat et de vente.
            Format:
            {
                'buy_signals': [
                    {'ticker': 'AIR.PA', 'signal': 'BUY', 'price': 120.50, 'date': ...},
                    ...
                ],
                'sell_signals': [...]
            }
        """
        # Initialiser les listes pour agréger les résultats
        all_buy_signals = []
        all_sell_signals = []
        all_hold_signals = []
        errors = []
        if self.debug:
            date = dt.datetime.strptime(self.debug_date, '%Y-%m-%d').date()
        else:
            date = None
        start_date = get_previous_date(max_window_range, today=date)
        if self.debug:
            end_date =  (dt.datetime.strptime(self.debug_date, '%Y-%m-%d').date() + dt.timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            end_date = (dt.datetime.now().date()+dt.timedelta(days=1)).strftime('%Y-%m-%d')

        for ticker in tickers:
            print(f"🔍 Analyse de {ticker}...")
            try:
                # 1. Charger les données HISTORIQUES pour le ticker
                hist_data = self.data_loader(
                    ticker,
                    start_date=start_date,
                    end_date=end_date
                )
                if hist_data is None or hist_data.empty:
                    error_msg = f"Aucune donnée pour {ticker}"
                    logger.warning(error_msg)
                    errors.append({'ticker': ticker, 'error': error_msg})
                    continue

                # 2. Récupérer les paramètres SPÉCIFIQUES à ce ticker
                config = ticker_params.get(ticker, {})

                # 3. Générer le signal pour ce dernier jour
                if self.strategy_cls.__name__ == 'HybridStrategy':
                    if 'model_path' in kwargs:
                        model_artifacts = load_model(kwargs['model_path'])
                    elif "model_artifacts" in kwargs:
                        model_artifacts = kwargs['model_artifacts']
                    else:
                        raise ValueError("Pour la stratégie Hybride, 'model_path' ou 'model_artifacts' doit être fourni dans kwargs.")
                    self.strategy = self.strategy_cls(model_artifacts, **config)
                else:
                    self.strategy = self.strategy_cls(**config)

                processed_data = calculate_indicators(hist_data, **config)
                signal_series = self.strategy.generate_signals(processed_data)
                signal_series = signal_series.map({1: 'BUY', -1: 'SELL', 0: 'HOLD'})
                signal = signal_series.iloc[-1]  # Prendre le dernier signal
                price = processed_data['Close'].iloc[-1]
                date = processed_data.index[-1]

                # 4. Agréger le signal
                signal_data = {'ticker': ticker, 'signal': signal, 'price': price, 'date': date}
                if signal == 'BUY':
                    all_buy_signals.append(signal_data)
                elif signal == 'SELL':
                    all_sell_signals.append(signal_data)
                else:
                    all_hold_signals.append(signal_data)

            except Exception as e:
                # Loguer l'erreur et l'ajouter au rapport
                error_msg = f"Erreur lors du traitement de {ticker}: {str(e)}"
                logger.error(error_msg, exc_info=True)  # <-- Log l'exception complète
                errors.append({'ticker': ticker, 'error': error_msg})
                continue

        # 5. Retourner le rapport agrégé
        return {
            'buy_signals': all_buy_signals,
            'sell_signals': all_sell_signals,
            'hold_signals': all_hold_signals, # <-- Optionnel, pour debug
            'errors': errors, # <-- NOUVEAU: Inclut les erreurs dans le rapport
            'report_date': dt.datetime.now().date(),
            'total_tickers_analyzed': len(tickers)
        }

    def format_report_to_html(self, report: Dict) -> str:
        """Transforme le rapport en HTML pour l'email."""
        html_content = f"""
        <h2>📊 Rapport de Trading Automatique</h2>
        <p>Date du scan: <strong>{report['report_date']}</strong><br>
        Nombre d'actions analysées: <strong>{report['total_tickers_analyzed']}</strong></p>
        """

        if "buy_signals" in report and len(report['buy_signals']) > 0:
            html_content += "<h3>🎯 Signaux d'ACHAT</h3><ul>"
            for signal in report['buy_signals']:
                html_content += f"<li><strong>{signal['ticker']}</strong> - Prix: {signal['price']:.2f}€ (Date: {signal['date'].strftime('%Y-%m-%d')})</li>"
            html_content += "</ul>"
        else:
            html_content += "<p>❌ Aucun signal d'achat aujourd'hui.</p>"

        if "sell_signals" in report and len(report['sell_signals']) > 0:
            html_content += "<h3>📉 Signaux de VENTE</h3><ul>"
            for signal in report['sell_signals']:
                html_content += f"<li><strong>{signal['ticker']}</strong> - Prix: {signal['price']:.2f}€ (Date: {signal['date'].strftime('%Y-%m-%d')})</li>"
            html_content += "</ul>"
        else:
            html_content += "<p>❌ Aucun signal de vente aujourd'hui.</p>"

        if "hold_signals" in report and len(report['hold_signals']) > 0:
            html_content += "<h3>📉 Signaux de conservation</h3><ul>"
            for signal in report['hold_signals']:
                html_content += f"<li><strong>{signal['ticker']}</strong> - Prix: {signal['price']:.2f}€ (Date: {signal['date'].strftime('%Y-%m-%d')})</li>"
            html_content += "</ul>"
        else:
            html_content += "<p>❌ Aucun signal de conservation aujourd'hui.</p>"


        if "errors" in report:
            html_content += "<h3>⚠️ Erreurs Rencontrées</h3><ul>"
            for error in report['errors']:
                html_content += f"<li><strong>{error['ticker']}</strong>: {error['error']}</li>"
            html_content += "</ul>"

        return html_content