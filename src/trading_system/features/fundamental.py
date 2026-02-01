from datetime import datetime
import pandas as pd

def get_previous_date(open_days, today=None):
    if today is None:
        today = datetime.now().date()
    # Utiliser bdate_range pour les jours ouvrés
    date_result = pd.bdate_range(end=today, periods=open_days+1)[0].date()

    return date_result.strftime('%Y-%m-%d')