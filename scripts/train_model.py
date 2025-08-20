# scripts/train_model.py
from src.trading_system.ml.trainer import ModelTrainer
from src.trading_system.data import load_yfinance_data
import joblib


def main():
    # 1. Charger les données
    data = load_yfinance_data("SAN.PA", start_date="2020-01-01")

    # 2. Entraîner le modèle
    trainer = ModelTrainer({
        'n_estimators': 200,
        'target_horizon': 3  # Prévoir 3 jours à l'avance
    })
    artifacts = trainer.train(data)

    # 3. Sauvegarder
    joblib.dump(artifacts, "models/xgb_model_v1.joblib")
    print(f"Modèle sauvegardé. Score moyen: {artifacts['test_score']:.2%}")


if __name__ == "__main__":
    main()