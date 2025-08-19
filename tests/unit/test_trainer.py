# tests/unit/test_trainer.py
from src.ml.trainer import ModelTrainer


def test_trainer_initialization():
    trainer = ModelTrainer({'n_estimators': 50})
    assert trainer.config['n_estimators'] == 50
    assert trainer.config['target_horizon'] == 5


def test_full_training(sample_data):
    # Réduire les fenêtres maximales pour les tests
    config = {
        'max_depth': 2,
        'ema_windows': [10, 20],  # Au lieu de [20, 50, 200]
        'bb_window': 10,  # Au lieu de 20
        'rsi_window': 10  # Au lieu de 14
    }
    trainer = ModelTrainer(config)
    results = trainer.train(sample_data)

    assert 'model' in results
    assert 'scaler' in results
    assert isinstance(results['feature_names'], list)
    assert len(results['feature_names']) > 0
    assert 0 <= results['test_score'] <= 1