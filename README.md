# 🚀 Trading System - Système de Trading Algorithmique Hybride

[![CI](https://github.com/echesneau/trading-system/actions/workflows/ci.yml/badge.svg)](https://github.com/echesneau/trading-system/actions)
[![Coverage](https://codecov.io/gh/votre-username/trading-system/branch/main/graph/badge.svg)](https://codecov.io/gh/votre-username/trading-system)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Système de trading algorithmique avancé combinant **machine learning** et **analyse technique** pour la Bourse de Paris. Développé en Python avec une architecture modulaire et professionnelle.

## ✨ Fonctionnalités

### 🤖 Stratégies de Trading
- **🔍 Stratégie Hybride** : Combinaison de modèles ML (XGBoost) et indicateurs techniques
- **📊 Stratégie Classique** : RSI, MACD, Bollinger Bands, et analyse technique pure
- **🎯 Backtesting Avancé** : Métriques de performance complètes (Sharpe, Drawdown, etc.)

### 📈 Data Pipeline
- **📡 Intégration Yahoo Finance** : Données temps réel et historiques
- **⚡ Calcul d'Indicateurs** : 20+ indicateurs techniques (RSI, MACD, ATR, EMA, etc.)
- **🔄 Gestion Robustesse** : Cache, retry automatique, fallback synthétique

### 🧪 Qualité Industrielle
- **✅ Tests Complets** : Unitaires, d'intégration et coverage >50%
- **🔒 CI/CD Automatisée** : GitHub Actions avec linting et security scanning
- **📦 Packaging Modern** : pyproject.toml et versioning sémantique automatique

## 🚀 Installation Rapide

### Prérequis
- Python 3.9+
- pip

### Installation
```bash
# Cloner le dépôt
git clone https://github.com/votre-username/trading-system.git
cd trading-system

# Installation en mode développement
pip install -e .

# Installer les dépendances de développement (optionnel)
pip install -e .[dev,test]
```
## 💻 Utilisation
### Backtesting d'une Stratégie  
```python
from trading_system.backtesting import BacktestingEngine
from trading_system.strategies.hybrid import HybridStrategy
from trading_system.data.loader import load_yfinance_data

# Chargement des données
data = load_yfinance_data("AIR.PA", start_date="2024-01-01")

# Initialisation de la stratégie hybride
strategy = HybridStrategy(
    model=model,  # Modèle entraîné
    scaler=scaler,
    feature_names=['RSI', 'MACD', 'Close', 'BB_Upper', 'BB_Lower'],
    rsi_buy=30,
    rsi_sell=70
)

# Lancement du backtest
engine = BacktestingEngine(
    strategy=strategy,
    data=data,
    initial_capital=10000,
    transaction_fee=0.001
)

results = engine.run()
print(f"Rendement: {results['performance']['return']:.2%}")
```
### Entraînement d'un Modèle
```bash
# Entraînement d'un nouveau modèle
python scripts/train_model.py --ticker SAN.PA --start-date 2020-01-01

# Backtest avec le modèle entraîné
python scripts/run_backtest.py --strategy hybrid --ticker AIR.PA
```
## 🏗️ Architecture
```text
trading-system/
├── src/trading_system/
│   ├── backtesting/          # Moteur de backtesting
│   ├── strategies/           # Stratégies de trading
│   ├── data/                 # Data pipeline et connecteurs
│   ├── features/             # Feature engineering
│   └── ml/                   # Machine learning
├── tests/                    # Suite de tests complète
├── scripts/                  # Scripts d'utilisation
└── models/                   # Modèles entraînés (gitignore)
```
## 📊 Métriques de Performance
Le système calcule automatiquement :
- 📈 Rendement total et annualisé 
- 📉 Maximum Drawdown (MDD)
- ⚖️ Ratio de Sharpe 
- 📋 Statistiques de trades 
- 🎯 Performance vs benchmark

## 🧪 Tests et Qualité
```bash
# Lancer tous les tests
pytest tests/ -v

# Tests unitaires seulement
pytest tests/unit/ -v

# Tests d'intégration
pytest tests/integration/ -v

# Vérification de la qualité de code
flake8 src/
black --check src/ tests/
mypy src/
```
## 🤝 Contribution
Les contributions sont les bienvenues ! Voici le processus :
1. Fork le projet 
2. Créez une branche (git checkout -b feature/ma-fonctionnalite)
3. Commitez vos changements (git commit -am 'Ajout ma fonctionnalite')
4. Push sur la branche (git push origin feature/ma-fonctionnalite)
5. Ouvrez une Pull Request

### Standards de Code
- Black pour le formatting 
- Flake8 pour le linting 
- Types hints pour la maintenabilité 
- Tests obligatoires pour les nouvelles fonctionnalités

## 📝 Licence
Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.

## ⚠️ Disclaimer
ATTENTION : Ce logiciel est fourni à des fins éducatives et de recherche seulement.
Le trading comporte des risques de perte en capital. Les auteurs ne sont pas responsables
des pertes financières résultant de l'utilisation de ce logiciel.

## 📞 Support
- 🐛 Bug Reports : Issues GitHub 
- 💡 Feature Requests : Discussions GitHub