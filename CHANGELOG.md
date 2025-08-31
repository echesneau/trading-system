# Changelog

Tous les changements notables de ce projet seront documentés dans ce fichier.

Le format est basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/),
et ce projet adhère à [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## A venir
### Documentation
- Documentation des API internes
- Exemples d'utilisation pour chaque module

## [0.2.0] - 2025-08-31
### Ajouté
- Systèmes de génération de rapports
  - Agrégation des signaux de trading pour multiples tickers en un seul rapport 
  - génération d'un rapport HTML
- Système de Notifications par Email 
  - Module notifications avec classe EmailSender
  - Envoi d'emails HTML
  - Support des listes de destinataires (multiple recipients)
  - Gestion robuste des erreurs SMTP avec reprise gracieuse
- Tests
  - tests unitaires complet pour vérifier le comportement attendu
  - test integration pour valider le flux de bout en bout avec des données réelles

### Modifié
- remonté des paramètres de configuration pour les stratégies de trading
## [0.1.0] - 2025-08-21

### Ajouté
- **Architecture complète** du système de trading hybride
- **Moteur de backtesting** avec métriques de performance (Sharpe ratio, drawdown)
- **Stratégie classique** basée sur RSI, MACD et Bollinger Bands
- **Stratégie hybride** combinant machine learning et indicateurs techniques
- **Intégration Yahoo Finance** avec gestion robuste des données
- **Pipeline de features techniques** (20+ indicateurs calculés)
- **Entraînement XGBoost** avec validation temporelle
- **Suite de tests complète** (unitaires et d'intégration)
- **Configuration modernisée** avec pyproject.toml et setuptools-scm
- **CI/CD GitHub Actions** avec tests multi-versions Python
- **Gestion de version sémantique** automatique basée sur les tags Git

### Modifié
- Refactorisation de la structure du projet en package Python conventionnel
- Amélioration de la gestion d'erreurs et de la robustesse
- Optimisation des calculs d'indicateurs techniques


# Template de Changelog
## [X.Y.Z] - YYYY-MM-DD

### Ajouté
- blalbla

## Types de Changements
- `Ajouté` pour les nouvelles fonctionnalités
- `Modifié` pour les changements de fonctionnalités existantes
- `Déprécié` pour les fonctionnalités soon-to-be removed
- `Supprimé` pour les fonctionnalités supprimées
- `Corrigé` pour les corrections de bugs
- `Sécurité` en cas de vulnérabilités