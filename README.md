# Dashboard Crypto - Machine Learning

Application web Flask pour la prédiction de cryptomonnaies utilisant plusieurs modèles de machine learning.

## Modèles utilisés

1. **KMeans** - Clustering non supervisé pour segmenter les cryptomonnaies
2. **Logistic Regression** - Classification binaire pour prédire la direction du prix (Hausse/Baisse)
3. **Random Forest** - Classification multiclasse pour classer le niveau de risque (Stable, Volatile, Speculative)
4. **XGBoost Regressor** - Prédiction du prix USD

## Installation

1. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

### Étape 1 : Entraîner les modèles

Avant de lancer l'application, vous devez entraîner et sauvegarder les modèles :

```bash
python train_models.py
```

Ce script va :
- Charger et nettoyer les données depuis `cryptocurrency.csv`
- Entraîner tous les modèles
- Sauvegarder les modèles dans des fichiers `.pkl`

### Étape 2 : Lancer l'application Flask

```bash
python app.py
```

L'application sera accessible à l'adresse : `http://localhost:5000`

## Utilisation du Dashboard

1. **Connexion** : Connectez-vous avec les identifiants :
   - Username: `admin`
   - Password: `admin`

2. **Sélectionner une cryptomonnaie** : Choisissez une cryptomonnaie depuis le menu déroulant

3. **Obtenir les prédictions** : Cliquez sur "Obtenir les Prédictions"

4. **Visualiser les résultats** : Le dashboard affichera :
   - Le cluster assigné (KMeans)
   - La direction prédite (Hausse/Baisse) avec probabilités (Logistic Regression)
   - Le niveau de risque (Stable/Volatile/Speculative) (Random Forest)
   - Le prix prédit (XGBoost Regressor)
   - Plusieurs graphiques visuels

## Structure des fichiers

```
tp ml/
├── app.py                      # Application Flask principale
├── train_models.py             # Script d'entraînement des modèles
├── requirements.txt            # Dépendances Python
├── README.md                   # Documentation
├── .gitignore                  # Fichiers à ignorer par Git
├── templates/
│   ├── index.html             # Interface principale du dashboard
│   └── login.html             # Page de connexion
├── cryptocurrency.csv          # Données d'entraînement
├── scaler.pkl                  # Scaler sauvegardé (généré)
├── kmeans_model.pkl           # Modèle KMeans (généré)
├── logistic_regression_model.pkl # Modèle Logistic Regression (généré)
├── random_forest_model.pkl     # Modèle Random Forest (généré)
├── xgb_regressor_model.pkl    # Modèle XGBoost Regressor (généré)
└── feature_info.pkl            # Informations sur les features (généré)
```

## Notes importantes

- Les modèles sont entraînés sur les données du fichier `cryptocurrency.csv`
- Le scaler utilisé pour la standardisation est sauvegardé et chargé automatiquement
- Toutes les routes sont protégées par authentification

## Technologies utilisées

- Flask - Framework web
- scikit-learn - KMeans, Logistic Regression, Random Forest
- XGBoost - Modèle de régression
- Matplotlib/Seaborn - Visualisation
- HTML/CSS/JavaScript - Interface utilisateur
