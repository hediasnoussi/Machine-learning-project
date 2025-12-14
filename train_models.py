"""
Script pour entraîner et sauvegarder tous les modèles de machine learning
"""
# Imports
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def convert_numeric_like(df, cols, inplace=False):
    """Convertit les colonnes en format numérique"""
    out = df if inplace else df.copy()
    for col in cols:
        s = out[col].astype(str)
        s = (s.str.replace(r'[\u00A0\u202F]', ' ', regex=True)
               .str.replace('$', '', regex=False)
               .str.replace(',', '', regex=False)
               .str.replace('%', '', regex=False)
               .str.replace('–', '-', regex=False)
               .str.replace(r'[^0-9eE\+\-\. KkMmBb]', '', regex=True)
               .str.strip())
        extracted = s.str.extract(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(?:\s*([KkMmBb]))?')
        vals = pd.to_numeric(extracted[0], errors='coerce')
        mult = extracted[1].str.upper().map({'K':1e3, 'M':1e6, 'B':1e9}).fillna(1)
        out[col] = vals * mult
    return out

def generate_risk_column(df):
    """Génère la colonne risk basée sur la volatilité future"""
    df_risk = df.copy()
    df_risk = df_risk.sort_values(['symbol', 'timestamp'])
    df_risk = df_risk.set_index(['symbol', 'timestamp'])
    
    df_risk['future_vol_7d'] = (
        df_risk.groupby(level='symbol')['price_usd']
              .pct_change(7)
              .shift(-7)
              .abs()
    )
    
    def classify_future_risk(v):
        if v < 0.05:
            return 'Stable'
        elif v < 0.20:
            return 'Volatile'
        else:
            return 'Speculative'
    
    df_risk['risk'] = df_risk['future_vol_7d'].apply(classify_future_risk)
    df_risk_reset = df_risk.reset_index()
    df = df.merge(
        df_risk_reset[['symbol', 'timestamp', 'risk']],
        on=['symbol', 'timestamp'],
        how='left'
    )
    return df.dropna(subset=['risk'])

# ============================================================================
# PRÉPARATION DES DONNÉES
# ============================================================================

print("Chargement des données...")
df = pd.read_csv('cryptocurrency.csv', low_memory=False)

# Nettoyage
num_cols = ['price_usd', 'vol_24h', 'chg_24h', 'chg_7d', 'market_cap']
df = convert_numeric_like(df, num_cols, inplace=False)
df = df.dropna(subset=num_cols)

# Génération de la colonne risk
print("Génération de la colonne risk...")
df = generate_risk_column(df)

# Transformation log
log_cols = ['price_usd', 'vol_24h', 'market_cap']
for col in log_cols:
    df[col] = np.log1p(df[col])

# Standardisation
print("Standardisation des données...")
scaler = StandardScaler()
cols_to_scale = ['price_usd', 'vol_24h', 'market_cap', 'chg_24h', 'chg_7d']
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# Sauvegarder le scaler
joblib.dump(scaler, 'scaler.pkl')
print("Scaler sauvegardé dans scaler.pkl")

# Définition des features
features_clustering = ['price_usd', 'vol_24h', 'market_cap', 'chg_24h', 'chg_7d']
features_classification = ['price_usd', 'vol_24h', 'chg_24h', 'chg_7d', 'market_cap']
features_regression = ['vol_24h', 'market_cap', 'chg_24h', 'chg_7d']

# ============================================================================
# ENTRAÎNEMENT DES MODÈLES
# ============================================================================

# 1. KMeans - Clustering
print("\n=== Entraînement KMeans ===")
X_clustering = df[features_clustering]
kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, max_iter=300, random_state=42)
kmeans.fit(X_clustering)
joblib.dump(kmeans, 'kmeans_model.pkl')
print("KMeans sauvegardé dans kmeans_model.pkl")

# 2. Logistic Regression - Classification Binaire
print("\n=== Entraînement Logistic Regression Classification Binaire ===")
df['target'] = (df['chg_24h'].shift(-1) > 0).astype(int)
X_binary = df[features_classification].iloc[:-1]
y_binary = df['target'].iloc[:-1]

X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_binary, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

logreg = LogisticRegression(max_iter=2000, random_state=42)
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs']
}

grid = GridSearchCV(
    estimator=logreg,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train_bin, y_train_bin)
logreg_binary = grid.best_estimator_
print(f"Meilleurs paramètres: {grid.best_params_}")
print(f"Meilleure accuracy CV: {grid.best_score_:.4f}")

joblib.dump(logreg_binary, 'logistic_regression_model.pkl')
print("Logistic Regression sauvegardé dans logistic_regression_model.pkl")

# 3. Random Forest - Classification Multiclasse
print("\n=== Entraînement Random Forest Classification Multiclasse ===")
X_multi = df[features_classification]
y_multi = df['risk']

X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42, stratify=y_multi
)

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
param_grid = {
    'n_estimators': [300, 500, 700],
    'max_depth': [20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid = GridSearchCV(
    rf,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train_multi, y_train_multi)
rf_multi = grid.best_estimator_
print(f"Meilleurs paramètres: {grid.best_params_}")
print(f"Meilleure accuracy CV: {grid.best_score_:.4f}")

joblib.dump(rf_multi, 'random_forest_model.pkl')
print("Random Forest sauvegardé dans random_forest_model.pkl")

# 4. XGBoost Regressor - Prédiction de Prix
print("\n=== Entraînement XGBoost Regressor ===")
X_reg = df[features_regression]
y_reg = df['price_usd']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

xgb_regressor = XGBRegressor(
    n_estimators=400,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=1.0,
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)
xgb_regressor.fit(X_train_reg, y_train_reg)
joblib.dump(xgb_regressor, 'xgb_regressor_model.pkl')
print("XGBoost Regressor sauvegardé dans xgb_regressor_model.pkl")

# Sauvegarder les informations sur les features
feature_info = {
    'clustering': features_clustering,
    'classification': features_classification,
    'regression': features_regression
}
with open('feature_info.pkl', 'wb') as f:
    pickle.dump(feature_info, f)

print("\n=== Tous les modèles ont été entraînés et sauvegardés avec succès! ===")
