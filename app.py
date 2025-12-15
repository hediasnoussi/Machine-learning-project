"""
Flask application for the cryptocurrency prediction dashboard
"""
# Flask imports
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from functools import wraps

# Standard Python imports
import pickle
import base64
import io

# Scientific imports
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Flask configuration
app = Flask(__name__)
app.secret_key = 'crypto_ml_dashboard_secret_key_2024'

# Authentication credentials
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'admin'

# Global variables for models
scaler = None
kmeans_model = None
logistic_regression_model = None
random_forest_model = None
xgb_regressor_model = None
feature_info = None
crypto_df = None

# Readable labels for the interface
FRIENDLY_RISK_LABELS = {
    'Stable': 'Balanced profile',
    'Volatile': 'Dynamic profile',
    'Speculative': 'Profile to monitor'
}

CLUSTER_LABELS = {
    0: 'Normal behaviour',
    1: 'Behaviour to monitor'
}

RISK_COLORS = {
    FRIENDLY_RISK_LABELS['Stable']: '#3498db',
    FRIENDLY_RISK_LABELS['Volatile']: '#f39c12',
    FRIENDLY_RISK_LABELS['Speculative']: '#e74c3c'
}

CLUSTER_COLORS = {
    0: '#3498db',
    1: '#e74c3c'
}

TIME_HORIZON_LABELS = {
    '24h': 'Next 24 hours',
    '7d': 'Next 7 days'
}

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_models():
    """Load all trained models"""
    global scaler, kmeans_model, logistic_regression_model
    global random_forest_model, xgb_regressor_model, feature_info
    
    try:
        scaler = joblib.load('scaler.pkl')
        kmeans_model = joblib.load('kmeans_model.pkl')
        logistic_regression_model = joblib.load('logistic_regression_model.pkl')
        random_forest_model = joblib.load('random_forest_model.pkl')
        xgb_regressor_model = joblib.load('xgb_regressor_model.pkl')
        
        with open('feature_info.pkl', 'rb') as f:
            feature_info = pickle.load(f)
        
        print("All models loaded successfully!")
        print("- KMeans (Clustering)")
        print("- Logistic Regression (Binary Classification)")
        print("- Random Forest (Multiclass Classification)")
        print("- XGBoost Regressor (Regression)")
    except FileNotFoundError as e:
        missing_file = str(e).split("'")[1] if "'" in str(e) else "unknown file"
        print(f"\n❌ ERROR: Missing file - {missing_file}")
        print("\n⚠️  WARNING: Models must be retrained!")
        print("Run: py train_models.py")
    except Exception as e:
        import traceback
        print(f"\n❌ ERROR while loading models:")
        print(f"   {type(e).__name__}: {e}")
        traceback.print_exc()
        print("\n⚠️  Solution: Run 'py train_models.py' to retrain the models")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def convert_numeric_like(df, cols, inplace=False):
    """Convert columns to numeric format"""
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

def load_crypto_data():
    """Load cryptocurrency data from CSV"""
    try:
        df = pd.read_csv('cryptocurrency.csv', low_memory=False)
        num_cols = ['price_usd', 'vol_24h', 'chg_24h', 'chg_7d', 'market_cap']
        df = convert_numeric_like(df, num_cols, inplace=False)
        df = df.dropna(subset=num_cols)
        return df
    except Exception as e:
        print(f"Error while loading data: {e}")
        return pd.DataFrame()

def preprocess_input_values(price_usd, vol_24h, market_cap, chg_24h, chg_7d, is_raw=True):
    """Transform raw values into standardized values if needed"""
    if is_raw:
        data = pd.DataFrame({
            'price_usd': [price_usd],
            'vol_24h': [vol_24h],
            'market_cap': [market_cap],
            'chg_24h': [chg_24h],
            'chg_7d': [chg_7d]
        })
        
        # Log transformation for some columns
        log_cols = ['price_usd', 'vol_24h', 'market_cap']
        for col in log_cols:
            if data[col].iloc[0] > 0:
                data[col] = np.log1p(data[col])
            else:
                data[col] = 0
        
        # Standardization
        cols_to_scale = ['price_usd', 'vol_24h', 'market_cap', 'chg_24h', 'chg_7d']
        data_scaled = scaler.transform(data[cols_to_scale])
        
        return {
            'price_usd': float(data_scaled[0][0]),
            'vol_24h': float(data_scaled[0][1]),
            'market_cap': float(data_scaled[0][2]),
            'chg_24h': float(data_scaled[0][3]),
            'chg_7d': float(data_scaled[0][4])
        }
    else:
        return {
            'price_usd': price_usd,
            'vol_24h': vol_24h,
            'market_cap': market_cap,
            'chg_24h': chg_24h,
            'chg_7d': chg_7d
        }

def create_plot_base64(fig):
    """Convert a matplotlib figure to base64 for HTML display"""
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)
    return plot_url

# ============================================================================
# AUTHENTICATION
# ============================================================================

def login_required(f):
    """Decorator to protect routes requiring authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session or not session['logged_in']:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ============================================================================
# AUTHENTICATION ROUTES
# ============================================================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Incorrect credentials')
    
    if 'logged_in' in session and session['logged_in']:
        return redirect(url_for('index'))
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """Logout"""
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
def root():
    """Redirect to login if not connected, otherwise to dashboard"""
    if 'logged_in' in session and session['logged_in']:
        return redirect(url_for('index'))
    return redirect(url_for('login'))

# ============================================================================
# MAIN ROUTES
# ============================================================================

@app.route('/dashboard')
@login_required
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/cryptos', methods=['GET'])
@login_required
def get_cryptos():
    """Return the list of available cryptocurrencies"""
    try:
        if crypto_df.empty:
            return jsonify({'success': False, 'error': 'Data not available'}), 500
        
        cryptos = crypto_df[['name', 'symbol']].drop_duplicates().sort_values('name')
        cryptos_list = [
            {'name': row['name'], 'symbol': row['symbol']} 
            for _, row in cryptos.iterrows()
        ]
        
        return jsonify({
            'success': True,
            'cryptos': cryptos_list
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Route for predictions"""
    if not all([scaler, kmeans_model, logistic_regression_model, random_forest_model, xgb_regressor_model]):
        return jsonify({
            'success': False,
            'error': 'Models are not loaded. Please run train_models.py first.'
        }), 500
    
    try:
        data = request.json
        crypto_name = data.get('crypto_name', None)
        crypto_symbol = data.get('crypto_symbol', None)
        time_horizon = data.get('time_horizon', '24h')
        readable_horizon = TIME_HORIZON_LABELS.get(time_horizon, 'Next 24 hours')
        
        # Retrieve crypto data or use direct values
        if crypto_name or crypto_symbol:
            if crypto_df.empty:
                return jsonify({
                    'success': False,
                    'error': 'Cryptocurrency data not available'
                }), 400
            
            if crypto_name:
                crypto_data = crypto_df[crypto_df['name'].str.contains(crypto_name, case=False, na=False)]
            else:
                crypto_data = crypto_df[crypto_df['symbol'].str.contains(crypto_symbol, case=False, na=False)]
            
            if crypto_data.empty:
                return jsonify({
                    'success': False,
                    'error': f'Cryptocurrency "{crypto_name or crypto_symbol}" not found'
                }), 400
            
            crypto_row = crypto_data.iloc[-1]
            price_usd = float(crypto_row['price_usd'])
            vol_24h = float(crypto_row['vol_24h'])
            market_cap = float(crypto_row['market_cap'])
            chg_24h = float(crypto_row['chg_24h'])
            chg_7d = float(crypto_row['chg_7d'])
            processed = preprocess_input_values(price_usd, vol_24h, market_cap, chg_24h, chg_7d, is_raw=True)
        else:
            is_raw = data.get('is_raw', True)
            price_usd = float(data.get('price_usd', 0))
            vol_24h = float(data.get('vol_24h', 0))
            market_cap = float(data.get('market_cap', 0))
            chg_24h = float(data.get('chg_24h', 0))
            chg_7d = float(data.get('chg_7d', 0))
            processed = preprocess_input_values(price_usd, vol_24h, market_cap, chg_24h, chg_7d, is_raw)
        
        # Prepare features
        features_clustering = np.array([[processed['price_usd'], processed['vol_24h'], 
                                          processed['market_cap'], processed['chg_24h'], processed['chg_7d']]])
        features_classification = np.array([[processed['price_usd'], processed['vol_24h'], 
                                            processed['chg_24h'], processed['chg_7d'], processed['market_cap']]])
        features_regression = np.array([[processed['vol_24h'], processed['market_cap'], 
                                       processed['chg_24h'], processed['chg_7d']]])
        
        # Predictions
        cluster = int(kmeans_model.predict(features_clustering)[0])
        
        binary_pred = logistic_regression_model.predict(features_classification)[0]
        binary_proba = logistic_regression_model.predict_proba(features_classification)[0]
        direction = "Up" if binary_pred == 1 else "Down"
        proba_up = float(binary_proba[1]) * 100
        proba_down = float(binary_proba[0]) * 100
        
        risk_pred = random_forest_model.predict(features_classification)[0]
        risk_proba = random_forest_model.predict_proba(features_classification)[0]
        risk_classes = random_forest_model.classes_
        risk_proba_dict = {risk_classes[i]: float(risk_proba[i]) * 100 
                          for i in range(len(risk_classes))}
        readable_risk = FRIENDLY_RISK_LABELS.get(risk_pred, str(risk_pred))
        readable_risk_proba = {
            FRIENDLY_RISK_LABELS.get(key, key): value
            for key, value in risk_proba_dict.items()
        }
        
        price_pred = float(xgb_regressor_model.predict(features_regression)[0])
        readable_cluster = CLUSTER_LABELS.get(cluster, f'Profile {cluster}')
        
        # Generate plots
        plots = generate_plots(
            binary_proba,
            readable_risk_proba,
            readable_risk,
            price_pred,
            cluster,
            readable_cluster,
            direction,
            readable_horizon
        )
        
        return jsonify({
            'success': True,
            'predictions': {
                'cluster': readable_cluster,
                'direction': direction,
                'proba_hausse': round(proba_up, 2),
                'proba_baisse': round(proba_down, 2),
                'risk': readable_risk,
                'risk_proba': readable_risk_proba,
                'price_predicted': round(price_pred, 4),
                'time_horizon': readable_horizon
            },
            'plots': plots
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

# ============================================================================
# PLOT GENERATION
# ============================================================================

def generate_plots(binary_proba, risk_proba_dict, risk_label, price_pred, cluster_id, cluster_label, direction, horizon_label):
    """Generate all plots for the dashboard"""
    plots = {}
    
    # Direction probability plot
    fig, ax = plt.subplots(figsize=(8, 6))
    categories = ['Down', 'Up']
    probas = [binary_proba[0] * 100, binary_proba[1] * 100]
    colors = ['#e74c3c', '#2ecc71']
    bars = ax.bar(categories, probas, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Probability (%)', fontsize=12)
    ax.set_title(f'Price direction ({horizon_label}): {direction}', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    for i, (bar, prob) in enumerate(zip(bars, probas)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{prob:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plots['direction'] = create_plot_base64(fig)
    
    # Risk probability plot
    fig, ax = plt.subplots(figsize=(8, 6))
    risk_categories = list(risk_proba_dict.keys())
    risk_probas = list(risk_proba_dict.values())
    bar_colors = [RISK_COLORS.get(cat, '#95a5a6') for cat in risk_categories]
    bars = ax.bar(risk_categories, risk_probas, color=bar_colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Probability (%)', fontsize=12)
    ax.set_title(f'Attention profiles: {risk_label}', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    for bar, prob in zip(bars, risk_probas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{prob:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plots['risk'] = create_plot_base64(fig)
    
    # Risk pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    colors_pie = [RISK_COLORS.get(cat, '#95a5a6') for cat in risk_categories]
    wedges, texts, autotexts = ax.pie(risk_probas, labels=risk_categories, 
                                      autopct='%1.1f%%', colors=colors_pie,
                                      startangle=90, textprops={'fontsize': 12})
    ax.set_title('Attention profile distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plots['risk_pie'] = create_plot_base64(fig)
    
    # Price prediction plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(['Predicted price'], [price_pred], color='#9b59b6', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Price (standardized)', fontsize=12)
    ax.set_title('USD price prediction', fontsize=14, fontweight='bold')
    ax.text(price_pred, 0, f'{price_pred:.4f}', 
            ha='left', va='center', fontsize=12, fontweight='bold', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plots['price'] = create_plot_base64(fig)
    
    # Cluster plot
    fig, ax = plt.subplots(figsize=(8, 6))
    cluster_color = CLUSTER_COLORS.get(cluster_id, '#95a5a6')
    ax.bar(['Category'], [1], color=cluster_color, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Assigned profile', fontsize=12)
    ax.set_title(f'Assigned category: {cluster_label}', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.2)
    ax.text(0, 0.5, cluster_label, ha='center', va='center',
            fontsize=16, fontweight='bold', color='white')
    plt.tight_layout()
    plots['cluster'] = create_plot_base64(fig)
    
    return plots

# ============================================================================
# INITIALIZATION
# ============================================================================

if __name__ == '__main__':
    load_models()
    crypto_df = load_crypto_data()
    app.run(debug=True, host='0.0.0.0', port=5000)
