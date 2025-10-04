"""
Generate scaler for user profile XGBoost model
This script creates and saves a StandardScaler for the features used in user_profile_xgb.py
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import os

# Set random state for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def load_and_preprocess_data(csv_path):
    """
    Load and preprocess data using the same feature engineering as user_profile_xgb.py
    """
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    # Convert data types
    df['timestamp_dt'] = pd.to_datetime(df['timestamp_dt'], errors='coerce')
    df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce')
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)
    df['is_fraud'] = df['is_fraud'].astype(int)
    
    print("Engineering features...")
    feat_df = df.copy()
    
    # Time-based features
    feat_df['hour'] = feat_df['timestamp_dt'].dt.hour.fillna(0).astype(int)
    feat_df['dayofweek'] = feat_df['timestamp_dt'].dt.dayofweek.fillna(0).astype(int)
    feat_df['is_weekend'] = (feat_df['dayofweek'] >= 5).astype(int)
    feat_df['hour_sin'] = np.sin(2 * np.pi * feat_df['hour'] / 24)
    feat_df['hour_cos'] = np.cos(2 * np.pi * feat_df['hour'] / 24)
    feat_df['time_since_signup_days'] = (feat_df['timestamp_dt'] - feat_df['signup_date']).dt.total_seconds().div(3600*24).fillna(0.0)
    
    # Amount-based features
    feat_df['amount_log1p'] = np.log1p(feat_df['amount'])
    feat_df['amount_ratio_to_user_avg'] = feat_df['amount'] / (feat_df['avg_amount_by_user'] + 1e-9)
    feat_df['user_amount_std'] = feat_df.groupby('user_id')['amount'].transform('std').fillna(0.0)
    feat_df['amount_zscore_user'] = (feat_df['amount'] - feat_df['avg_amount_by_user']) / (feat_df['user_amount_std'] + 1e-9)
    feat_df['seq_ratio'] = feat_df['seq_for_user'] / (feat_df['total_txn_by_user'] + 1e-9)
    feat_df['txns_last_1hr_by_user'] = feat_df['txns_last_1hr_by_user'].fillna(0).astype(int)
    
    # Categorical frequency encoding
    cat_cols = ['merchant_category', 'merchant_name', 'city', 'device', 'country']
    for c in cat_cols:
        col_counts = feat_df[c].value_counts()
        feat_df[c + '_freq'] = feat_df[c].map(col_counts).fillna(0).astype(int)
    
    # Lag features
    feat_df = feat_df.sort_values(['user_id','timestamp_dt']).reset_index(drop=True)
    for lag in [1,2,3]:
        feat_df[f'prev_amount_{lag}'] = feat_df.groupby('user_id')['amount'].shift(lag).fillna(0.0)
    feat_df['prev_amounts_mean_3'] = feat_df[[f'prev_amount_{l}' for l in [1,2,3]]].mean(axis=1)
    
    # Handle infinite and NaN values
    feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat_df.fillna(0, inplace=True)
    
    return feat_df

def generate_scaler(csv_path, output_dir):
    """
    Generate and save a StandardScaler for the user profile model features
    """
    # Load and preprocess data
    feat_df = load_and_preprocess_data(csv_path)
    
    # Define features (same as in user_profile_xgb.py)
    FEATURES = [
        'amount', 'amount_log1p', 'amount_ratio_to_user_avg', 'amount_zscore_user',
        'hour', 'hour_sin', 'hour_cos', 'dayofweek', 'is_weekend',
        'time_since_signup_days', 'seq_ratio',
        'total_txn_by_user', 'avg_amount_by_user', 'txns_last_1hr_by_user',
        'merchant_category_freq', 'merchant_name_freq', 'city_freq', 'device_freq', 'country_freq',
        'prev_amount_1','prev_amount_2','prev_amount_3','prev_amounts_mean_3'
    ]
    
    # Check if all features exist
    missing_feats = [f for f in FEATURES if f not in feat_df.columns]
    if missing_feats:
        raise ValueError("Missing engineered features: " + ", ".join(missing_feats))
    
    print(f"Using {len(FEATURES)} features for scaling")
    print("Features:", FEATURES)
    
    # Extract features for scaling
    X = feat_df[FEATURES]
    
    # Create and fit the scaler
    print("Fitting StandardScaler...")
    scaler = StandardScaler()
    scaler.fit(X)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the scaler
    scaler_path = os.path.join(output_dir, 'user_profile_scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")
    
    # Save feature statistics for reference
    feature_stats = {
        'features': FEATURES,
        'feature_means': scaler.mean_.tolist(),
        'feature_stds': scaler.scale_.tolist(),
        'feature_vars': scaler.var_.tolist(),
        'n_features_in': scaler.n_features_in_,
        'generated_at': datetime.now().isoformat(),
        'data_shape': X.shape,
        'scaler_type': 'StandardScaler'
    }
    
    stats_path = os.path.join(output_dir, 'user_profile_scaler_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(feature_stats, f, indent=2)
    print(f"Feature statistics saved to: {stats_path}")
    
    # Print some statistics
    print("\nFeature scaling statistics:")
    print("=" * 50)
    for i, feature in enumerate(FEATURES):
        print(f"{feature:25s} | Mean: {scaler.mean_[i]:8.4f} | Std: {scaler.scale_[i]:8.4f}")
    
    return scaler, scaler_path, stats_path

def test_scaler(scaler_path, csv_path, sample_size=1000):
    """
    Test the generated scaler with a sample of data
    """
    print(f"\nTesting scaler with {sample_size} samples...")
    
    # Load the scaler
    scaler = joblib.load(scaler_path)
    
    # Load and preprocess a sample of data
    feat_df = load_and_preprocess_data(csv_path)
    sample_df = feat_df.sample(n=min(sample_size, len(feat_df)), random_state=RANDOM_STATE)
    
    # Define features
    FEATURES = [
        'amount', 'amount_log1p', 'amount_ratio_to_user_avg', 'amount_zscore_user',
        'hour', 'hour_sin', 'hour_cos', 'dayofweek', 'is_weekend',
        'time_since_signup_days', 'seq_ratio',
        'total_txn_by_user', 'avg_amount_by_user', 'txns_last_1hr_by_user',
        'merchant_category_freq', 'merchant_name_freq', 'city_freq', 'device_freq', 'country_freq',
        'prev_amount_1','prev_amount_2','prev_amount_3','prev_amounts_mean_3'
    ]
    
    X_sample = sample_df[FEATURES]
    
    # Transform the data
    X_scaled = scaler.transform(X_sample)
    
    print("Scaled data statistics:")
    print("=" * 30)
    print(f"Shape: {X_scaled.shape}")
    print(f"Mean (should be ~0): {X_scaled.mean(axis=0).mean():.6f}")
    print(f"Std (should be ~1): {X_scaled.std(axis=0).mean():.6f}")
    print(f"Min value: {X_scaled.min():.4f}")
    print(f"Max value: {X_scaled.max():.4f}")
    
    return X_scaled

if __name__ == "__main__":
    # Configuration
    csv_path = "../../synthetic_transactions.csv"  # Adjust path as needed
    output_dir = "user-profile models"
    
    try:
        # Generate the scaler
        scaler, scaler_path, stats_path = generate_scaler(csv_path, output_dir)
        
        # Test the scaler
        test_scaler(scaler_path, csv_path)
        
        print("\n" + "="*60)
        print("Scaler generation completed successfully!")
        print(f"Scaler file: {scaler_path}")
        print(f"Stats file: {stats_path}")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"Error: Could not find the data file at {csv_path}")
        print("Please update the csv_path variable to point to your synthetic_transactions.csv file")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
