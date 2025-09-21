'''Run on Google Collab for easier Computing'''

from google.colab import drive
drive.mount('/content/drive')


import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tqdm import tqdm
import pandas as pd, numpy as np, os, json, joblib, math, random
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


RANDOM_STATE=42

df=pd.read_csv('/content/drive/MyDrive/synthetic_transactions.csv')
df['timestamp_dt'] = pd.to_datetime(df['timestamp_dt'], errors='coerce')
df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce')
df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)
df['is_fraud'] = df['is_fraud'].astype(int)

print("Engineering features...")
feat_df = df.copy()
feat_df['hour'] = feat_df['timestamp_dt'].dt.hour.fillna(0).astype(int)
feat_df['dayofweek'] = feat_df['timestamp_dt'].dt.dayofweek.fillna(0).astype(int)
feat_df['is_weekend'] = (feat_df['dayofweek'] >= 5).astype(int)
feat_df['hour_sin'] = np.sin(2 * np.pi * feat_df['hour'] / 24)
feat_df['hour_cos'] = np.cos(2 * np.pi * feat_df['hour'] / 24)
feat_df['time_since_signup_days'] = (feat_df['timestamp_dt'] - feat_df['signup_date']).dt.total_seconds().div(3600*24).fillna(0.0)

feat_df['amount_log1p'] = np.log1p(feat_df['amount'])
feat_df['amount_ratio_to_user_avg'] = feat_df['amount'] / (feat_df['avg_amount_by_user'] + 1e-9)
feat_df['user_amount_std'] = feat_df.groupby('user_id')['amount'].transform('std').fillna(0.0)
feat_df['amount_zscore_user'] = (feat_df['amount'] - feat_df['avg_amount_by_user']) / (feat_df['user_amount_std'] + 1e-9)
feat_df['seq_ratio'] = feat_df['seq_for_user'] / (feat_df['total_txn_by_user'] + 1e-9)
feat_df['txns_last_1hr_by_user'] = feat_df['txns_last_1hr_by_user'].fillna(0).astype(int)

cat_cols = ['merchant_category', 'merchant_name', 'city', 'device', 'country']
for c in tqdm(cat_cols, desc="Encoding categorical freq"):
    col_counts = feat_df[c].value_counts()
    feat_df[c + '_freq'] = feat_df[c].map(col_counts).fillna(0).astype(int)

feat_df = feat_df.sort_values(['user_id','timestamp_dt']).reset_index(drop=True)
for lag in [1,2,3]:
    feat_df[f'prev_amount_{lag}'] = feat_df.groupby('user_id')['amount'].shift(lag).fillna(0.0)
feat_df['prev_amounts_mean_3'] = feat_df[[f'prev_amount_{l}' for l in [1,2,3]]].mean(axis=1)

feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
feat_df.fillna(0, inplace=True)

FEATURES = [
    'amount', 'amount_log1p', 'amount_ratio_to_user_avg', 'amount_zscore_user',
    'hour', 'hour_sin', 'hour_cos', 'dayofweek', 'is_weekend',
    'time_since_signup_days', 'seq_ratio',
    'total_txn_by_user', 'avg_amount_by_user', 'txns_last_1hr_by_user',
    'merchant_category_freq', 'merchant_name_freq', 'city_freq', 'device_freq', 'country_freq',
    'prev_amount_1','prev_amount_2','prev_amount_3','prev_amounts_mean_3'
]

missing_feats = [f for f in FEATURES if f not in feat_df.columns]
if missing_feats:
    raise ValueError("Missing engineered features: " + ", ".join(missing_feats))

# Train/test split
print("Creating train/test stratified split...")
X = feat_df[FEATURES]
y = feat_df['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
print("Train size:", X_train.shape, "Test size:", X_test.shape)
print("Positive ratio train:", y_train.mean(), "test:", y_test.mean())

# Try XGBoost, fallback to RandomForest
use_xgb = True
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
except Exception as e:
    print("XGBoost import failed:", e)
    use_xgb = False

model = None
model_path = "/content/drive/MyDrive/"

if use_xgb:
    print("Training XGBoost classifier (with early stopping). Progress will print per boosting round...")
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    scale_pos_weight = (n_neg / (n_pos + 1e-9)) if n_pos > 0 else 1.0

    xgb_clf = XGBClassifier(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=4,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=30
    )
    xgb_clf.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=20,
    )
    model = xgb_clf
    model.save_model(model_path + "user_profile_xgb.json")
    print("Saved XGBoost model to", model_path + "user_profile_xgb.json")
else:
    print("Training RandomForest fallback...")
    rf = RandomForestClassifier(n_estimators=300, n_jobs=4, random_state=RANDOM_STATE, class_weight='balanced')
    rf.fit(X_train, y_train)
    model = rf
    joblib.dump(model, model_path + "user_profile_rf.joblib")
    print("Saved RandomForest model to", model_path + "user_profile_rf.joblib")

# Evaluation
print("Evaluating model...")
y_pred_proba = model.predict_proba(X_test)[:,1]
y_pred = (y_pred_proba >= 0.5).astype(int)

roc_auc = roc_auc_score(y_test, y_pred_proba)
ap = average_precision_score(y_test, y_pred_proba)

print(f"ROC AUC: {roc_auc:.4f}")
print(f"Average precision (PR AUC): {ap:.4f}")
print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)

# Feature importances
try:
    importances = model.feature_importances_
    feat_imp = sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True)[:20]
    feat_imp_df = pd.DataFrame(feat_imp, columns=['feature','importance'])
    print(feat_imp_df)
except Exception as e:
    print("Feature importance extraction failed:", e)

# Plots
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.grid(True)
plt.show()

precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.figure(figsize=(6,5))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid(True)
plt.show()

# Save metadata and feature list
meta = {
    "features": FEATURES,
    "model_path": model_path + ("_xgb.json" if use_xgb else "user_profile_rf.joblib"),
    "model_type": "xgboost" if use_xgb else "random_forest",
    "train_rows": int(X_train.shape[0]),
    "test_rows": int(X_test.shape[0])
}
with open(model_path + "_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Saved model meta to", model_path + "_meta.json")
print("Pipeline finished.")
print("-", model_path + ("user_profile_xgb.json" if use_xgb else "user_profile_rf.joblib"))
print("-", model_path + "user_profile_meta.json")