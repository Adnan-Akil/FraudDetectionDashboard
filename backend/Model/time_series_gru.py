'''Run in google colab after mounting gdrive'''

import os, random, math, json
import numpy as np, pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
import joblib

DATA_PATH = "/content/drive/MyDrive/synthetic_transactions.csv"   
SEQ_LEN = 10
RANDOM_STATE = 42
DEBUG = False          
BACKEND = "auto"       
EPOCHS = 30 if not DEBUG else 3
BATCH_SIZE = 128 if not DEBUG else 64
USE_GPU = True         
SAVE_DIR = "/content/drive/MyDrive/ts_model"
os.makedirs(SAVE_DIR, exist_ok=True)

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

df = pd.read_csv(DATA_PATH)
if 'timestamp_dt' not in df.columns:
  df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce')

# --- Feature engineering (time-series features) ---
df['timestamp_dt'] = pd.to_datetime(df['timestamp_dt'], errors='coerce')
df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce')
df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)
df['is_fraud'] = df['is_fraud'].astype(int)
df['hour'] = df['timestamp_dt'].dt.hour.fillna(0).astype(int)
df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
df['time_since_prev_txn_seconds'] = df.groupby('user_id')['timestamp_dt'].diff().dt.total_seconds().fillna(0.0)
df['amount_log1p'] = np.log1p(df['amount'])
df['user_amount_std'] = df.groupby('user_id')['amount'].transform('std').fillna(0.0)
df['amount_ratio_to_user_avg'] = df['amount'] / (df['avg_amount_by_user'] + 1e-9)
df['amount_zscore_user'] = (df['amount'] - df['avg_amount_by_user']) / (df['user_amount_std'] + 1e-9)
df['seq_ratio'] = df['seq_for_user'] / (df['total_txn_by_user'] + 1e-9)
df['txns_last_1hr_by_user'] = df['txns_last_1hr_by_user'].fillna(0).astype(int)

cat_cols = ['merchant_category','merchant_name','city','device','country']
for c in cat_cols:
    df[c + '_freq'] = df[c].map(df[c].value_counts()).fillna(0).astype(int)

SEQ_FEATURES = [
 'amount_log1p','amount_ratio_to_user_avg','amount_zscore_user',
 'hour_sin','hour_cos','time_since_prev_txn_seconds',
 'txns_last_1hr_by_user','seq_ratio',
 'merchant_category_freq','merchant_name_freq','city_freq','device_freq','country_freq'
]

df = df.sort_values(['user_id','timestamp_dt']).reset_index(drop=True)

# --- build sliding windows (per-user) ---
X, y, meta = [], [], []
users = df['user_id'].unique().tolist()
for uid in tqdm(users, desc="building windows"):
    g = df[df['user_id']==uid].reset_index(drop=True)
    n = len(g)
    if n < SEQ_LEN:
        continue
    arr = g[SEQ_FEATURES].values
    labels = g['is_fraud'].values
    txn_ids = g['txn_id'].values
    for start in range(0, n - SEQ_LEN + 1):
        X.append(arr[start:start+SEQ_LEN])
        y.append(labels[start+SEQ_LEN-1])
        meta.append((uid, int(txn_ids[start+SEQ_LEN-1])))

X = np.array(X); y = np.array(y)
print("Windows:", X.shape, "Pos:", int(y.sum()), "Neg:", len(y)-int(y.sum()))

# --- split by user to avoid leakage ---
unique_users = np.array(users)
np.random.shuffle(unique_users)
train_users, test_users = train_test_split(unique_users, test_size=0.2, random_state=RANDOM_STATE)
train_mask = np.isin([m[0] for m in meta], train_users)
test_mask = np.isin([m[0] for m in meta], test_users)
X_train = X[train_mask]; y_train = y[train_mask]
X_test = X[test_mask]; y_test = y[test_mask]

# debug sample (fast)
if DEBUG:
    # keep first 2000 windows for train, 500 for test
    X_train = X_train[:2000]; y_train = y_train[:2000]
    X_test = X_test[:500]; y_test = y_test[:500]

print("Train windows:", X_train.shape, "Test windows:", X_test.shape)
print("Train positive ratio:", y_train.mean(), "Test positive ratio:", y_test.mean())

# scale features (fit on train windows)
n_features = X_train.shape[2]
scaler = StandardScaler()
X_train_2d = X_train.reshape(-1, n_features)
scaler.fit(X_train_2d)
X_train = scaler.transform(X_train_2d).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
joblib.dump(scaler, os.path.join(SAVE_DIR, "ts_scaler.joblib"))

# --- pick backend ---
backend = BACKEND
if BACKEND == "auto":
    try:
        import tensorflow as tf
        backend = "tf"
    except Exception:
        try:
            import torch
            backend = "torch"
        except Exception:
            backend = None
if backend is None:
    raise RuntimeError("No TF or Torch available. Install TF or Torch in Colab.")

# ---------- TENSORFLOW path (recommended on Colab with GPU) ----------
if backend == "tf":
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Masking, GRU, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

    model = Sequential([
        Masking(mask_value=0., input_shape=(SEQ_LEN, n_features)),
        GRU(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc')])
    print(model.summary())

    # class weights
    from sklearn.utils import class_weight
    cw = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {int(c): float(w) for c,w in zip(np.unique(y_train), cw)}
    print("class weights:", class_weight_dict)

    # callbacks & training
    model_path = os.path.join(SAVE_DIR, "_gru.h5")
    es = EarlyStopping(monitor='val_auc', mode='max', patience=6, restore_best_weights=True, verbose=1)
    mc = ModelCheckpoint(model_path, monitor='val_auc', mode='max', save_best_only=True, verbose=1)
    rl = ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=3, verbose=1)

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=EPOCHS, batch_size=BATCH_SIZE, class_weight=class_weight_dict,
                        callbacks=[es, mc, rl], verbose=2)

    # evaluate
    y_proba = model.predict(X_test, batch_size=256).ravel()
    y_pred = (y_proba >= 0.5).astype(int)
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print("PR AUC:", average_precision_score(y_test, y_proba))
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # save meta
    meta = {"backend":"tf","model_path":model_path,"scaler":os.path.join(SAVE_DIR,"ts_scaler.joblib"),
            "seq_len":SEQ_LEN,"features":SEQ_FEATURES}
    with open(os.path.join(SAVE_DIR,"ts_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

# ---------- PYTORCH path ----------
elif backend == "torch":
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() and USE_GPU else "cpu")
    print("Using device:", device)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    class GRUClassifier(nn.Module):
        def __init__(self, input_size, hidden_size=64):
            super().__init__()
            self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32,1),
                nn.Sigmoid()
            )
        def forward(self, x):
            out, _ = self.gru(x)
            out = out[:, -1, :]
            return self.fc(out).squeeze(-1)

    model = GRUClassifier(n_features).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_ap = 0.0
    model_path = os.path.join(SAVE_DIR, "ts_gru_torch.pth")
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch} train"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_train_loss = total_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        preds_all, y_all = [], []
        with torch.no_grad():
            for xb, yb in tqdm(test_loader, desc=f"Epoch {epoch} eval"):
                xb = xb.to(device)
                preds = model(xb).detach().cpu().numpy()
                preds_all.append(preds)
                y_all.append(yb.numpy())
        y_proba = np.concatenate(preds_all)
        y_true = np.concatenate(y_all)
        val_ap = average_precision_score(y_true, y_proba)
        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f} val_ap={val_ap:.4f}")
        if val_ap > best_ap:
            best_ap = val_ap
            torch.save(model.state_dict(), model_path)

    # final eval
    y_pred = (y_proba >= 0.5).astype(int)
    print("Best val AP:", best_ap)
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    # save meta
    meta = {"backend":"torch","model_path":model_path,"scaler":os.path.join(SAVE_DIR,"ts_scaler.joblib"),
            "seq_len":SEQ_LEN,"features":SEQ_FEATURES}
    with open(os.path.join(SAVE_DIR,"ts_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

print("Done. Models & scaler saved in", SAVE_DIR)
