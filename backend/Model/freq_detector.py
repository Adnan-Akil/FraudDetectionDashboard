'''Run in google colab with synthetic_transactions.csv in your Google Drive'''

from tqdm import tqdm
import numpy as np, pandas as pd, os, random, math, json, joblib, warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")
RND = 42
np.random.seed(RND)
random.seed(RND)

OUT_DIR = "/content/drive/MyDrive/freq_models_full"
os.makedirs(OUT_DIR, exist_ok=True)

DEBUG = False               
NUM_USERS = 120 if DEBUG else 600
TARGET_ROWS = 8000 if DEBUG else 30000
INTERVAL_HOURS = 6        # aggregate interval (6 hours)
SEQ_LEN = 12              # sequence length (# intervals) for GRU
GRU_EPOCHS = 6 if DEBUG else 30
RF_ESTIMATORS = 60 if DEBUG else 200
IF_ESTIMATORS = 200


DATA_PATH = "/content/drive/MyDrive/synthetic_transactions.csv"

df = pd.read_csv(DATA_PATH)
df['timestamp_dt'] = pd.to_datetime(df['timestamp_dt'], errors='coerce')
df = df.sort_values(['user_id','timestamp_dt']).reset_index(drop=True)

# ---------- Build interval panel (INTERVAL_HOURS) ----------
records = []
for uid, g in tqdm(df.groupby('user_id'), total=df['user_id'].nunique(), desc="Users"):
    times = g['timestamp_dt']
    if times.isna().all(): continue
    start = times.min().floor('H'); end = times.max().ceil('H')
    bins = pd.date_range(start=start, end=end + pd.Timedelta(hours=INTERVAL_HOURS), freq=f"{INTERVAL_HOURS}H")
    for i in range(len(bins)-1):
        b0, b1 = bins[i], bins[i+1]
        mask = (g['timestamp_dt'] >= b0) & (g['timestamp_dt'] < b1)
        cnt = int(mask.sum())
        fraud_present = int(((g['timestamp_dt'] >= b0) & (g['timestamp_dt'] < b1) & (g['is_fraud'] == 1)).any())
        records.append({
            "user_id": int(uid),
            "interval_start": b0,
            "interval_end": b1,
            "hour_of_day": int(b0.hour),
            "dayofweek": int(b0.dayofweek),
            "is_weekend": int(b0.dayofweek >= 5),
            "count": cnt,
            "fraud_present": fraud_present,
            "total_txns": int(g.shape[0]),
            "avg_amount_by_user": float(g['amount'].mean())
        })

panel = pd.DataFrame(records).sort_values(['user_id','interval_start']).reset_index(drop=True)

# ---------- Rolling features ----------
panel['prev_1']=0; panel['prev_3']=0; panel['prev_6']=0; panel['prev_avg']=0.0
for uid, g in tqdm(panel.groupby('user_id'), total=panel['user_id'].nunique(), desc="Rolling"):
    idx = g.index
    counts = g['count'].values
    prev1 = np.concatenate([[0], counts[:-1]])
    prev3 = [int(np.sum(counts[max(0,i-3):i])) for i in range(len(counts))]
    prev6 = [int(np.sum(counts[max(0,i-6):i])) for i in range(len(counts))]
    prevavg = [np.mean(counts[max(0,i-24):i]) if i>0 else 0.0 for i in range(len(counts))]
    panel.loc[idx, 'prev_1'] = prev1
    panel.loc[idx, 'prev_3'] = prev3
    panel.loc[idx, 'prev_6'] = prev6
    panel.loc[idx, 'prev_avg'] = prevavg

# ---------- Regression dataset ----------
features = ['hour_of_day','dayofweek','is_weekend','total_txns','avg_amount_by_user','prev_1','prev_3','prev_6','prev_avg']
X = panel[features].fillna(0)
y = panel['count'].values

from sklearn.model_selection import train_test_split
users = panel['user_id'].unique()
train_users, test_users = train_test_split(users, test_size=0.2, random_state=RND)
train_mask = panel['user_id'].isin(train_users)
X_train = X[train_mask]; y_train = y[train_mask]
X_test = X[~train_mask]; y_test = y[~train_mask]

# ---------- Fit Poisson GLM and Negative Binomial (statsmodels) ----------
import statsmodels.api as sm
X_train_sm = sm.add_constant(X_train); X_test_sm = sm.add_constant(X_test)
poisson_model = sm.GLM(y_train, X_train_sm, family=sm.families.Poisson()).fit(maxiter=100)
nb_model = sm.GLM(y_train, X_train_sm, family=sm.families.NegativeBinomial()).fit(maxiter=100)

joblib.dump(poisson_model, os.path.join(OUT_DIR, "poisson_glm_result.pkl"))
joblib.dump(nb_model, os.path.join(OUT_DIR, "nb_glm_result.pkl"))

# ---------- Residuals & simple CUSUM threshold ----------
yhat_poisson = poisson_model.predict(X_test_sm)
resid_poisson = y_test - yhat_poisson
z_poisson = resid_poisson / (np.sqrt(yhat_poisson) + 1e-9)
score_poisson_abs = np.abs(resid_poisson)

# compute threshold from train residuals
yhat_train = poisson_model.predict(X_train_sm)
resid_train = y_train - yhat_train
thr_cusum = float(resid_train.mean() + 4 * resid_train.std())
with open(os.path.join(OUT_DIR, "cusum_threshold.json"), "w") as f:
    json.dump({"threshold": thr_cusum}, f)

# Save meta
joblib.dump({"features": features}, os.path.join(OUT_DIR, "poisson_meta.pkl"))

# ---------- Isolation Forest ----------
from sklearn.ensemble import IsolationForest
iso_features = ['prev_1','prev_3','prev_6','prev_avg','hour_of_day','dayofweek','total_txns']
iso_X = panel[iso_features].fillna(0).values
from sklearn.preprocessing import StandardScaler
scaler_iso = StandardScaler().fit(iso_X)
iso_X_scaled = scaler_iso.transform(iso_X)
iso = IsolationForest(n_estimators=IF_ESTIMATORS, contamination=0.02, random_state=RND, n_jobs=-1)
iso.fit(iso_X_scaled[train_mask.values])
joblib.dump(iso, os.path.join(OUT_DIR, "isolation_forest_freq.pkl"))
joblib.dump(scaler_iso, os.path.join(OUT_DIR, "isolation_scaler.pkl"))

# ---------- Sequence GRU model on sliding windows (seq-to-one) ----------
# Build sliding windows per user
seq_X = []; seq_y = []; meta = []
for uid, g in panel.groupby('user_id'):
    counts = g['count'].values
    frauds = g['fraud_present'].values
    n = len(counts)
    if n < SEQ_LEN: continue
    for start in range(0, n - SEQ_LEN + 1):
        end = start + SEQ_LEN
        seq_X.append(counts[start:end])
        seq_y.append(int(frauds[end-1]))
        meta.append(uid)
X_seq = np.array(seq_X); y_seq = np.array(seq_y)
print("Seq windows:", X_seq.shape, "Positives:", int(y_seq.sum()))

# split by user
unique_users_seq = np.unique(meta)
train_u_seq, test_u_seq = train_test_split(unique_users_seq, test_size=0.2, random_state=RND)
mask_seq = np.array([u in train_u_seq for u in meta])
X_seq_train = X_seq[mask_seq]; y_seq_train = y_seq[mask_seq]
X_seq_test = X_seq[~mask_seq]; y_seq_test = y_seq[~mask_seq]

# scale sequences per-feature
from sklearn.preprocessing import StandardScaler
scaler_seq = StandardScaler()
scaler_seq.fit(X_seq_train.reshape(-1,1))
X_seq_train_s = scaler_seq.transform(X_seq_train.reshape(-1,1)).reshape(X_seq_train.shape)
X_seq_test_s = scaler_seq.transform(X_seq_test.reshape(-1,1)).reshape(X_seq_test.shape)
joblib.dump(scaler_seq, os.path.join(OUT_DIR, "scaler_seq.joblib"))

# Train small GRU (TF) if available else RF fallback
use_tf = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense, Dropout, InputLayer
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    use_tf = True
    print("TensorFlow available:", tf.__version__)
except Exception:
    print("TF not available – will use RF fallback for sequences")

if use_tf and X_seq_train_s.shape[0] > 8:
    tf.random.set_seed(RND)
    model = Sequential([InputLayer(input_shape=(SEQ_LEN,1)), GRU(32), Dropout(0.2), Dense(16, activation='relu'), Dense(1, activation='sigmoid')])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    Xtr = X_seq_train_s.reshape((X_seq_train_s.shape[0], SEQ_LEN, 1))
    Xte = X_seq_test_s.reshape((X_seq_test_s.shape[0], SEQ_LEN, 1))
    es = EarlyStopping(monitor='val_auc', mode='max', patience=3, restore_best_weights=True)
    mc = ModelCheckpoint(os.path.join(OUT_DIR, "gru_seq_best.h5"), monitor='val_auc', mode='max', save_best_only=True, verbose=0)
    model.fit(Xtr, y_seq_train, validation_data=(Xte, y_seq_test), epochs=GRU_EPOCHS, batch_size=64, callbacks=[es,mc], verbose=2)
    model.save(os.path.join(OUT_DIR, "gru_seq_model.h5"))
    print("Saved GRU model to", os.path.join(OUT_DIR, "gru_seq_model.h5"))
else:
    from sklearn.ensemble import RandomForestClassifier
    def seq_stats(arr): return np.array([arr.mean(), arr.std(), arr.max(), arr.min(), arr[-1], arr.sum()])
    Xtr_feat = np.array([seq_stats(s) for s in X_seq_train])
    Xte_feat = np.array([seq_stats(s) for s in X_seq_test])
    clf = RandomForestClassifier(n_estimators=100, random_state=RND, n_jobs=-1)
    clf.fit(Xtr_feat, y_seq_train)
    joblib.dump(clf, os.path.join(OUT_DIR, "rf_seq_fallback.pkl"))
    print("Saved RF sequence fallback to", os.path.join(OUT_DIR, "rf_seq_fallback.pkl"))

# ---------- Save meta file ----------
meta = {
    "interval_hours": INTERVAL_HOURS,
    "features_regression": features,
    "iso_features": iso_features,
    "seq_len": SEQ_LEN,
    "artifacts": {
        "poisson": os.path.join(OUT_DIR, "poisson_glm_result.pkl"),
        "nb": os.path.join(OUT_DIR, "nb_glm_result.pkl"),
        "cusum_threshold": os.path.join(OUT_DIR, "cusum_threshold.json"),
        "isolation_forest": os.path.join(OUT_DIR, "isolation_forest_freq.pkl"),
        "isolation_scaler": os.path.join(OUT_DIR, "isolation_scaler.pkl"),
        "sequence_model": os.path.join(OUT_DIR, "gru_seq_model.h5") if use_tf and X_seq_train_s.shape[0]>8 else os.path.join(OUT_DIR, "rf_seq_fallback.pkl"),
        "scaler_seq": os.path.join(OUT_DIR, "scaler_seq.joblib")
    }
}
with open(os.path.join(OUT_DIR, "meta_all.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("Done. Artifacts saved to:", OUT_DIR)
