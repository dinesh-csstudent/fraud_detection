"""
Credit Card Fraud Detection
Autoencoder (unsupervised feature extractor) + XGBoost (supervised classifier)

Run this script first to train and save models before starting the Flask app.
Usage: python train_model.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import time
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import Callback, EarlyStopping
from xgboost import XGBClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class TimingCallback(Callback):
    def on_train_begin(self, logs=None):
        self.train_start_time = time.time()

    def on_train_end(self, logs=None):
        total = time.time() - self.train_start_time
        print(f"\nTotal training time: {total:.2f} seconds.")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        duration = time.time() - self.epoch_start_time
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | loss: {logs.get('loss',0):.6f} | val_loss: {logs.get('val_loss',0):.6f} | {duration:.2f}s")


# ── 1. Load Data ──────────────────────────────
print("=" * 60)
print("FRAUD DETECTION: Autoencoder + XGBoost Pipeline")
print("=" * 60)

if not os.path.exists("creditcard.csv"):
    raise FileNotFoundError(
        "creditcard.csv not found.\n"
        "Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
    )

print("\n[1/6] Loading dataset...")
df = pd.read_csv("creditcard.csv")
print(f"  Shape: {df.shape}")
print(f"  Missing values: {df.isnull().values.sum()}")
print(f"  Class distribution:\n{df['Class'].value_counts().to_string()}")
print(f"  Fraud rate: {df['Class'].mean()*100:.4f}%")


# ── 2. Preprocessing (split FIRST, scale after to avoid leakage) ──
print("\n[2/6] Preprocessing...")
X = df.drop(columns=['Class'])
y = df['Class']
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"  Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")


# ── 3. Build Autoencoder via Functional API ───
print("\n[3/6] Building Autoencoder (Functional API)...")
input_dim    = X_train_scaled.shape[1]   # 30
encoding_dim = 20                         # bottleneck / latent space

inputs   = Input(shape=(input_dim,), name='encoder_input')
enc1     = Dense(25, activation='relu',         name='enc_1')(inputs)
bottleneck = Dense(encoding_dim, activation='relu', name='bottleneck')(enc1)
dec1     = Dense(25, activation='relu',         name='dec_1')(bottleneck)
outputs  = Dense(input_dim, activation='sigmoid', name='decoder_output')(dec1)

autoencoder = Model(inputs, outputs,    name='autoencoder')
encoder     = Model(inputs, bottleneck, name='encoder')   # encoder-only model

autoencoder.compile(optimizer='adam', loss='mse')
print(f"  Input dim: {input_dim}  →  Latent dim: {encoding_dim}")


# ── 4. Train Autoencoder (unsupervised — labels NOT used) ──
print("\n[4/6] Training Autoencoder (unsupervised)...")
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
timing_cb  = TimingCallback()

history = autoencoder.fit(
    X_train_scaled, X_train_scaled,
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_split=0.2,
    verbose=0,
    callbacks=[timing_cb, early_stop]
)

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'],     label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
plt.title('Autoencoder Training Loss')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig('models/autoencoder_loss.png', dpi=120); plt.close()
print("  Loss plot saved → models/autoencoder_loss.png")


# ── 5. Encode Features → XGBoost ─────────────
print("\n[5/6] Extracting latent features → training XGBoost...")
X_train_latent = encoder.predict(X_train_scaled, verbose=0)
X_test_latent  = encoder.predict(X_test_scaled,  verbose=0)
print(f"  Latent shape: {X_train_latent.shape}")

class_ratio = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=class_ratio,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)
xgb.fit(X_train_latent, y_train)
print("  XGBoost trained ✓")


# ── 6. Evaluate ───────────────────────────────
print("\n[6/6] Evaluating...")
y_pred       = xgb.predict(X_test_latent)
y_pred_proba = xgb.predict_proba(X_test_latent)[:, 1]

print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC: {roc_auc:.4f}")

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}', color='crimson')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('FPR'); plt.ylabel('TPR')
plt.title('ROC Curve — Autoencoder + XGBoost')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig('models/roc_curve.png', dpi=120); plt.close()

tn, fp, fn, tp = cm.ravel()
metrics = {
    "roc_auc":   round(roc_auc, 4),
    "precision": round(tp / (tp + fp) if (tp+fp) > 0 else 0, 4),
    "recall":    round(tp / (tp + fn) if (tp+fn) > 0 else 0, 4),
    "f1":        round(2*tp / (2*tp+fp+fn) if (2*tp+fp+fn) > 0 else 0, 4),
    "accuracy":  round((tp+tn) / (tp+tn+fp+fn), 4),
    "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    "encoding_dim": encoding_dim, "input_dim": input_dim,
    "train_samples": int(len(X_train_scaled)),
    "test_samples":  int(len(X_test_scaled))
}
with open('models/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)


# ── 7. Save Models ────────────────────────────
encoder.save('models/encoder.keras')
with open('models/xgboost.pkl', 'wb') as f:
    pickle.dump(xgb, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/feature_names.json', 'w') as f:
    json.dump(feature_names, f)

print("\n" + "=" * 60)
print("All models saved to models/")
print("Next step: python app.py")
print("=" * 60)
