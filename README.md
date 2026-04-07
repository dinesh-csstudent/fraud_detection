# FraudScan — Credit Card Fraud Detection
**Autoencoder (unsupervised) + XGBoost (supervised) hybrid system**

## Architecture
```
Raw Transaction (30 features)
        ↓
  StandardScaler          ← fit on train only (no data leakage)
        ↓
  Autoencoder Encoder     ← unsupervised: learns normal patterns
  [30 → 25 → 20]         ← extracts 20-dim latent representation
        ↓
  XGBoost Classifier      ← supervised: fraud vs normal
        ↓
  Prediction + Confidence
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download dataset
Get `creditcard.csv` from:  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
Place it in the project root.

### 3. Train models
```bash
python train_model.py
```
This saves to `models/`:
- `encoder.keras` — encoder-only model (for inference)
- `xgboost.pkl` — trained classifier
- `scaler.pkl` — StandardScaler (fit on training data)
- `feature_names.json`
- `metrics.json` — ROC-AUC, precision, recall, F1, confusion matrix
- `autoencoder_loss.png`, `roc_curve.png`

### 4. Start Flask server
```bash
python app.py
```
Open: http://localhost:5000

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/status` | Model load status + metrics |
| POST | `/api/predict` | Single transaction prediction |
| POST | `/api/batch_predict` | Multiple transactions |
| GET | `/api/features` | Feature names |
| GET | `/api/metrics` | Training metrics |

### `/api/predict` example
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0, -1.35, -0.07, 2.53, ...]}'
```

Response:
```json
{
  "prediction": 0,
  "label": "NORMAL",
  "confidence": 98.34,
  "latent_vector": [0.12, 0.44, ...],
  "latent_dim": 20
}
```

## Key Fixes vs Original Code
1. **Proper encoder extraction** — Uses Keras Functional API to split encoder from decoder; inference uses only the encoder (latent features → XGBoost), not the full reconstruction output.
2. **No data leakage** — `StandardScaler.fit()` called only on training data; test set is only transformed.
3. **EarlyStopping** — Avoids overfitting the autoencoder.
4. **Cleaner experiment loop** — Encoding dim experiments are easy to extend.

## Project Structure
```
fraud_detection/
├── train_model.py     ← Train & save models
├── app.py             ← Flask API + serve frontend
├── requirements.txt
├── creditcard.csv     ← (you provide)
├── models/            ← (auto-created by train_model.py)
│   ├── encoder.keras
│   ├── xgboost.pkl
│   ├── scaler.pkl
│   ├── feature_names.json
│   ├── metrics.json
│   └── *.png
└── templates/
    └── index.html     ← Frontend dashboard
```
