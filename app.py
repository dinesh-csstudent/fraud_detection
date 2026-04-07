"""
Flask Backend — Credit Card Fraud Detection
Pipeline: raw input → StandardScaler → Encoder (latent) → XGBoost → prediction
"""

import os, json, pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf

app = Flask(__name__)
CORS(app)

MODELS_DIR    = os.path.join(os.path.dirname(__file__), 'models')
encoder       = None
xgb_model     = None
scaler        = None
feature_names = None
model_metrics = None


def load_models():
    global encoder, xgb_model, scaler, feature_names, model_metrics
    paths = {
        'encoder':  os.path.join(MODELS_DIR, 'encoder.keras'),
        'xgb':      os.path.join(MODELS_DIR, 'xgboost.pkl'),
        'scaler':   os.path.join(MODELS_DIR, 'scaler.pkl'),
        'features': os.path.join(MODELS_DIR, 'feature_names.json'),
        'metrics':  os.path.join(MODELS_DIR, 'metrics.json'),
    }
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        print(f"[WARNING] Missing: {missing}. Run train_model.py first!")
        return False

    encoder = tf.keras.models.load_model(paths['encoder'])
    with open(paths['xgb'],      'rb') as f: xgb_model     = pickle.load(f)
    with open(paths['scaler'],   'rb') as f: scaler        = pickle.load(f)
    with open(paths['features']) as f:       feature_names = json.load(f)
    with open(paths['metrics'])  as f:       model_metrics = json.load(f)
    print("[OK] Models loaded.")
    return True


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def status():
    ready = encoder is not None
    return jsonify({
        "status": "ready" if ready else "not_ready",
        "message": "Models loaded" if ready else "Run train_model.py first",
        "metrics": model_metrics
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Single transaction: raw features → scale → encode → classify."""
    if encoder is None:
        return jsonify({"error": "Models not loaded. Run train_model.py first."}), 503

    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({"error": "Missing 'features' in JSON body."}), 400

    features = data['features']
    if len(features) != len(feature_names):
        return jsonify({"error": f"Expected {len(feature_names)} features, got {len(features)}."}), 400

    try:
        raw    = np.array(features, dtype=np.float32).reshape(1, -1)
        scaled = scaler.transform(raw)
        latent = encoder.predict(scaled, verbose=0)
        pred   = int(xgb_model.predict(latent)[0])
        proba  = float(xgb_model.predict_proba(latent)[0][1])

        return jsonify({
            "prediction":    pred,
            "label":         "FRAUD" if pred == 1 else "NORMAL",
            "confidence":    round(proba * 100, 2),
            "latent_vector": latent[0].tolist(),
            "latent_dim":    int(latent.shape[1])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """Multiple transactions: list of feature arrays."""
    if encoder is None:
        return jsonify({"error": "Models not loaded."}), 503

    data = request.get_json()
    if not data or 'transactions' not in data:
        return jsonify({"error": "Missing 'transactions' key."}), 400

    try:
        raw    = np.array(data['transactions'], dtype=np.float32)
        scaled = scaler.transform(raw)
        latent = encoder.predict(scaled, verbose=0)
        preds  = xgb_model.predict(latent).tolist()
        probas = xgb_model.predict_proba(latent)[:, 1].tolist()

        results = [{
            "index":      i,
            "prediction": preds[i],
            "label":      "FRAUD" if preds[i] == 1 else "NORMAL",
            "confidence": round(probas[i] * 100, 2)
        } for i in range(len(preds))]

        return jsonify({
            "total":        len(results),
            "fraud_count":  sum(p == 1 for p in preds),
            "normal_count": sum(p == 0 for p in preds),
            "results":      results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/features')
def get_features():
    if feature_names is None:
        return jsonify({"error": "Models not loaded."}), 503
    return jsonify({"features": feature_names, "count": len(feature_names)})


@app.route('/api/metrics')
def get_metrics():
    if model_metrics is None:
        return jsonify({"error": "Models not loaded."}), 503
    return jsonify(model_metrics)


if __name__ == '__main__':
    load_models()
    app.run(debug=True, host='0.0.0.0', port=5000)
