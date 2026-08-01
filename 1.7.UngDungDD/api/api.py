from flask import Blueprint, request
import pandas as pd
import pickle
import os

api = Blueprint("api", __name__)

# ==============LOAD MODEL=====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, '..', 'model')

model_cat = pickle.load(open(os.path.join(MODEL_DIR, 'catboost_model.pkl'), 'rb'))
model_rf = pickle.load(open(os.path.join(MODEL_DIR, 'random_forest_model.pkl'), 'rb'))
model_bagg = pickle.load(open(os.path.join(MODEL_DIR, 'bagging_dt_model.pkl'), 'rb'))

scaler = pickle.load(open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb'))

# ===== API PREDICT =====
@api.route("/api/predict", methods=["POST"])
def predict():

    data = request.get_json()
    model_name = data.get("model")

    # ===== CHỌN MODEL =====
    if model_name == "catboost":
        model = model_cat
    elif model_name == "randomforest":
        model = model_rf
    elif model_name == "bagging":
        model = model_bagg
    else:
        return {"error": "No model selected"}

    try:
        # ===== DATAFRAME =====
        df = pd.DataFrame([data])
        df.columns = df.columns.str.upper()

        # ===== SCALE =====
        numeric_cols = ['AGE','ENERGY_LEVEL','OXYGEN_SATURATION']
        df[numeric_cols] = scaler.transform(df[numeric_cols])

        # ===== SẮP XẾP FEATURE =====
        if hasattr(model, "feature_names_in_"):
            df = df[model.feature_names_in_]
        elif hasattr(model, "feature_names_"):
            df = df[model.feature_names_]

        # ===== PREDICT =====
        y_pred = model.predict(df)
        prob = model.predict_proba(df)[0][1]

        result = "Lung Cancer" if y_pred[0] == 1 else "No Lung Cancer"

        # ===== RISK (đơn giản) =====
        risk = None
        if y_pred[0] == 0:
            if prob > 0.7:
                risk = "High Risk"
            elif prob > 0.4:
                risk = "Moderate Risk"
            else:
                risk = "Low Risk"

        return {
            "prediction": result,
            "probability": float(prob),
            "risk_level": risk
        }

    except Exception as e:
        return {"error": str(e)}