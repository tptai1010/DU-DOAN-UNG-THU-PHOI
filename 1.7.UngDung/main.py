from flask import Flask, render_template, request, flash
import pickle
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)
app.secret_key = "secret123"

model_cat = pickle.load(open('model/catboost_model.pkl', 'rb'))
model_rf = pickle.load(open('model/random_forest_model.pkl', 'rb'))
model_bagg = pickle.load(open('model/bagging_dt_model.pkl', 'rb'))
model_cat_nguy_co = pickle.load(open('model/catboost_model_7colum.pkl', 'rb'))
model_rf_nguy_co = pickle.load(open('model/random_forest_model_7colum.pkl', 'rb'))
model_bagg_nguy_co = pickle.load(open('model/bagging_dt_model_7colum.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

@app.route("/")
def hello():
    return render_template(
        "index.html",
        row_data=[],
        column_names=[]
    )

@app.route("/single")
def single():
    return render_template(
        "single.html",
        row_data=[],
        column_names=[]
        )

@app.route("/multiple")
def multiple():
    return render_template(
        "multiple.html",
        row_data=[],
        column_names=[]
    )

@app.route("/predict", methods=['POST'])
def predict():

    model_name = request.form.get("model")
    file = request.files.get("file")

    # =========================
    # CHỌN MODEL
    # =========================
    if model_name == "catboost":
        model = model_cat
        model_risk = model_cat_nguy_co
    elif model_name == "randomforest":
        model = model_rf
        model_risk = model_rf_nguy_co
    elif model_name == "bagging":
        model = model_bagg
        model_risk = model_bagg_nguy_co
    else:
        flash("Vui lòng chọn model")
        return render_template(
            "multiple.html",
            row_data=[],
            column_names=[]
        )

    print("Model đang sử dụng:", model_name)
    
   # =========================
    # SINGLE PREDICTION
    # =========================
    if file is None or file.filename == "":

        try:
            data = {}
            patient_name = request.form.get("patient_name")

            for key in request.form:
                if key not in ["model", "patient_name"]:
                    value = request.form.get(key)
                    try:
                        value = float(value)
                    except:
                        pass

                    data[key] = value

            # ----------------------
            # TẠO DATAFRAME
            # ----------------------
            df = pd.DataFrame([data])
            df.columns = df.columns.str.upper()

            numeric_cols = ['AGE','ENERGY_LEVEL','OXYGEN_SATURATION']
            df[numeric_cols] = scaler.transform(df[numeric_cols])

            if hasattr(model, "feature_names_in_"):
                df = df[model.feature_names_in_]

            elif hasattr(model, "feature_names_"):
                df = df[model.feature_names_]

            print(df)
            if hasattr(model,"feature_names_in_"):
                print(model.feature_names_in_)
            elif hasattr(model,"feature_names_"):
                print(model.feature_names_)

            # ----------------------
            # PREDICT BỆNH
            # ----------------------
            y_pred = model.predict(df)
            prob = model.predict_proba(df)[0][1]
            
            # =========================
            # FEATURE IMPORTANCE TOP 10
            # =========================

            feature_importance = None
            features = df.columns.tolist()

            # RandomForest / DecisionTree
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_

                feature_importance = dict(zip(features, importances))
                feature_importance = dict(
                    sorted(feature_importance.items(),
                           key=lambda x: x[1],
                           reverse=True)[:10]
                )

            # Bagging Decision Tree
            elif model_name == "bagging":
                import numpy as np

                importances = []

                for tree in model.estimators_:
                    importances.append(tree.feature_importances_)

                mean_importance = np.mean(importances, axis=0)

                feature_importance = dict(zip(features, mean_importance))
                feature_importance = dict(
                    sorted(feature_importance.items(),
                           key=lambda x: x[1],
                           reverse=True)[:10]
                )

            result = "Lung Cancer" if y_pred[0] == 1 else "No Lung Cancer"

            risk_text = None

            # =========================
            # PREDICT RISK LEVEL
            # =========================
            if y_pred[0] == 0:

                df_risk = df[[
                    "SMOKING",
                    "AGE",
                    "FAMILY_HISTORY",
                    "LONG_TERM_ILLNESS",
                    "OXYGEN_SATURATION",
                    "EXPOSURE_TO_POLLUTION",
                    "BREATHING_ISSUE",
                    "CHEST_TIGHTNESS",
                    "THROAT_DISCOMFORT"
                ]].copy()

                # ----------------------
                # FEATURE ENGINEERING
                # ----------------------

                df_risk["age_risk"] = df_risk["AGE"].apply(
                    lambda x: 1 if x >= 60 else 0
                )

                df_risk["smoking_risk"] = df_risk["SMOKING"]

                df_risk["family_history_risk"] = df_risk["FAMILY_HISTORY"]

                df_risk["chronic_disease_risk"] = df_risk["LONG_TERM_ILLNESS"]

                df_risk["low_oxygen_risk"] = df_risk["OXYGEN_SATURATION"].apply(
                    lambda x: 1 if x < 92 else 0
                )

                df_risk["exposure_risk"] = df_risk["EXPOSURE_TO_POLLUTION"]

                df_risk["symptom_risk"] = df_risk[
                    ["BREATHING_ISSUE","CHEST_TIGHTNESS","THROAT_DISCOMFORT"]
                ].max(axis=1)

                # giữ đúng 7 feature
                df_risk = df_risk[[
                    "age_risk",
                    "smoking_risk",
                    "family_history_risk",
                    "chronic_disease_risk",
                    "low_oxygen_risk",
                    "exposure_risk",
                    "symptom_risk"
                ]]

                # sắp xếp đúng model
                if hasattr(model_risk, "feature_names_in_"):
                    df_risk = df_risk[model_risk.feature_names_in_]

                risk_pred = model_risk.predict(df_risk)[0]

                if risk_pred == 0:
                    risk_text = "High Risk"

                elif risk_pred == 2:
                    risk_text = "Moderate Risk"

                else:
                    risk_text = "Low Risk"

            return render_template(
                "single.html",
                patient_name=patient_name,
                prediction=result,
                probability=round(prob,3),
                risk_level=risk_text,
                model=request.form["model"],
                form_data=request.form,
                feature_importance=feature_importance
            )

        except Exception as e:

            flash("Lỗi dự đoán dữ liệu nhập!")
            print(e)

            return render_template(
                "single.html",
                prediction=None
            )

    # =========================
    # MULTIPLE PREDICTION
    # =========================
    if file is None:
        flash("Vui lòng upload file CSV")
        return render_template("multiple.html",
                               row_data=[],
                               column_names=[])

    if not file.filename.endswith(".csv"):
        flash("File phải là CSV")
        return render_template("multiple.html",
                               row_data=[],
                               column_names=[])
    # đọc file
    try:
        df1 = pd.read_csv(file)

        # convert YES/NO
        df1.replace({
            "YES":1,"Yes":1,"yes":1,
            "NO":0,"No":0,"no":0
        }, inplace=True)

        X_test = df1.iloc[:,1:-1]
        y_test = df1.iloc[:,-1]

        ms = df1.iloc[:,0]

        # sắp xếp feature
        try:
            X_test = X_test[model.feature_names_]
        except:
            pass

    except Exception as e:
        flash("Lỗi đọc file CSV!")
        print(e)

        return render_template(
            "multiple.html",
            column_names=[],
            row_data=[],
            zip=zip
        )
    # dự đoán bệnh
    try:
        y_pred = model.predict(X_test)

    except Exception as e:
        flash("Lỗi khi dự đoán!")
        print(e)

        return render_template(
            "multiple.html",
            column_names=[],
            row_data=[],
            zip=zip
        )

    df1["Disease_Prediction"] = y_pred

    # =========================
    # LỌC BỆNH / KHÔNG BỆNH
    # =========================
    df_positive = df1[df1["Disease_Prediction"] == 1].reset_index(drop=True)
    df_negative = df1[df1["Disease_Prediction"] == 0].reset_index(drop=True)

    # =========================
    # TẠO DATASET TẠM CHO MODEL NGUY CƠ
    # =========================
    if len(df_negative) > 0:

        df_risk = df_negative[[
            "SMOKING",
            "AGE",
            "FAMILY_HISTORY",
            "LONG_TERM_ILLNESS",
            "OXYGEN_SATURATION",
            "EXPOSURE_TO_POLLUTION",
            "BREATHING_ISSUE",
            "CHEST_TIGHTNESS",
            "THROAT_DISCOMFORT"
        ]].copy()

        # gộp symptom
        df_risk["SYMPTOM_RISK"] = df_risk[
            ["BREATHING_ISSUE","CHEST_TIGHTNESS","THROAT_DISCOMFORT"]
        ].max(axis=1)
            
        df_risk.drop(
            columns=["BREATHING_ISSUE","CHEST_TIGHTNESS","THROAT_DISCOMFORT"],
            inplace=True
        )

        # rename theo model train
        df_risk.rename(columns={
            "AGE":"age_risk",
            "SMOKING":"smoking_risk",
            "FAMILY_HISTORY":"family_history_risk",
            "LONG_TERM_ILLNESS":"chronic_disease_risk",
            "OXYGEN_SATURATION":"low_oxygen_risk",
            "EXPOSURE_TO_POLLUTION":"exposure_risk",
            "SYMPTOM_RISK":"symptom_risk"
        }, inplace=True)

        # sắp xếp feature đúng thứ tự khi train
        if hasattr(model_risk, "feature_names_in_"):
            df_risk = df_risk[model_risk.feature_names_in_]

        # dự đoán risk
        risk_pred = model_risk.predict(df_risk)

    else:
        risk_pred = []
        
    # =========================
    # EVALUATION (IN TERMINAL)
    # =========================
    from sklearn.metrics import accuracy_score

    correct = (y_pred == y_test).sum()
    wrong = (y_pred != y_test).sum()

    print("\n========== PREDICTION RESULT ==========")
    print(f"Total samples: {len(y_test)}")
    print(f"Correct predictions: {correct}")
    print(f"Wrong predictions: {wrong}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # =========================
    # IN CÁC MẪU DỰ ĐOÁN SAI
    # =========================
    print("\n========== WRONG SAMPLES ==========")

    wrong_index = y_test[y_test != y_pred].index

    for i in wrong_index:
        print(f"MS: {df1.iloc[i,0]} | True: {y_test[i]} | Predicted: {y_pred[i]}")

    # =========================
    # CHUYỂN SANG TEXT
    # =========================
    risk_text = []

    for x in risk_pred:

        if x == 0:
            risk_text.append("High Risk")

        elif x == 2:
            risk_text.append("Moderate Risk")

        else:
            risk_text.append("Low Risk")

    # =========================
    # TẠO BẢNG HIỂN THỊ
    # =========================
    df_positive_display = pd.DataFrame({
        "MS": df_positive.iloc[:,0],
        "Disease Prediction": "Lung Cancer"
    })

    df_negative_display = pd.DataFrame({
        "MS": df_negative.iloc[:,0],
        "Disease Prediction": "No Disease",
        "Risk Level": risk_text
    })

    df_result = pd.concat(
        [df_positive_display, df_negative_display],
        ignore_index=True
    )

    df_result = df_result.sort_values(by="MS")
    
    return render_template(
        "multiple.html",
        column_names=df_result.columns.values,
        row_data=list(df_result.values.tolist()),
        zip=zip
    )

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)