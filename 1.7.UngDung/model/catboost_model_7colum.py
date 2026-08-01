from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import pickle
import os
from catboost import CatBoostClassifier

# ==================================================
# 1️⃣ XÁC ĐỊNH ĐƯỜNG DẪN THƯ MỤC GỐC PROJECT
# ==================================================

# Thư mục chứa file .py hiện tại
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Lùi lên 1 cấp (từ Model -> CBHV_CNTT)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# ==================================================
# 2️⃣ ĐƯỜNG DẪN FILE CSV
# ==================================================

data_path = os.path.join(PROJECT_ROOT, "data", "Lung_Cancer_Risk_3Levels_Normalized.csv")

print("📂 Đang đọc file tại:", data_path)
print("📌 File tồn tại?", os.path.exists(data_path))

if not os.path.exists(data_path):
    raise FileNotFoundError("❌ Không tìm thấy file CSV. Kiểm tra lại thư mục data.")

# ==================================================
# 3️⃣ ĐỌC DỮ LIỆU
# ==================================================

df = pd.read_csv(data_path)

X = df.iloc[:, :-2]  
y = df.iloc[:, -1]

# ==================================================
# 4️⃣ TRAIN RANDOM FOREST
# ==================================================

catboost = CatBoostClassifier(iterations = 100, depth = 7, learning_rate = 0.05, random_state=42)
catboost.fit(X, y)

# ==================================================
# 5️⃣ ĐÁNH GIÁ MODEL
# ==================================================

y_pred = catboost.predict(X)

print("\n===== ĐÁNH GIÁ TRAINING =====")
print("Accuracy :", accuracy_score(y, y_pred))
print("Precision:", precision_score(y, y_pred, average='weighted'))
print("Recall   :", recall_score(y, y_pred, average='weighted'))
print("F1-score :", f1_score(y, y_pred, average='weighted'))

# ==================================================
# 6️⃣ LƯU MODEL
# ==================================================

model_path = os.path.join(PROJECT_ROOT, "catboost_model_7colum.pkl")

with open(model_path, "wb") as f:
    pickle.dump(catboost, f)

print("\n✅ Model đã lưu tại:", model_path)
print("🎉 Hoàn tất training!")