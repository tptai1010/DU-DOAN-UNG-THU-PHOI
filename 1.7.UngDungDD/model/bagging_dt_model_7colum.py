from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import pickle
import os

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
decision_tree = DecisionTreeClassifier(max_depth=8, criterion='entropy',random_state=42)
bagging = BaggingClassifier(decision_tree, n_estimators=300, bootstrap=True, random_state=42, n_jobs=8)
bagging.fit(X, y)

# ==================================================
# 5️⃣ ĐÁNH GIÁ MODEL
# ==================================================

y_pred = bagging.predict(X)

print("\n===== ĐÁNH GIÁ TRAINING =====")
print("Accuracy :", accuracy_score(y, y_pred))
print("Precision:", precision_score(y, y_pred, average='weighted'))
print("Recall   :", recall_score(y, y_pred, average='weighted'))
print("F1-score :", f1_score(y, y_pred, average='weighted'))

# ==================================================
# 6️⃣ LƯU MODEL
# ==================================================

model_path = os.path.join(PROJECT_ROOT, "bagging_dt_model_7colum.pkl")

with open(model_path, "wb") as f:
    pickle.dump(bagging, f)

print("\n✅ Model đã lưu tại:", model_path)
print("🎉 Hoàn tất training!")