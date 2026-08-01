from catboost import CatBoostClassifier
from sklearn import linear_model
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC

# Đọc dữ liệu từ tập tin
df = pd.read_csv('../data/lung_cancer_preprocessed.csv')

X = df.iloc[:, : -1]
y = df.iloc[:, -1]

random_forest = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
random_forest.fit(X, y)

pickle.dump(random_forest, open('random_forest_model.pkl','wb')) # save the model

#print(knn.predict([[15, 61]]))  # format of input
print(f'score: {random_forest.score(X, y)}')
