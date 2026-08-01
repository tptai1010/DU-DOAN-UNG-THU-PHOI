from catboost import CatBoostClassifier
from sklearn import linear_model
import pandas as pd
import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC

# Đọc dữ liệu từ tập tin
df = pd.read_csv('../data/lung_cancer_preprocessed.csv')

X = df.iloc[:, : -1]
y = df.iloc[:, -1]

decision_tree = DecisionTreeClassifier(max_depth=8, criterion='entropy',random_state=42)
bagging = BaggingClassifier(decision_tree, n_estimators=300, bootstrap=True, random_state=42, n_jobs=8)
bagging.fit(X, y)

pickle.dump(bagging, open('bagging_dt_model.pkl','wb')) # save the model

#print(knn.predict([[15, 61]]))  # format of input
print(f'score: {bagging.score(X, y)}')
