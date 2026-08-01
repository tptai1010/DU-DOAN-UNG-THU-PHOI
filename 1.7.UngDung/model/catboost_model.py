from catboost import CatBoostClassifier
from sklearn import linear_model
import pandas as pd
import pickle
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC

# Đọc dữ liệu từ tập tin
df = pd.read_csv('../data/lung_cancer_preprocessed.csv')

X = df.iloc[:, : -1]
y = df.iloc[:, -1]

catboost = CatBoostClassifier(iterations = 100, depth = 7, learning_rate = 0.05, random_state=42)
catboost.fit(X, y)

pickle.dump(catboost, open('catboost_model.pkl','wb')) # save the model

#print(knn.predict([[15, 61]]))  # format of input
print(f'score: {catboost.score(X, y)}')
