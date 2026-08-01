import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# đọc dataset
df = pd.read_csv('../data/lung_cancer_preprocessed.csv')

X = df.iloc[:, :-1]

# feature cần scale
numeric_cols = ['AGE','ENERGY_LEVEL','OXYGEN_SATURATION']

scaler = StandardScaler()
scaler.fit(X[numeric_cols])

# lưu scaler
pickle.dump(scaler, open('scaler.pkl','wb'))

print("Scaler saved")