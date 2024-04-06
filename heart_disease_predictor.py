import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
data = pd.read_csv(url, names=names)

data = data.replace('?', np.nan)
data = data.dropna()

X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

def get_user_input():
    age = int(input("Age: "))
    sex = int(input("Sex (0 for female, 1 for male): "))
    cp = int(input("Chest Pain Type (0-3): "))
    trestbps = int(input("Resting Blood Pressure (mm Hg): "))
    chol = int(input("Cholesterol (mg/dl): "))
    fbs = int(input("Fasting Blood Sugar > 120 mg/dl (1 for true, 0 for false): "))
    restecg = int(input("Resting Electrocardiographic Results (0-2): "))
    thalach = int(input("Maximum Heart Rate Achieved (bpm): "))
    exang = int(input("Exercise Induced Angina (1 for yes, 0 for no): "))
    oldpeak = float(input("ST Depression Induced by Exercise Relative to Rest: "))
    slope = int(input("Slope of the Peak Exercise ST Segment (0-2): "))
    ca = int(input("Number of Major Vessels Colored by Fluoroscopy (0-3): "))
    thal = int(input("Thalassemia (0-3): "))
    
    return np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

user_data = get_user_input()

prediction = model.predict(user_data)
print("Predicted likelihood of having heart disease:", prediction[0])

percentage_chance = (prediction[0] / 4) * 100
print("Percentage chance of having heart disease:", percentage_chance)
