import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# Simple preprocessing
df = df[["Sex", "Fare", "Survived"]].dropna()
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

x = df[["Sex", "Fare"]]
y = df["Survived"]

# Train model and save/dump
model = LinearRegression()
model.fit(x, y)
joblib.dump(model, 'TitanicModel.pkl')
