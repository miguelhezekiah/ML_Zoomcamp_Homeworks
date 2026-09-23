import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('./HW1/car_fuel_efficiency_2026.csv')

#Q1
record_count = data.shape[0]
print("1.",pd.__version__)

#Q2
print("2.",data.shape[0])

#Q3
fuel_type_count = data['fuel_type'].nunique(dropna=False)
print("3.",fuel_type_count)

#Q4
missing_value_count = (data.isna().sum() > 0).sum()
print("4.",missing_value_count)

#Q5
max_fuel_efficiency = data['fuel_efficiency_mpg'].max()
print("5.",max_fuel_efficiency)

#Q6
horsepower_median = data['horsepower'].dropna().median()
horsepower_most_frequent = data['horsepower'].value_counts().idxmax()
horsepower_median_two = data['horsepower'].fillna(horsepower_most_frequent).median()
print("6.",horsepower_median_two - horsepower_median)

#Q7
asian_cars = data[data['origin'] == "Asia"]
asian_cars = asian_cars[['vehicle_weight', 'model_year']].head(7)
X = asian_cars.to_numpy()
X_transpose = np.matrix_transpose(X)
XTX = np.matmul(X_transpose, X)
XTX_invert = np.linalg.inv(XTX)
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
w = sum(np.matmul(np.matmul(XTX_invert, X_transpose), y))
print("7.", w)
