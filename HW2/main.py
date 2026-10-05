import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./HW2/car_fuel_efficiency_2026.csv")

#EDA
print(df.describe())
print(df.dtypes)
print(df.tail(5))


#Q1
missing_values = df.columns[df.isnull().any()].tolist()
print(missing_values)

#Q2
hp_median = df['horsepower'].dropna().median()
print("Question 2:",hp_median)

n = len(df)
n_val = int(n*0.2)
n_test = int(n*0.2)
n_train = n - n_val - n_test

np.random.seed(42)
idx = np.arange(n)
np.random.shuffle(idx)

df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train+n_val]]
df_test = df.iloc[idx[n_train+n_val:]]

#Q3
