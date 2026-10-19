import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score, accuracy_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("./HW3/course_lead_scoring_2026.csv")
print("shape", df.shape)
print("missing:\n", df.isnull().sum())

cat = list(df.select_dtypes(include='object').columns)
num = [c for c in df.columns if c not in cat and c != 'converted']
print("cat", cat)
print("num", num)

df[cat] = df[cat].fillna('NA')
df[num] = df[num].fillna(0.0)

# Q1
print("\nQ1 mode industry:", df['industry'].mode()[0])
print(df['industry'].value_counts())

# Q2
corr = df[num].corr()
print("\nQ2 corr matrix:\n", corr)
pairs = [('interaction_count', 'lead_score'),
         ('number_of_courses_viewed', 'lead_score'),
         ('number_of_courses_viewed', 'interaction_count'),
         ('annual_income', 'interaction_count')]
for a, b in pairs:
    print(a, b, round(corr.loc[a, b], 4))

# split
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)
y_train = df_train.converted.values
y_val = df_val.converted.values
df_train = df_train.drop(columns=['converted']).reset_index(drop=True)
df_val = df_val.drop(columns=['converted']).reset_index(drop=True)
print("\nsizes", len(df_train), len(df_val), len(df_test))

# Q3
print("\nQ3 mutual info (rounded 2):")
for c in cat:
    s = mutual_info_score(df_train[c], y_train)
    print(c, round(s, 2), "(raw", round(s, 5), ")")

# Q4
features = cat + num
def fit_eval(feats, C=1.0):
    dv = DictVectorizer(sparse=False)
    Xtr = dv.fit_transform(df_train[feats].to_dict(orient='records'))
    Xva = dv.transform(df_val[feats].to_dict(orient='records'))
    m = LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=42)
    m.fit(Xtr, y_train)
    return accuracy_score(y_val, m.predict(Xva))

base = fit_eval(features)
print("\nQ4 accuracy:", round(base, 2), "(raw", base, ")")

# Q5
print("\nQ5 feature elimination:")
for f in ['lead_source', 'number_of_courses_viewed', 'interaction_count']:
    acc = fit_eval([x for x in features if x != f])
    print(f, "acc", acc, "diff", base - acc)

# Q6
print("\nQ6 regularization:")
for C in [0.000001, 0.00001, 0.0001, 0.001]:
    acc = fit_eval(features, C)
    print(C, round(acc, 3), "(raw", acc, ")")