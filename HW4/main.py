import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("./HW4/course_lead_scoring_2026.csv")

cat = list(df.select_dtypes(include='object').columns)
num = [c for c in df.columns if c not in cat and c != 'converted']
df[cat] = df[cat].fillna('NA')
df[num] = df[num].fillna(0.0)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
y_train = df_train.converted.values
y_val = df_val.converted.values
df_full_train = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

# Q1
print("Q1 AUC per numeric feature (train):")
for c in ['lead_score', 'number_of_courses_viewed', 'interaction_count', 'annual_income']:
    auc = roc_auc_score(y_train, df_train[c])
    if auc < 0.5:
        auc = roc_auc_score(y_train, -df_train[c])
        print(c, "inverted ->", round(auc, 4))
    else:
        print(c, round(auc, 4))

features = cat + num

def train(d, y, C=1.0):
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(d[features].to_dict(orient='records'))
    m = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
    m.fit(X, y)
    return dv, m

def predict(dv, m, d):
    X = dv.transform(d[features].to_dict(orient='records'))
    return m.predict_proba(X)[:, 1]

# Q2
dv, m = train(df_train.drop(columns=['converted']), y_train)
y_pred = predict(dv, m, df_val.drop(columns=['converted']))
print("\nQ2 val AUC:", round(roc_auc_score(y_val, y_pred), 3), "(raw", roc_auc_score(y_val, y_pred), ")")

# Q3 / Q4
rows = []
for t in np.arange(0.0, 1.01, 0.01):
    t = round(float(t), 2)
    pred = y_pred >= t
    tp = ((pred == 1) & (y_val == 1)).sum()
    fp = ((pred == 1) & (y_val == 0)).sum()
    fn = ((pred == 0) & (y_val == 1)).sum()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    rows.append((t, p, r, f1))
sc = pd.DataFrame(rows, columns=['t', 'p', 'r', 'f1'])
valid = sc[(sc.p > 0) | (sc.r > 0)].copy()
valid['diff'] = (valid.p - valid.r).abs()
best = valid.sort_values(['diff', 't']).head(5)
print("\nQ3 closest precision/recall thresholds:\n", best)
print("\nQ4 max F1:\n", sc.sort_values('f1', ascending=False).head(5))

# Q5
kf = KFold(n_splits=5, shuffle=True, random_state=1)
scores = []
for tr, va in kf.split(df_full_train):
    d_tr = df_full_train.iloc[tr]
    d_va = df_full_train.iloc[va]
    dv, m = train(d_tr.drop(columns=['converted']), d_tr.converted.values)
    p = predict(dv, m, d_va.drop(columns=['converted']))
    scores.append(roc_auc_score(d_va.converted.values, p))
print("\nQ5 fold scores:", [round(s, 4) for s in scores])
print("Q5 std:", round(np.std(scores), 3), "(raw", np.std(scores), ")")

# Q6
print("\nQ6:")
for C in [0.000001, 0.001, 1]:
    scores = []
    for tr, va in kf.split(df_full_train):
        d_tr = df_full_train.iloc[tr]
        d_va = df_full_train.iloc[va]
        dv, m = train(d_tr.drop(columns=['converted']), d_tr.converted.values, C)
        p = predict(dv, m, d_va.drop(columns=['converted']))
        scores.append(roc_auc_score(d_va.converted.values, p))
    print(C, "mean", round(np.mean(scores), 3), "std", round(np.std(scores), 3))