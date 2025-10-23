import os
import pandas as pd

CANDIDATES = ["medical_diagnosis_data.csv", os.path.join("data","medical_diagnosis_data.csv")]
for p in CANDIDATES:
    if os.path.exists(p):
        DATA_PATH = p
        break
else:
    raise FileNotFoundError("Place medical_diagnosis_data.csv next to this script or in ./data")

df = pd.read_csv(DATA_PATH)

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer

binary_cols = ['smoker', 'family_history', 'gender']
nominal_cols = ['blood_pressure', 'exercise_freq']
numeric_cols = ['age', 'bmi', 'cholesterol', 'stress_level', 'sleep_hours']

for c in binary_cols + ['has_disease']:
    if df[c].dtype == object:
        df[c] = df[c].astype(str).str.strip().replace({'Yes':1, 'No':0, '1':1, '0':0})
        df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.dropna(subset=['has_disease']).copy()

numeric_linear = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
pre_linear = ColumnTransformer([("num", numeric_linear, numeric_cols), ("cat", categorical, nominal_cols), ("bin", "passthrough", binary_cols)])

numeric_tree = Pipeline([("imputer", SimpleImputer(strategy="median"))])
pre_tree = ColumnTransformer([("num", numeric_tree, numeric_cols), ("cat", categorical, nominal_cols), ("bin", "passthrough", binary_cols)])

X = df[binary_cols + nominal_cols + numeric_cols]
y = df['has_disease'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

from sklearn.tree import DecisionTreeClassifier
pipe = Pipeline([("preprocess", pre_tree), ("clf", DecisionTreeClassifier(random_state=42))])
from sklearn.model_selection import GridSearchCV
param_grid = {"clf__max_depth":[5,None], "clf__min_samples_leaf":[1,5]}
gs = GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=3, n_jobs=-1)
gs.fit(X_train, y_train)
best = gs.best_estimator_
y_pred = best.predict(X_test); y_proba = best.predict_proba(X_test)[:,1]
print("Best Params:", gs.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1:", f1_score(y_test, y_pred, zero_division=0))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
