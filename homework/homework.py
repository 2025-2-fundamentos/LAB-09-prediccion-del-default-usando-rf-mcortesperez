# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import os
import gzip
import json
import pickle
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

# ------------------------
# Load train/test raw data
# ------------------------
train_zip = "files/input/train_data.csv.zip"
test_zip = "files/input/test_data.csv.zip"

train = pd.read_csv(train_zip, compression="zip")
test = pd.read_csv(test_zip, compression="zip")

# ------------------------
# STEP 1 — CLEANING
# ------------------------

# Rename target
train = train.rename(columns={"default payment next month": "default"})
test = test.rename(columns={"default payment next month": "default"})

# Remove ID
if "ID" in train.columns:
    train = train.drop(columns=["ID"])
if "ID" in test.columns:
    test = test.drop(columns=["ID"])

# Drop NA
train = train.dropna()
test = test.dropna()

# EDUCATION > 4 → 4 ("others")
train.loc[train["EDUCATION"] > 4, "EDUCATION"] = 4
test.loc[test["EDUCATION"] > 4, "EDUCATION"] = 4

# ------------------------
# STEP 2 — SPLIT X/Y
# ------------------------
X_train = train.drop(columns=["default"])
y_train = train["default"]

X_test = test.drop(columns=["default"])
y_test = test["default"]

# ------------------------
# STEP 3 — PIPELINE
# ------------------------

categorical_cols = ["SEX", "EDUCATION", "MARRIAGE"]
numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

pipe = Pipeline(
    steps=[
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(random_state=42)),
    ]
)

# ------------------------
# STEP 4 — HYPERPARAM OPT
# ------------------------

param_grid = {
    "clf__n_estimators": [800],
    "clf__max_depth": [20, 26, None],
    "clf__min_samples_split": [2],
    "clf__min_samples_leaf": [1],
    "clf__max_features": ["sqrt"], 
    "clf__bootstrap": [True],
}

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    scoring="balanced_accuracy",
    n_jobs=-1,
)

grid.fit(X_train, y_train)

# ------------------------
# STEP 5 — SAVE MODEL
# ------------------------

os.makedirs("files/models", exist_ok=True)

with gzip.open("files/models/model.pkl.gz", "wb") as f:
    pickle.dump(grid, f)

# ------------------------
# STEP 6 — METRICS
# ------------------------

def compute_metrics(model, X, y, dataset_name):
    pred = model.predict(X)
    return {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": precision_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "recall": recall_score(y, pred),
        "f1_score": f1_score(y, pred),
    }, pred

m_train, pred_train = compute_metrics(grid, X_train, y_train, "train")
m_test, pred_test = compute_metrics(grid, X_test, y_test, "test")

# ------------------------
# STEP 7 — CONFUSION MATRICES
# ------------------------

def cm_dict(y_true, y_pred, dataset_name):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {"predicted_0": int(tn), "predicted_1": int(fp)},
        "true_1": {"predicted_0": int(fn), "predicted_1": int(tp)},
    }

cm_train = cm_dict(y_train, pred_train, "train")
cm_test = cm_dict(y_test, pred_test, "test")

# Save metrics
os.makedirs("files/output", exist_ok=True)

with open("files/output/metrics.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(m_train) + "\n")
    f.write(json.dumps(m_test) + "\n")
    f.write(json.dumps(cm_train) + "\n")
    f.write(json.dumps(cm_test) + "\n")