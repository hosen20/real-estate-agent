# 1. Standard library
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# 2. Third-party packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# Load dataset
df = pd.read_csv("housing.csv")
target = "SalePrice"

df = df[[
    "ExterQual",
    "BsmtQual",
    "HeatingQC",
    "KitchenQual",
    "Neighborhood",
    "Foundation",
    "BsmtFinType1",
    "GarageType",
    "OverallQual",
    "GrLivArea",
    "SalePrice"
]]

# Drop rows where target is NaN
df = df.dropna(subset=[target])

# Separate features from the target
X = df.drop(columns=[target])
y = df[target]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns

# Train/val/test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=50)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Ordinal columns with known quality ordering
ordinal_features = [
    'ExterQual', 'BsmtQual', 'HeatingQC', 'KitchenQual'
]
quality_order = [['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']] * len(ordinal_features)

ordinal_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
    ('encoder', OrdinalEncoder(
        categories=quality_order,
        handle_unknown='use_encoded_value',
        unknown_value=-1
    ))
])

nominal_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("ord", ordinal_transformer, ordinal_features),
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]), numeric_features)
    ]
)

# Candidate models
models = {
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

results = []
best_model_name = None
best_score = -np.inf
best_pipeline = None

# Train and evaluate each model on validation set
for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    pipeline.fit(X_train, y_train)
    val_preds = pipeline.predict(X_val)
    
    mae = mean_absolute_error(y_val, val_preds)
    mse = mean_squared_error(y_val, val_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, val_preds)
    
    results.append([name, mae, mse, rmse, r2])
    
    if r2 > best_score:
        best_score = r2
        best_model_name = name
        best_pipeline = pipeline

joblib.dump(best_pipeline, f"best_pipeline_{best_model_name}.joblib")