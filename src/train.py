
# Customer Churn Prediction
# Based on the original Google Colab notebook

import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype(str).str.strip(),
    errors="coerce"
)

# 2. Separate features and target
X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Identify numerical and categorical columns
num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X_train.select_dtypes(include=["object"]).columns

# 5. Preprocess numerical features
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 6. Preprocess categorical features
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# 7. Create Logistic Regression pipeline
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

# 8. Train original model
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("Original Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nOriginal Classification Report:")
print(classification_report(y_test, y_pred))
print("\nOriginal Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 9. Hyperparameter tuning
param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],
    "classifier__solver": ["liblinear", "lbfgs"]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# 10. Evaluate tuned model
best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)

print("\nBest Parameters:", grid_search.best_params_)
print("Best CV F1 Score:", grid_search.best_score_)
print("Tuned Model Accuracy:", accuracy_score(y_test, y_pred_tuned))
print("\nTuned Classification Report:")
print(classification_report(y_test, y_pred_tuned))
print("\nTuned Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tuned))

# 11. Save the best complete pipeline
import joblib

joblib.dump(best_model, "churn_model.joblib")
print("\nModel saved as churn_model.joblib")
