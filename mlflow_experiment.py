# -----------------------------------------------------------
# Step 4: Machine Learning Model Development + MLflow Tracking
# -----------------------------------------------------------

import pandas as pd
import numpy as np
import joblib
import warnings
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
warnings.filterwarnings("ignore")

# -----------------------------------------------------------
# 1. Load Dataset
# -----------------------------------------------------------
csv_file = r"C:\Users\Alagu\EMI_Project\Cleaned_data\Feature_Engineering_Outputs\emi_dataset_feature_engineered.csv"
df = pd.read_csv(csv_file)

print("✅ Dataset Loaded Successfully!")
print(f"Shape: {df.shape}")
print("Columns:", list(df.columns))

# -----------------------------------------------------------
# 2. Identify Target Columns
# -----------------------------------------------------------
classification_targets = ['emi_eligible', 'EMI_Eligible', 'emi_eligibility', 'EMI_Eligibility', 'emi_eligibility_Not_Eligible']
regression_target = 'max_monthly_emi'

target_class = None
for col in df.columns:
    if col in classification_targets:
        target_class = col
        break

# -----------------------------------------------------------
# 3. Set up MLflow
# -----------------------------------------------------------
print("📁 Current tracking URI:", mlflow.get_tracking_uri())
print("🧪 Current experiment:", mlflow.get_experiment_by_name("EMI_Prediction_Experiment_01"))

mlflow.set_tracking_uri("file:///C:/Users/Alagu/EMI_Project/mlruns")  # local tracking directory
mlflow.set_experiment("EMI_Prediction_Experiment_01")

# ===========================================================
# 🧠 CLASSIFICATION MODEL: EMI ELIGIBILITY
# ===========================================================
if target_class:
    print(f"\n🎯 Classification Target Found: {target_class}")

    X = df.drop(columns=[target_class])
    y = df[target_class]

    # Encode target if categorical
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(0)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define models
    models = {
        "Logistic Regression (L1)": LogisticRegression(max_iter=1000, penalty='l1', solver='liblinear'),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    print("\n⚙️ Training Classification Models...")

    for name, model in models.items():
        with mlflow.start_run(run_name=f"Classification_{name}"):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan

            # Log parameters and metrics
            mlflow.log_param("Model_Type", name)
            mlflow.log_metric("Accuracy", acc)
            mlflow.log_metric("Precision", prec)
            mlflow.log_metric("Recall", rec)
            mlflow.log_metric("F1_Score", f1)
            mlflow.log_metric("ROC_AUC", roc)

            # Log model
            mlflow.sklearn.log_model(model, name="classification_model")

            print(f"\n🔹 {name}:")
            print(f"Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, ROC-AUC={roc:.4f}")
            print(f"📊 Logged to MLflow run: {mlflow.active_run().info.run_id}")

    print("\n✅ Classification Models Logged to MLflow Successfully.")

else:
    print("\n⚠️ No Classification Target Found. Skipping Classification Section.")


# ===========================================================
# 💰 REGRESSION MODEL: MAXIMUM EMI PREDICTION
# ===========================================================
if regression_target not in df.columns:
    raise ValueError(f"❌ Target column '{regression_target}' not found!")

print(f"\n🎯 Regression Target Found: {regression_target}")

X = df.drop(columns=[regression_target])
y = df[regression_target]

X = pd.get_dummies(X, drop_first=True)
X.fillna(X.mean(numeric_only=True), inplace=True)
y.fillna(y.mean(), inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

reg_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(random_state=42),
    "XGBoost Regressor": XGBRegressor(random_state=42, objective='reg:squarederror')
}

print("\n⚙️ Training Regression Models...")

for name, model in reg_models.items():
    with mlflow.start_run(run_name=f"Regression_{name}"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log parameters and metrics
        mlflow.log_param("Model_Type", name)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

        # Log model
        mlflow.sklearn.log_model(model, name="regression_model")

        print(f"\n🔹 {name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        print(f"📊 Logged to MLflow run: {mlflow.active_run().info.run_id}")

print("\n✅ Regression Models Logged to MLflow Successfully.")
