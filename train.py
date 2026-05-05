# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import dagshub

# 1. Connect to your DagsHub MLflow server
dagshub.init(repo_owner='yashyegare', repo_name='Diabetes-Prediction-Model-mlops', mlflow=True)

# Load dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

# Prepare data
X = df[["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]]
y = df["Outcome"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Hyperparameters
n_estimators = 100
max_depth = 5
random_seed = 42

# 2. Start an MLflow experiment run
with mlflow.start_run():
    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_seed)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # 3. Log metrics and params to MLflow
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_seed)
    mlflow.log_metric("accuracy", accuracy)

    # Save locally for DVC
    joblib.dump(model, "diabetes_model.pkl")
    
    # 4. Log the model artifact directly to MLflow
    mlflow.sklearn.log_model(model, "random_forest_model")

    print(f"✅ Model trained and logged to MLflow | Accuracy: {accuracy:.4f}")