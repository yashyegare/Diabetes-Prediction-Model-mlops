# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score  # <-- Added for evaluating the model
from dvclive import Live  # <-- Added for DVC experiment tracking
import joblib

# Load dataset from a working source (Kaggle/hosted)
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

print("✅ Columns:", df.columns.tolist())  # Debug print

# Prepare data
X = df[["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]]
y = df["Outcome"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Define Hyperparameters ---
# Defining them as variables makes it easy to log them and change them later
n_estimators = 100
max_depth = 5
random_seed = 42

# Train model
model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_seed)
model.fit(X_train, y_train)

# --- Evaluate Model ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# --- DVCLive Logging ---
with Live() as live:
    # Log the parameters we used
    live.log_param("n_estimators", n_estimators)
    live.log_param("max_depth", max_depth)
    live.log_param("random_state", random_seed)
    
    # Log the final accuracy score
    live.log_metric("accuracy", accuracy)

# Save
joblib.dump(model, "diabetes_model.pkl")
print(f"✅ Model saved as diabetes_model.pkl | Accuracy: {accuracy:.4f}")