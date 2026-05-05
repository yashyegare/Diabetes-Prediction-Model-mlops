# monitor.py
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

# 1. Load your training data (Reference)
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
reference_data = pd.read_csv(url)

# 2. Simulate new API data (Current)
# In a real scenario, you would pull this from your FastAPI logs or a database
current_data = reference_data.sample(n=100, random_state=1)
current_data['Glucose'] = current_data['Glucose'] * 1.1  # Simulate a 10% increase in glucose levels

# 3. Create the Data Drift Report
report = Report(metrics=[
    DataDriftPreset(),
    TargetDriftPreset()
])

report.run(reference_data=reference_data, current_data=current_data)

# 4. Save the report as an interactive HTML dashboard
report.save_html("drift_report.html")

print("✅ Monitoring report generated: drift_report.html")