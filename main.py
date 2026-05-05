import sys
import os

venv_path = os.path.join(os.getcwd(), ".mlops", "Lib", "site-packages")
if venv_path not in sys.path:
    sys.path.insert(0, venv_path)

from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import numpy as np
import time

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge

app = FastAPI()

# ── Prometheus metrics ────────────────────────────────────────────────────────

# Prediction outcome counters
DIABETIC_COUNT = Counter(
    "diabetes_prediction_diabetic_total",
    "Total number of diabetic predictions"
)
NOT_DIABETIC_COUNT = Counter(
    "diabetes_prediction_not_diabetic_total",
    "Total number of non-diabetic predictions"
)

# Input feature histograms — lets you spot data drift over time
GLUCOSE_HIST = Histogram(
    "input_glucose",
    "Distribution of incoming Glucose values",
    buckets=[70, 90, 110, 130, 150, 170, 199, float("inf")]
)
BMI_HIST = Histogram(
    "input_bmi",
    "Distribution of incoming BMI values",
    buckets=[18.5, 25, 30, 35, 40, 45, float("inf")]
)
AGE_HIST = Histogram(
    "input_age",
    "Distribution of incoming Age values",
    buckets=[20, 30, 40, 50, 60, 70, float("inf")]
)
PREGNANCIES_HIST = Histogram(
    "input_pregnancies",
    "Distribution of incoming Pregnancies values",
    buckets=[0, 1, 2, 4, 6, 10, float("inf")]
)
BLOOD_PRESSURE_HIST = Histogram(
    "input_blood_pressure",
    "Distribution of incoming BloodPressure values",
    buckets=[60, 70, 80, 90, 100, float("inf")]
)

# Inference latency
INFERENCE_LATENCY = Histogram(
    "model_inference_latency_seconds",
    "Time taken to run model.predict()",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
)

# Live gauge — diabetic prediction ratio (last 1 = all diabetic, 0 = none)
DIABETIC_RATIO = Gauge(
    "diabetes_prediction_ratio",
    "Rolling ratio of diabetic predictions (updated each request)"
)

# Internal state for ratio calculation
_total = 0
_diabetic = 0

# ── Auto-instrument all FastAPI routes (request count, latency, status codes) ─
Instrumentator().instrument(app).expose(app)   # exposes /metrics endpoint

# ── Model load ────────────────────────────────────────────────────────────────
model = joblib.load("diabetes_model.pkl")


# ── Schemas ───────────────────────────────────────────────────────────────────
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    BMI: float
    Age: int


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def read_root():
    return {"message": "Diabetes Prediction API is live"}


@app.post("/predict")
def predict(data: DiabetesInput):
    global _total, _diabetic

    # Record input feature distributions
    GLUCOSE_HIST.observe(data.Glucose)
    BMI_HIST.observe(data.BMI)
    AGE_HIST.observe(data.Age)
    PREGNANCIES_HIST.observe(data.Pregnancies)
    BLOOD_PRESSURE_HIST.observe(data.BloodPressure)

    # Run inference and track latency
    input_array = np.array([[
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.BMI,
        data.Age
    ]])

    start = time.perf_counter()
    prediction = model.predict(input_array)[0]
    INFERENCE_LATENCY.observe(time.perf_counter() - start)

    # Track prediction outcomes
    result = bool(prediction)
    _total += 1
    if result:
        DIABETIC_COUNT.inc()
        _diabetic += 1
    else:
        NOT_DIABETIC_COUNT.inc()

    DIABETIC_RATIO.set(_diabetic / _total)

    return {"diabetic": result}