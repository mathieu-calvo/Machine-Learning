# Model Serving with FastAPI + Docker

A production-ready REST API for serving ML model predictions.

## Architecture

```
Client ──> FastAPI ──> sklearn Pipeline ──> Response
              │
              ├── /health        (GET)  - Health check for load balancers
              ├── /predict       (POST) - Single prediction
              └── /predict/batch (POST) - Batch predictions
```

## Quick Start

### 1. Train and Save a Model

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_breast_cancer
import joblib

data = load_breast_cancer()
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", GradientBoostingClassifier(n_estimators=100)),
])
pipe.fit(data.data, data.target)
joblib.dump(pipe, "model.pkl")
```

### 2. Run Locally

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Run with Docker

```bash
docker build -t ml-api .
docker run -p 8000:8000 ml-api
```

### 4. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"instances": [[17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]]}'

# Interactive docs
open http://localhost:8000/docs
```

## Production Considerations

| Concern | Solution |
|---------|----------|
| **Scaling** | Run multiple replicas behind a load balancer (Kubernetes) |
| **Monitoring** | Add Prometheus metrics endpoint for latency, throughput, error rate |
| **Logging** | Structured JSON logs shipped to ELK/CloudWatch |
| **Model updates** | Blue/green deployment - run old and new model simultaneously |
| **Input validation** | Pydantic schemas reject malformed requests |
| **Authentication** | Add API key middleware or OAuth2 |
