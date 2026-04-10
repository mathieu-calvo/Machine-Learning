# CI/CD for Machine Learning

## Why CI/CD for ML?

Traditional CI/CD tests code. ML CI/CD must also test **data quality**, **model performance**, and **prediction behavior**.

```
Code Change ──┐
Data Change ──┤──> CI Pipeline ──> CD Pipeline ──> Production
Config Change ─┘
```

## GitHub Actions ML Pipeline

### `.github/workflows/ml_pipeline.yml`

```yaml
name: ML Pipeline

on:
  push:
    branches: [main]
    paths:
      - 'src/**'
      - 'data/**'
      - 'configs/**'
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run unit tests
        run: pytest tests/unit/ -v

      - name: Run data validation
        run: python tests/validate_data.py

  train-and-evaluate:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Train model
        run: python src/train.py --config configs/production.yaml

      - name: Evaluate model
        run: python src/evaluate.py

      - name: Model performance gate
        run: |
          python -c "
          import json
          with open('metrics/scores.json') as f:
              metrics = json.load(f)
          assert metrics['accuracy'] > 0.90, f'Accuracy {metrics[\"accuracy\"]} below threshold 0.90'
          assert metrics['roc_auc'] > 0.95, f'ROC-AUC {metrics[\"roc_auc\"]} below threshold 0.95'
          print('All performance gates passed!')
          "

      - name: Upload model artifact
        if: github.ref == 'refs/heads/main'
        uses: actions/upload-artifact@v4
        with:
          name: trained-model
          path: models/

  deploy:
    needs: train-and-evaluate
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Download model artifact
        uses: actions/download-artifact@v4
        with:
          name: trained-model
          path: models/

      - name: Build Docker image
        run: docker build -t ml-model-api:${{ github.sha }} .

      - name: Push to registry
        run: |
          echo "Push to container registry (ECR/GCR/ACR)"
          # docker push ...
```

## Data Validation Tests

### `tests/validate_data.py`

```python
import pandas as pd
import great_expectations as gx

def validate_training_data():
    """Validate training data quality before model training."""
    df = pd.read_csv("data/processed/train.csv")

    # Schema checks
    assert set(df.columns) == {"feature_1", "feature_2", "target"}, "Schema mismatch"
    assert len(df) > 1000, f"Too few samples: {len(df)}"
    assert df["target"].nunique() >= 2, "Target must have at least 2 classes"

    # Quality checks
    null_pct = df.isnull().mean()
    assert (null_pct < 0.05).all(), f"Columns with >5% nulls: {null_pct[null_pct >= 0.05]}"

    # Distribution checks (detect data drift from training baseline)
    assert df["feature_1"].mean() > -10, "feature_1 mean suspiciously low"
    assert df["feature_1"].std() > 0, "feature_1 has zero variance"

    print("All data validations passed!")

if __name__ == "__main__":
    validate_training_data()
```

## Model Validation Tests

### `tests/test_model.py`

```python
import pytest
import numpy as np
import joblib

@pytest.fixture
def model():
    return joblib.load("models/model.pkl")

@pytest.fixture
def sample_input():
    return np.random.randn(1, 30)  # Match feature count

def test_model_loads(model):
    assert model is not None

def test_prediction_shape(model, sample_input):
    pred = model.predict(sample_input)
    assert pred.shape == (1,)

def test_prediction_range(model, sample_input):
    pred = model.predict(sample_input)
    assert pred[0] in [0, 1], "Binary classifier should output 0 or 1"

def test_probability_sum(model, sample_input):
    probs = model.predict_proba(sample_input)
    assert abs(probs.sum() - 1.0) < 1e-6, "Probabilities must sum to 1"

def test_deterministic(model, sample_input):
    pred1 = model.predict(sample_input)
    pred2 = model.predict(sample_input)
    assert np.array_equal(pred1, pred2), "Model should be deterministic"
```

## ML-Specific CI/CD Concerns

| Stage | Code CI/CD | ML CI/CD |
|-------|-----------|----------|
| **Build** | Compile, lint | + Data validation, feature checks |
| **Test** | Unit tests | + Model performance gates, bias checks |
| **Deploy** | Blue/green deploy | + Shadow mode, A/B testing, canary release |
| **Monitor** | Error rates, latency | + Data drift, prediction drift, feature skew |

## Deployment Strategies for ML

### Canary Deployment
```
Production traffic: 95% → existing model
                     5% → new model (canary)
Monitor for 24h, then gradually increase
```

### Shadow Mode
```
All traffic → existing model (serves predictions)
All traffic → new model (logs predictions, doesn't serve)
Compare predictions offline
```

### A/B Testing
```
50% traffic → Model A
50% traffic → Model B
Measure business metrics (conversion, revenue)
Choose winner after statistical significance reached
```

## Key Takeaways

1. **Automate everything** - training, evaluation, deployment should not require manual steps
2. **Performance gates prevent regression** - never deploy a model worse than the current one
3. **Test data quality in CI** - bad data is the #1 cause of model failures in production
4. **Shadow mode before full deployment** - validate on real traffic without risk
5. **Version everything together** - code + data + config + model = reproducible experiment
