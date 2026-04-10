# Feature Stores

## The Feature Store Problem

Without a feature store, feature engineering is duplicated, inconsistent, and error-prone:

```
Training Pipeline:  raw data → feature_eng_v1.py → training features
Serving Pipeline:   raw data → feature_eng_v2.py → serving features  ← DIFFERENT CODE!
```

**Training-serving skew** is one of the top causes of ML system failures.

## What is a Feature Store?

A feature store is a centralized system that:
1. **Stores** computed features in a consistent format
2. **Serves** features with low latency for real-time inference
3. **Ensures consistency** between training and serving
4. **Enables reuse** across teams and models

```
                    ┌─────────────────┐
Raw Data ──────────>│  Feature Store   │
                    │                  │
                    │  ┌─────────────┐ │
                    │  │ Offline     │ │──> Training Pipeline
                    │  │ Store       │ │    (batch reads)
                    │  │ (historical)│ │
                    │  └─────────────┘ │
                    │                  │
                    │  ┌─────────────┐ │
                    │  │ Online      │ │──> Serving Pipeline
                    │  │ Store       │ │    (low-latency reads)
                    │  │ (latest)    │ │
                    │  └─────────────┘ │
                    └─────────────────┘
```

## Feast: Open-Source Feature Store

[Feast](https://feast.dev/) is the most popular open-source feature store.

### Setup

```bash
pip install feast
feast init my_feature_repo
cd my_feature_repo
```

### Define Features

```python
# feature_repo/features.py
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64
from datetime import timedelta

# Entity (primary key)
customer = Entity(
    name="customer_id",
    join_keys=["customer_id"],
)

# Data source
customer_source = FileSource(
    path="data/customer_features.parquet",
    timestamp_field="event_timestamp",
)

# Feature view
customer_features = FeatureView(
    name="customer_features",
    entities=[customer],
    ttl=timedelta(days=1),
    schema=[
        Field(name="total_purchases", dtype=Int64),
        Field(name="avg_order_value", dtype=Float32),
        Field(name="days_since_last_order", dtype=Int64),
        Field(name="lifetime_value", dtype=Float32),
    ],
    source=customer_source,
)
```

### Use in Training

```python
from feast import FeatureStore
from datetime import datetime

store = FeatureStore(repo_path="feature_repo/")

# Get historical features for training (point-in-time correct)
entity_df = pd.DataFrame({
    "customer_id": [1001, 1002, 1003],
    "event_timestamp": [datetime(2024, 1, 15)] * 3,
})

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "customer_features:total_purchases",
        "customer_features:avg_order_value",
        "customer_features:days_since_last_order",
        "customer_features:lifetime_value",
    ],
).to_df()
```

### Use in Serving

```python
# Materialize latest features to online store
store.materialize_incremental(end_date=datetime.now())

# Get features for real-time inference (low latency)
online_features = store.get_online_features(
    features=[
        "customer_features:total_purchases",
        "customer_features:avg_order_value",
    ],
    entity_rows=[{"customer_id": 1001}],
).to_dict()
```

## Online vs Offline Store

| Aspect | Offline Store | Online Store |
|--------|--------------|-------------|
| **Use case** | Training, batch scoring | Real-time serving |
| **Latency** | Seconds to minutes | < 10ms |
| **Storage** | Data warehouse (BigQuery, Snowflake, S3+Parquet) | Key-value store (Redis, DynamoDB) |
| **Query pattern** | Historical point-in-time joins | Latest value by entity key |
| **Data volume** | Full history | Latest values only |

## Point-in-Time Correctness

The most critical feature store capability. When generating training data, features must reflect what was known **at the time of the event**, not what we know now.

```
Timeline:    Jan 1      Feb 1      Mar 1
Features:    v1         v2         v3
Event:                  ^
                        │
Correct feature = v1 (not v2 or v3!)
```

Without point-in-time correctness, you get **data leakage** - the model sees future information during training.

## When Do You Need a Feature Store?

| Signal | You Need a Feature Store |
|--------|------------------------|
| Multiple models share features | Yes - avoid duplication |
| Real-time serving required | Yes - need online store |
| Training-serving skew issues | Yes - single source of truth |
| Team > 3 ML engineers | Probably - collaboration benefits |
| Single model, batch only | Probably not - keep it simple |

## Alternatives to Feast

| Tool | Type | Best For |
|------|------|----------|
| **Feast** | Open-source | Small-medium teams, cloud-agnostic |
| **Tecton** | Managed | Enterprise, real-time features |
| **Databricks Feature Store** | Managed | Databricks-native teams |
| **Vertex AI Feature Store** | Managed | GCP-native teams |
| **SageMaker Feature Store** | Managed | AWS-native teams |
| **Hopsworks** | Open-source/Managed | Large-scale, streaming features |

## Key Takeaways

1. **Training-serving skew is a top ML failure mode** - feature stores prevent it
2. **Point-in-time correctness prevents data leakage** in training data generation
3. **Online stores enable real-time ML** with sub-10ms feature retrieval
4. **Start simple** - you may not need a feature store until you have multiple models sharing features
5. **Feature stores enable feature reuse** - compute once, use across models and teams
