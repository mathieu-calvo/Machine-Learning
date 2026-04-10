# Data Versioning with DVC

## Why Data Versioning?

Git tracks code, but ML projects also depend on data and model artifacts that are too large for Git. [DVC (Data Version Control)](https://dvc.org/) bridges this gap.

**Problems DVC solves:**
- Datasets too large for Git (>100MB)
- Need to reproduce results with the exact data used for training
- Track which data version produced which model
- Share datasets across a team

## Setup

```bash
pip install dvc dvc-s3  # or dvc-gs, dvc-azure for other cloud providers
cd your-ml-project
dvc init
```

## Core Workflow

### 1. Track a Large Dataset

```bash
# Add a data file to DVC tracking
dvc add data/train.csv

# This creates:
# - data/train.csv.dvc  (small metadata file - commit this to Git)
# - .gitignore entry for data/train.csv (the actual data is NOT in Git)

git add data/train.csv.dvc data/.gitignore
git commit -m "Track training data v1"
```

### 2. Store Data Remotely

```bash
# Configure remote storage (S3 example)
dvc remote add -d myremote s3://my-bucket/dvc-storage

# Push data to remote
dvc push

# On another machine, pull the data
dvc pull
```

### 3. Version Data with Git Tags

```bash
# Version 1 of training data
git tag -a "data-v1" -m "Initial training dataset"

# Update the dataset
# ... modify data/train.csv ...
dvc add data/train.csv
git add data/train.csv.dvc
git commit -m "Update training data v2"
git tag -a "data-v2" -m "Added 10k new samples"

# Switch between data versions
git checkout data-v1
dvc checkout  # Downloads the v1 data
```

## DVC Pipelines

DVC can define reproducible ML pipelines with `dvc.yaml`:

```yaml
# dvc.yaml
stages:
  preprocess:
    cmd: python src/preprocess.py
    deps:
      - data/raw/
      - src/preprocess.py
    outs:
      - data/processed/

  train:
    cmd: python src/train.py
    deps:
      - data/processed/
      - src/train.py
    params:
      - train.n_estimators
      - train.learning_rate
    outs:
      - models/model.pkl
    metrics:
      - metrics/scores.json:
          cache: false

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - models/model.pkl
      - data/processed/test.csv
      - src/evaluate.py
    metrics:
      - metrics/eval.json:
          cache: false
    plots:
      - metrics/confusion_matrix.csv
```

```bash
# Run the full pipeline
dvc repro

# Only re-runs stages whose dependencies changed
# Compare metrics across experiments
dvc metrics diff
dvc plots diff
```

## DVC + Git: The Complete Picture

```
Git tracks:          DVC tracks:
├── src/             ├── data/train.csv (→ S3)
├── dvc.yaml         ├── data/test.csv (→ S3)
├── params.yaml      ├── models/model.pkl (→ S3)
├── *.dvc files      └── (large artifacts)
└── metrics/
```

## Best Practices

| Practice | Why |
|----------|-----|
| **One `.dvc` file per logical dataset** | Easier to version and track changes |
| **Use DVC pipelines for E2E** | Ensures reproducibility from raw data to model |
| **Store remote on cloud** | Team-wide access, backup, CI/CD integration |
| **Tag data versions** | Easy rollback and experiment reproduction |
| **Track params.yaml** | DVC compares experiments by parameter changes |

## Key Commands

```bash
dvc init              # Initialize DVC in a Git repo
dvc add <file>        # Start tracking a file with DVC
dvc push              # Upload tracked files to remote storage
dvc pull              # Download tracked files from remote
dvc repro             # Reproduce the pipeline (re-run changed stages)
dvc metrics diff      # Compare metrics between commits/branches
dvc plots diff        # Generate comparison plots
dvc gc                # Garbage collect unused cache
```
