# Machine Learning: From Foundations to Production

A comprehensive, hands-on educational repository that walks through all major machine learning concepts, model families, and production best practices. Every section includes runnable end-to-end examples with real datasets.

## Who This Is For

- Data scientists looking for a structured reference across ML topics
- Engineers transitioning into ML who want practical, runnable examples
- Students who want to go beyond theory with state-of-the-art implementations
- Anyone preparing for ML interviews or building a portfolio

## Prerequisites

- Python 3.10+
- Familiarity with NumPy and Pandas
- Basic linear algebra and statistics

## Setup

```bash
git clone https://github.com/mathieu-calvo/Machine-Learning.git
cd Machine-Learning
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Repository Structure

```
Machine-Learning/
├── 01_foundations/
│   ├── 01_data_exploration_and_preprocessing.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_evaluation_metrics_and_validation.ipynb
├── 02_supervised_learning/
│   ├── 01_linear_regression.ipynb
│   ├── 02_logistic_regression.ipynb
│   ├── 03_decision_trees_and_random_forests.ipynb
│   ├── 04_gradient_boosting.ipynb
│   ├── 05_support_vector_machines.ipynb
│   ├── 06_knn.ipynb
│   └── 07_naive_bayes.ipynb
├── 03_unsupervised_learning/
│   ├── 01_kmeans_clustering.ipynb
│   ├── 02_hierarchical_clustering.ipynb
│   ├── 03_dbscan.ipynb
│   ├── 04_pca_dimensionality_reduction.ipynb
│   └── 05_anomaly_detection.ipynb
├── 04_deep_learning/
│   ├── 01_feedforward_neural_networks.ipynb
│   ├── 02_convolutional_neural_networks.ipynb
│   ├── 03_recurrent_neural_networks.ipynb
│   ├── 04_transformers_and_attention.ipynb
│   └── 05_transfer_learning.ipynb
├── 05_nlp/
│   ├── 01_text_preprocessing_and_embeddings.ipynb
│   └── 02_sentiment_analysis_with_transformers.ipynb
├── 06_time_series/
│   ├── 01_classical_forecasting.ipynb
│   └── 02_deep_learning_forecasting.ipynb
├── 07_reinforcement_learning/
│   └── 01_q_learning_intro.ipynb
├── 08_mlops/
│   ├── 01_experiment_tracking_mlflow.ipynb
│   ├── 02_model_registry_and_versioning.ipynb
│   ├── 03_data_versioning_dvc.md
│   ├── 05_ci_cd_for_ml.md
│   ├── 06_monitoring_and_drift_detection.ipynb
│   ├── 07_feature_stores.md
│   └── model_serving/
│       ├── app.py
│       ├── Dockerfile
│       ├── requirements.txt
│       └── README.md
├── data/
│   └── README.md
├── requirements.txt
├── LICENSE
├── README.md
└── .gitignore
```

---

## Learning Path

### 1. Foundations (`01_foundations/`)

Build the core skills that every ML workflow depends on.

| Notebook | Topics |
|----------|--------|
| [01 - Data Exploration & Preprocessing](01_foundations/01_data_exploration_and_preprocessing.ipynb) | EDA, missing values, encoding, scaling, train/test splits |
| [02 - Feature Engineering](01_foundations/02_feature_engineering.ipynb) | Feature creation, selection, polynomial features, target encoding |
| [03 - Evaluation Metrics & Validation](01_foundations/03_evaluation_metrics_and_validation.ipynb) | Cross-validation, stratified splits, classification & regression metrics, ROC/AUC, bias-variance tradeoff |

### 2. Supervised Learning (`02_supervised_learning/`)

End-to-end examples for every major supervised model family.

| Notebook | Model Family | Task |
|----------|-------------|------|
| [01 - Linear Regression](02_supervised_learning/01_linear_regression.ipynb) | Linear Models | Regression |
| [02 - Logistic Regression](02_supervised_learning/02_logistic_regression.ipynb) | Linear Models | Classification |
| [03 - Decision Trees & Random Forests](02_supervised_learning/03_decision_trees_and_random_forests.ipynb) | Tree-Based | Classification & Regression |
| [04 - Gradient Boosting](02_supervised_learning/04_gradient_boosting.ipynb) | Ensemble (XGBoost, LightGBM, CatBoost) | Classification |
| [05 - Support Vector Machines](02_supervised_learning/05_support_vector_machines.ipynb) | Kernel Methods | Classification |
| [06 - K-Nearest Neighbors](02_supervised_learning/06_knn.ipynb) | Instance-Based | Classification |
| [07 - Naive Bayes](02_supervised_learning/07_naive_bayes.ipynb) | Probabilistic | Text Classification |

### 3. Unsupervised Learning (`03_unsupervised_learning/`)

| Notebook | Topics |
|----------|--------|
| [01 - K-Means Clustering](03_unsupervised_learning/01_kmeans_clustering.ipynb) | K-Means, elbow method, silhouette analysis |
| [02 - Hierarchical Clustering](03_unsupervised_learning/02_hierarchical_clustering.ipynb) | Agglomerative clustering, dendrograms, linkage methods |
| [03 - DBSCAN](03_unsupervised_learning/03_dbscan.ipynb) | Density-based clustering, epsilon tuning, noise handling |
| [04 - PCA & Dimensionality Reduction](03_unsupervised_learning/04_pca_dimensionality_reduction.ipynb) | PCA, explained variance, t-SNE, UMAP |
| [05 - Anomaly Detection](03_unsupervised_learning/05_anomaly_detection.ipynb) | Isolation Forest, Local Outlier Factor, One-Class SVM |

### 4. Deep Learning (`04_deep_learning/`)

All notebooks use **PyTorch** with modern best practices.

| Notebook | Architecture | Task |
|----------|-------------|------|
| [01 - Feedforward Neural Networks](04_deep_learning/01_feedforward_neural_networks.ipynb) | MLP | Tabular classification |
| [02 - Convolutional Neural Networks](04_deep_learning/02_convolutional_neural_networks.ipynb) | CNN | Image classification (CIFAR-10) |
| [03 - Recurrent Neural Networks](04_deep_learning/03_recurrent_neural_networks.ipynb) | LSTM/GRU | Sequence modeling |
| [04 - Transformers & Attention](04_deep_learning/04_transformers_and_attention.ipynb) | Transformer | Sequence classification |
| [05 - Transfer Learning](04_deep_learning/05_transfer_learning.ipynb) | ResNet fine-tuning | Image classification |

### 5. Natural Language Processing (`05_nlp/`)

| Notebook | Topics |
|----------|--------|
| [01 - Text Preprocessing & Embeddings](05_nlp/01_text_preprocessing_and_embeddings.ipynb) | Tokenization, TF-IDF, Word2Vec, sentence embeddings |
| [02 - Sentiment Analysis with Transformers](05_nlp/02_sentiment_analysis_with_transformers.ipynb) | Fine-tuning HuggingFace models for classification |

### 6. Time Series (`06_time_series/`)

| Notebook | Topics |
|----------|--------|
| [01 - Classical Forecasting](06_time_series/01_classical_forecasting.ipynb) | ARIMA, seasonal decomposition, Prophet |
| [02 - Deep Learning Forecasting](06_time_series/02_deep_learning_forecasting.ipynb) | LSTM-based time series prediction |

### 7. Reinforcement Learning (`07_reinforcement_learning/`)

| Notebook | Topics |
|----------|--------|
| [01 - Q-Learning Introduction](07_reinforcement_learning/01_q_learning_intro.ipynb) | Q-tables, exploration vs exploitation, Gymnasium environments |

### 8. MLOps - Productionizing ML (`08_mlops/`)

Take models from notebooks to production with state-of-the-art tooling.

| Resource | Topics |
|----------|--------|
| [01 - Experiment Tracking with MLflow](08_mlops/01_experiment_tracking_mlflow.ipynb) | Logging params/metrics/artifacts, comparing runs, MLflow UI |
| [02 - Model Registry & Versioning](08_mlops/02_model_registry_and_versioning.ipynb) | MLflow Model Registry, model stages, reproducibility |
| [03 - Data Versioning with DVC](08_mlops/03_data_versioning_dvc.md) | DVC pipelines, remote storage, data lineage |
| [04 - Model Serving with FastAPI](08_mlops/model_serving/) | REST API, Docker, health checks, batch prediction |
| [05 - CI/CD for ML](08_mlops/05_ci_cd_for_ml.md) | GitHub Actions, automated testing, model validation gates |
| [06 - Monitoring & Drift Detection](08_mlops/06_monitoring_and_drift_detection.ipynb) | Data drift, model drift, Evidently AI dashboards |
| [07 - Feature Stores](08_mlops/07_feature_stores.md) | Feature engineering at scale, online/offline stores, Feast |

---

## Key Concepts Covered

| Concept | Where |
|---------|-------|
| Bias-Variance Tradeoff | Foundations 03, Supervised 01 |
| Regularization (L1/L2) | Supervised 01-02, Deep Learning 01 |
| Cross-Validation | Foundations 03 |
| Hyperparameter Tuning | Supervised 04 (Optuna) |
| Ensemble Methods | Supervised 03-04 |
| Backpropagation | Deep Learning 01 |
| Attention Mechanism | Deep Learning 04 |
| Transfer Learning | Deep Learning 05, NLP 02 |
| Experiment Tracking | MLOps 01-02 |
| Model Deployment | MLOps 04 |
| Monitoring in Production | MLOps 06 |

## References

- Bishop, C. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Murphy, K. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.
- Huyen, C. (2022). *Designing Machine Learning Systems*. O'Reilly.
- Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NeurIPS.

## License

MIT License. See [LICENSE](LICENSE) for details.
