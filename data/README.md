# Datasets

Most notebooks in this repository use built-in datasets from sklearn, torchvision, or HuggingFace that are automatically downloaded on first run. No manual data download is needed.

## Datasets Used

| Notebook | Dataset | Source | Auto-download |
|----------|---------|--------|:---:|
| Foundations 01 | Titanic | `seaborn.load_dataset("titanic")` | Yes |
| Foundations 02 | Ames Housing | `sklearn.datasets.fetch_openml("house_prices")` | Yes |
| Foundations 03 | Breast Cancer, California Housing | `sklearn.datasets` | Yes |
| Supervised 01 | California Housing | `sklearn.datasets` | Yes |
| Supervised 02 | Breast Cancer, Iris | `sklearn.datasets` | Yes |
| Supervised 03 | Wine | `sklearn.datasets` | Yes |
| Supervised 04 | Breast Cancer | `sklearn.datasets` | Yes |
| Supervised 05 | Digits | `sklearn.datasets` | Yes |
| Supervised 06 | Iris | `sklearn.datasets` | Yes |
| Supervised 07 | 20 Newsgroups | `sklearn.datasets` | Yes |
| Unsupervised 01-03 | Synthetic + Iris | `sklearn.datasets` | Yes |
| Unsupervised 04 | Digits | `sklearn.datasets` | Yes |
| Deep Learning 01 | MNIST | `torchvision.datasets` | Yes |
| Deep Learning 02 | CIFAR-10 | `torchvision.datasets` | Yes |
| Deep Learning 03 | 20 Newsgroups | `sklearn.datasets` | Yes |
| NLP 02 | IMDB | `datasets.load_dataset("imdb")` | Yes |
| Time Series | Airline Passengers | CSV from URL | Yes |
| RL | FrozenLake | `gymnasium` | Yes |

## Large Datasets

If you want to work with larger datasets, place them in this directory. The `.gitignore` is configured to ignore CSV and Parquet files here (except files prefixed with `sample_`).
