# IEEE Fraud Detection with Anomaly Detection Algorithms
[IEEE Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) is a competition that you are predicting the probability that an online transaction is fraudulent, as denoted by the binary target isFraud. The data is broken into two files identity and transaction, which are joined by TransactionID. Not all transactions have corresponding identity information.

This repo provides training hyperparameter optimization and evaluation scripts for [IEEE Fraud Detection data](https://www.kaggle.com/c/ieee-fraud-detection/data) with [XGBoost](https://github.com/dmlc/xgboost) and [CatBoost](https://github.com/catboost/catboost) algorithms.

Inspect [CatBoost notebook](<https://github.com/fcakyon/ieee-fraud-detection/tree/main/notebook/fraud-detection-catboost.ipynb>) and [XGBoost notebook](<https://github.com/fcakyon/ieee-fraud-detection/tree/main/notebook/fraud-detection-xgboost.ipynb>) for detailed demo.

## Usage
- Clone:
```bash
git clone https://github.com/fcakyon/ieee-fraud-detection.git
```

- Prepare conda environment:
```bash
conda env create -f environment.yml
```

```bash
conda activate frauddetection
```

- Fir XGBoost or CatBoost model:
```bash
python train_xgboost.py
```
```bash
python train_catboost.py
```

## Detailed Usage
- After downloading the data to ./data folder, you can get the preprocessed data via get_data() method in preprocessing.py module:
```python
from preprocessing import get_data

float: val_split = 0.2 # splits %80 of data for train and %20 for val
bool: apply_label_encoding = True # applies label encoding to categorical features
bool: fillna = True # fills missing values with -999

data = get_data(val_split, apply_label_encoding, fillna)
```

- After initializing CatBoost or XGBoost classifier, perform automatic hyperparameter optimization via perform_random_search() or perform_bayes_search() in hyperparam_optimizing.py module:
```python
from hyperparam_optimizing import perform_bayes_search, CATBOOST_BAYESSEARCH_PARAMS
import catboost as cb

# define xgboost or catboost instance
estimator = cb.CatBoostClassifier(
    n_estimators=200,
    learning_rate=0.05,
    metric_period=500,
    od_wait=500,
    task_type="CPU",
    depth=8,
) 

# parse get_data() output
X_train = data["X_train"]
X_val = data["X_val"]
y_train = data["y_train"]
y_val = data["y_val"]

# define parameter grid
param_grid = CATBOOST_BAYESSEARCH_PARAMS

# define scoring metric, full list can be seen at https://scikit-learn.org/stable/modules/model_evaluation.html
scoring = 'roc_auc'

# perform bayes parameter search for catboost classifier
best_estimator = perform_bayes_search(
    estimator, X_train, X_val, y_train, y_val, param_grid, scoring
)
```

- After fitting a model, print results such as accuracy, auc score, confusion matrix, f1 scores using calculate_scores() method from scoring.py module:
```python
from scoring import calculate_scores

scores = calculate_scores(estimator=best_estimator, X_val=X_val, y_val=y_val)
```