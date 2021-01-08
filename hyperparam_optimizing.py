import catboost as cb
from scipy.stats import randint as sp_randInt
from scipy.stats import uniform as sp_randFloat
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV
from skopt.space import Integer, Real

SCORING_LIST = ["accuracy", "roc_auc", "f1"]

XGBOOST_RANDOMSEARCH_PARAMS = {
    "silent": [False],
    "max_depth": sp_randInt(6, 20),
    "learning_rate": sp_randFloat(0.01, 0.3),
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bylevel": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
    "gamma": [0, 0.25, 0.5, 1.0],
    "reg_lambda": [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
    "n_estimators": [200],
}

XGBOOST_BAYESSEARCH_PARAMS = {
    "silent": [False],
    "max_depth": Integer(6, 20),
    "learning_rate": Real(0.01, 0.3),
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bylevel": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
    "gamma": [0, 0.25, 0.5, 1.0],
    "reg_lambda": [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
    "n_estimators": [200],
}

CATBOOST_RANDOMSEARCH_PARAMS = {
    "silent": [False],
    "learning_rate": sp_randFloat(0.01, 0.3),
    "depth": sp_randInt(6, 16),
    "l2_leaf_reg": [3, 1, 5, 10, 100],
    "loss_function": ["Logloss", "CrossEntropy"],
    "n_estimators": [200],
}

CATBOOST_BAYESSEARCH_PARAMS = {
    "silent": [False],
    "learning_rate": Real(0.01, 0.3),
    "depth": Integer(6, 16),
    "l2_leaf_reg": [3, 1, 5, 10, 100],
    "loss_function": ["Logloss", "CrossEntropy"],
    "n_estimators": [200],
}


def perform_random_search(
    estimator, X_train, X_val, y_train, y_val, param_grid, scoring=None
):
    if isinstance(estimator, cb.core.CatBoostClassifier):
        eval_set = (X_val, y_val)
    else:
        eval_set = [[X_val, y_val]]

    hyperparam_optimizer = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        scoring=scoring,
        cv=2,
        n_iter=20,
        n_jobs=1,
        refit=True,
        random_state=13,
    )
    if isinstance(estimator, cb.CatBoostClassifier):
        hyperparam_optimizer.fit(X_train, y_train, eval_set=eval_set)
    else:
        hyperparam_optimizer.fit(X_train, y_train, eval_set=eval_set)

    return hyperparam_optimizer.best_estimator_


def perform_bayes_search(
    estimator, X_train, X_val, y_train, y_val, param_grid, scoring=None
):
    if isinstance(estimator, cb.core.CatBoostClassifier):
        eval_set = (X_val, y_val)
    else:
        eval_set = [[X_val, y_val]]

    hyperparam_optimizer = BayesSearchCV(
        estimator=estimator,
        search_spaces=param_grid,
        scoring=scoring,
        cv=2,
        n_iter=20,
        n_jobs=1,
        refit=True,
        return_train_score=False,
        optimizer_kwargs={"base_estimator": "GP"},
        random_state=13,
        fit_params={
            "eval_set": eval_set,
        },
    )
    hyperparam_optimizer.fit(X_train, y_train)

    return hyperparam_optimizer.best_estimator_
