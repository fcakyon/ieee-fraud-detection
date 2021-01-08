import xgboost as xgb

from hyperparam_optimizing import (
    SCORING_LIST,
    XGBOOST_BAYESSEARCH_PARAMS,
    XGBOOST_RANDOMSEARCH_PARAMS,
    perform_bayes_search,
    perform_random_search,
)
from preprocessing import get_data
from scoring import calculate_scores

VAL_SPLIT = 0.2
data = get_data(val_split=VAL_SPLIT, apply_label_encoding=True, fillna=True)
X_train, X_val, X_test, y_train, y_val = (
    data["X_train"],
    data["X_val"],
    data["X_test"],
    data["y_train"],
    data["y_val"],
)
clf = xgb.XGBClassifier(
    n_estimators=200,
    n_jobs=4,
    max_depth=9,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="gpu_hist",
    missing=-999,
    use_label_encoder=False,
)

print("Fitting a xgboost model...")
clf.fit(X_train, y_train)

_ = calculate_scores(clf, X_val, y_val)

for scoring in SCORING_LIST:
    print("Optimizing xgboost params for", scoring, "with random search...")
    best_estimator = perform_random_search(
        estimator=clf,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        param_grid=XGBOOST_RANDOMSEARCH_PARAMS,
        scoring=scoring,
    )
    _ = calculate_scores(best_estimator, X_val, y_val)

for scoring in SCORING_LIST:
    print("Optimizing xgboost params for", scoring, "with bayes search...")
    best_estimator = perform_bayes_search(
        estimator=clf,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        param_grid=XGBOOST_BAYESSEARCH_PARAMS,
        scoring=scoring,
    )
    _ = calculate_scores(best_estimator, X_val, y_val)
