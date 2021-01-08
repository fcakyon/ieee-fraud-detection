import catboost as cb

from hyperparam_optimizing import (
    CATBOOST_BAYESSEARCH_PARAMS,
    CATBOOST_RANDOMSEARCH_PARAMS,
    SCORING_LIST,
    perform_bayes_search,
    perform_random_search,
)
from preprocessing import get_data
from scoring import calculate_scores

VAL_SPLIT = 0.2
data = get_data(val_split=VAL_SPLIT, apply_label_encoding=True, fillna=True)
X_train, X_val, X_test, y_train, y_val, categorical_features = (
    data["X_train"],
    data["X_val"],
    data["X_test"],
    data["y_train"],
    data["y_val"],
    data["categorical_features"],
)
clf = cb.CatBoostClassifier(
    n_estimators=200,
    learning_rate=0.05,
    metric_period=500,
    od_wait=500,
    task_type="CPU",
    depth=8,
)

print("Fitting a catboost model...")
clf.fit(X_train, y_train, cat_features=categorical_features)

_ = calculate_scores(clf, X_val, y_val)

for scoring in SCORING_LIST:
    print("Optimizing catboost params for", scoring, "with random search...")
    best_estimator = perform_random_search(
        estimator=clf,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        param_grid=CATBOOST_RANDOMSEARCH_PARAMS,
        scoring=scoring,
    )
    _ = calculate_scores(best_estimator, X_val, y_val)

for scoring in SCORING_LIST:
    print("Optimizing catboost params for", scoring, "with bayes search...")
    best_estimator = perform_bayes_search(
        estimator=clf,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        param_grid=CATBOOST_BAYESSEARCH_PARAMS,
        scoring=scoring,
    )
    _ = calculate_scores(best_estimator, X_val, y_val)
