import sklearn.metrics as metrics


def calculate_scores(estimator, X_val, y_val):
    y_val_prediction = estimator.predict(X_val)
    y_val_proba = estimator.predict_proba(X_val)[:, 1]

    conf_matrix = metrics.confusion_matrix(y_val, y_val_prediction)
    accuracy_score = metrics.accuracy_score(y_val, y_val_prediction)
    roc_auc_score = metrics.roc_auc_score(y_val, y_val_proba)
    f1_score = metrics.f1_score(y_val, y_val_prediction)
    classification_report = metrics.classification_report(y_val, y_val_prediction)

    print("Confusion Matrix: \n%s" % str(conf_matrix))
    print("\nAccuracy: %.4f" % accuracy_score)
    print("\nAUC: %.4f" % roc_auc_score)
    print("\nF1 Score: %.4f" % f1_score)
    print("\nClassification Report: \n", classification_report)

    return {
        "conf_matrix": conf_matrix,
        "accuracy_score": accuracy_score,
        "roc_auc_score": roc_auc_score,
        "f1_score": f1_score,
        "classification_report": classification_report,
    }
