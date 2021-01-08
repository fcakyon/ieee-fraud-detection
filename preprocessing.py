import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from utils import reduce_mem_usage

VAL_SPLIT = 0.2


def get_data(val_split: float = VAL_SPLIT, apply_label_encoding=False, fillna=False):
    """
    You need to download required data from: https://www.kaggle.com/c/ieee-fraud-detection/data into ./data folder
    """
    # Import data:
    print("Importing data...")
    train_transaction = pd.read_csv("data/train_transaction.csv")
    train_identity = pd.read_csv("data/train_identity.csv")
    test_transaction = pd.read_csv("data/test_transaction.csv")
    test_identity = pd.read_csv("data/test_identity.csv")

    # Combine transaction and idendity columns:
    print("Merging transaction and idendity data...")
    train = pd.merge(train_transaction, train_identity, on="TransactionID", how="left")
    train = train.set_index("TransactionID", drop="True")
    test = pd.merge(test_transaction, test_identity, on="TransactionID", how="left")
    test = test.set_index("TransactionID", drop="True")
    del train_transaction, train_identity, test_transaction, test_identity

    # Rename test data columns
    mapping = {}
    for column_name in test.columns:
        mapping[column_name] = column_name.replace("-", "_")
    test.rename(columns=mapping, inplace=True)

    # Reduce memory usage
    print("Reducing memory usage...")
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)

    # Split train into train/val
    y = train["isFraud"].copy()
    X = train.drop("isFraud", axis=1)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SPLIT, random_state=13
    )

    # fill nans:
    X_test = test.copy()
    del train, test, X, y

    if fillna:
        X_train = X_train.fillna(-999)
        X_val = X_val.fillna(-999)
        X_test = X_test.fillna(-999)

    # Label encoding:
    categorical_features = []
    for f in X_train.columns:
        if X_train[f].dtype == "object" or X_test[f].dtype == "object":
            categorical_features.append(f)
            if apply_label_encoding:
                lbl = preprocessing.LabelEncoder()
                lbl.fit(
                    list(X_train[f].values)
                    + list(X_test[f].values)
                    + list(X_val[f].values)
                )
                X_train[f] = lbl.transform(list(X_train[f].values))
                X_val[f] = lbl.transform(list(X_val[f].values))
                X_test[f] = lbl.transform(list(X_test[f].values))

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "categorical_features": categorical_features,
    }
