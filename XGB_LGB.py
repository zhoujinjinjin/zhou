import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from numpy import mean
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import os


def normalize(input):
    output = (input - input.min()) / (input.max() - input.min())
    output = round(output, 1)
    return output


def LoadData(file_name, normalization):
    database = pd.read_csv(file_name)

    if normalization == True:
        database['Balance'] = normalize(database['Balance'])
        database['EstimatedSalary'] = normalize(database['EstimatedSalary'])
    else:
        pass

    # attribute = ['CustomerId', 'Geography', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
    #              'EstimatedSalary', 'Exited', 'CreditLevel']

    database['Geography'] = database['Geography'].map({'Spain': 0, 'France': 1, 'Germany': 2})

    unwanted_cols = ['CustomerId']
    # database['BalanceSalaryRatio'] = database.Balance / database.EstimatedSalary
    database.drop(unwanted_cols, inplace=True, axis=1)

    return database


def split(database):
    target = database["CreditLevel"] - 1
    database.drop(["CreditLevel"], inplace=True, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(database, target, test_size=0.2,
                                                        stratify=target)

    return database, X_train, X_test, y_train, y_test


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    database = LoadData('BankChurners.csv', True)

    database, X_train, X_test, y_train, y_test = split(database)

    print(database.columns)

    xgb_model = XGBClassifier(learning_rate=0.09, n_estimators=250, max_depth=3, objective='multi:softmax',
                              subsample=0.8,
                              alpha=0.01,
                              gamma=0.1, min_child_weight=3, use_label_encoder=False, colsample_bytree=0.8,
                              eval_metric='mlogloss', num_class=10)

    xgb_model.fit(X_train, y_train)
    y_train_xgb = xgb_model.predict(X_train)
    print("XGB train accuracy:", accuracy_score(y_train_xgb, y_train) * 100)
    y_test_xgb = xgb_model.predict(X_test)
    print("XGB test accuracy:", accuracy_score(y_test_xgb, y_test) * 100)

    lgb_model = LGBMClassifier(learning_rate=0.2, num_leaves=250, boosting_type='gbdt', objective='multiclass',
                               metric='multi_logloss', max_depth=3, n_estimators=2500, subsample_for_bin=40000,
                               min_split_gain=2, min_child_weight=2, min_child_samples=5, subsample=0.9, num_class=10)

    lgb_model.fit(X_train, y_train)
    y_train_lgb = lgb_model.predict(X_train)
    print("LGB train accuracy:", accuracy_score(y_train_lgb, y_train) * 100)
    y_test_lgb = lgb_model.predict(X_test)
    print("LGB test accuracy:", accuracy_score(y_test_lgb, y_test) * 100)
