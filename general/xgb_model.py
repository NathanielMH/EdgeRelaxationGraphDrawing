# Description: XGBoost model for graph classification
# XGBoost data preparation
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

T = 0.5
# A parameter grid for XGBoost
params = {
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [6, 8, 10]
}


def preprocess_data(df: pd.DataFrame, return_labels: bool = True, drop_labels: bool = True):
    """Preprocesses data for XGBoost.

    Args:
        df2 (pd.DataFrame): DataFrame with information.

    Returns:
        Xn (np.array): Array of features.
        yn (np.array): Array of labels.
    """
    df['is_bridge'] = df['is_bridge'].astype(float)
    if return_labels:
        yn = (df['edge_cross_norm'] > 0).to_numpy(dtype=int)

    if drop_labels:
        df = df.drop(labels=['edge_cross_norm','diff_cross'],axis=1)
        
    df = df.drop(labels=['edge_id', 'graph_id', 'num_nodes', 'num_edges',
                           'benchmark', 'max_deg', 'min_deg', 'is_bridge'], axis=1)

    Xn = df.to_numpy(dtype=float)

    if return_labels:
        return Xn, yn

    return Xn


def perform_grid_search(xgb: XGBClassifier, X_train: np.array, y_train: np.array, params_grid: dict = params, njobs: int = 4, folds: int = 3):
    """
    Performs grid search on XGBClassifier

    Args:
        params_grid: grid of parameters
        X_train: train set features
        y_train: train set labels
        njobs: number of jobs to run in parallel

    Returns:
        None
    """

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)
    grid = GridSearchCV(estimator=xgb, param_grid=params, scoring='roc_auc',
                        n_jobs=njobs, cv=skf.split(X_train, y_train), verbose=3)
    grid.fit(X_train, y_train)

    return grid


def make_predictions(grid: GridSearchCV or XGBClassifier, X_test: np.array, T: float = 0.5):
    """
    Makes predictions on test set with threshold T

    Args:
        grid: grid search object
        X_test: test set features
        T: threshold

    Returns:
        predictions: predictions on test set
    """
    if isinstance(grid, GridSearchCV):
        y_pred = grid.best_estimator_.predict_proba(X_test)
    else:
        y_pred = grid.predict_proba(X_test)
    y_pred = [1 if y[1] > T else 0 for y in y_pred]

    return y_pred


def evaluate_accuracy(yn_test: np.array, yn_res: np.array):
    """
    Evaluates accuracy of predictions

    Args:
        yn_test: test set labels
        yn_res: predictions

    Returns:
        [f1_score,accuracy]
    """
    f1 = f1_score(yn_test, yn_res)
    acc = (sum([1 if yn_test[i] == yn_res[i]
           else 0 for i in range(len(yn_res))]))/len(yn_test)
    return [f1, acc]


def plot_precision_recall_with_threshold(yn_test: np.array, yn_res: np.array):
    """
    Plots precision and recall on test set according to threshold

    Args:
        yn_test: test set labels
        Xn_test: test set features
        xgb: XGBClassifier

    Returns:
        None
    """
    prec, rec, thresh = precision_recall_curve(
        yn_test, yn_res)

    fig, ax = plt.subplots()
    ax.plot(np.concatenate(([0], thresh)), prec, color='purple')
    ax.plot(np.concatenate(([0], thresh)), rec, color='green')

    ax.legend(['precision', 'recall'])

    # display plot
    plt.show()


def main_xgb():
    """Main function."""
    df = pd.read_csv('graph_train_experiment_kk.csv')
    Xn, yn = preprocess_data(df)
    Xn_train, Xn_test, yn_train, yn_test = train_test_split(
        Xn, yn, shuffle=True)
    xgb = XGBClassifier(learning_rate=0.02, n_estimators=600,
                        objective='binary:logistic', silent=True, nthread=1)
    #grid = perform_grid_search(Xn_train, yn_train, xgb=xgb)
    xgb.fit(Xn_train, yn_train)
    yn_res = make_predictions(xgb, Xn_test)
    evaluate_accuracy(yn_test, yn_res)
    # plot_precision_recall_with_threshold(yn_test, yn_res)

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main_xgb()