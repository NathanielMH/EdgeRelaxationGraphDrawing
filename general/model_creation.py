import os
import sys
import click

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from general.model_utils import preprocess_data, make_predictions, evaluate_accuracy
from general.validation import relax_and_recompute, relax_block, relax_one, just_relax  
from general.data_generation import draw_fa2, draw_kk
from general.validation import eval
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
from fa.forceatlas2 import ForceAtlas2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier


fa2 = ForceAtlas2()

DRAW_FS = {
    'kk': draw_kk,
    'fa2': draw_fa2
}
# Add spectral/FR

#@click.command()
#@click.option('-r', '--res', 'results_file', type=click.Path(writable=True), prompt='Saving results in', default='results.txt', help='Path to save results.')
#@click.option('-i', '--ignore_feat', 'unwanted_features', help='Features to ignore.', type=str, multiple=True)
#@click.option('-a', '--alg', 'algo_name', type=click.Choice(['kk', 'fa2'], case_sensitive=False), help="Drawing algorithm to use")

def perform_experiment_model(results_file: str, unwanted_features: list, algo_name: str, classifier: str = 'xgb'):
    """Runs the experiments of Edge Relaxation Graph Drawing.
    
    Args:
        results_file: Path to save results.
        unwanted_features: Features to ignore.
        algo_name: Drawing algorithm to use.
    
    Returns:
        None
    """

    # Select dataframe that contains that was generated with the algorithm I chose
    df = pd.read_csv(f'../training_data/graph_train_{algo_name}.csv')

    # Select features I want to use in the experiment from the dataframe, drop the others
    df.drop([col for col in df.columns if col in unwanted_features],
            axis=1, inplace=True)
    
    # Train a model with the selected features
    df_train, df_test = train_test_split(df, shuffle=True, test_size=0.001,random_state=42)
    Xn_train, yn_train = preprocess_data(df_train)
    Xn_test, yn_test = preprocess_data(df_test)

    if classifier == 'xgb':
        # Grid search for Weighted XGBoost
        grid = {'scale_pos_weight': [0.5, 0.8, 1, 1.2, 1.5], 'max_depth': [2, 3, 4, 5, 6], 'n_estimators': [100, 200, 300, 400, 500]}
        grid_search = GridSearchCV(estimator = XGBClassifier(random_state=42), param_grid = grid, cv = 5, n_jobs = -1, verbose = 0)
        grid_search.fit(Xn_train, yn_train)
        model = grid_search.best_estimator_
        # model = XGBClassifier(scale_pos_weight=(len(yn_train)-np.sum(yn_train))/np.sum(yn_train)*0.8, random_state=42)
    elif classifier == 'rf':
        model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    
    elif classifier == 'cbc':
        # Grid search for Weighted CatBoost
        grid = {'scale_pos_weight': [0.5, 0.8, 1, 1.2, 1.5], 'max_depth': [2, 3, 4, 5, 6], 'n_estimators': [100, 200, 300, 400, 500]}
        grid_search = GridSearchCV(estimator = CatBoostClassifier(verbose=False, random_state=42), param_grid = grid, cv = 5, n_jobs = -1, verbose = 0)
        grid_search.fit(Xn_train, yn_train)
        model = grid_search.best_estimator_
        # model = CatBoostClassifier(iterations = 5000, scale_pos_weight=(len(yn_train)-np.sum(yn_train))/np.sum(yn_train)*0.8,verbose=False, random_state=42)

    model.fit(Xn_train, yn_train)
    model.save_model(f'../data/{classifier}_{algo_name}.bin')

    # Evaluate the model and save the results in results_file
    y_res = make_predictions(model, Xn_test)
    f1, acc, pr, re = evaluate_accuracy(yn_test, y_res)
    nPos = np.sum(yn_test)

    with open(results_file, 'w') as f:
        f.write(f'Algorithm: {algo_name} \n')
        f.write(f'Unwanted features: {unwanted_features} \n')
        f.write(f'F1 score: {f1} \n')
        f.write(f'Accuracy: {acc} \n')
        f.write(f'Precision: {pr} \n')
        f.write(f'Recall: {re} \n')
        f.write(f'Number of positive samples: {nPos} \n')
        f.write(f'Total number of samples: {len(yn_test)} \n')
        f.write(f'----------------------------------------\n')

if __name__ == '__main__':
    for classifier in ['cbc','xgb']:
        for algo in ['kk', 'fa2']:
            filename = '../results/results_{}_{}.txt'.format(classifier, algo)
            perform_experiment_model(filename, [], algo, classifier)