import os
import sys
import click

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from general.xgb_model import preprocess_data, make_predictions, evaluate_accuracy
from general.data_generation import read_list_of_graphs, generate_data_from_list, draw_fa2, draw_kk
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
from fa.forceatlas2 import ForceAtlas2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

fa2 = ForceAtlas2()

DRAW_FS = {
    'kk': nx.kamada_kawai_layout,
    'fa2': fa2.forceatlas2_networkx_layout
}

#@click.command()
#@click.option('-r', '--res', 'results_file', type=click.Path(writable=True), prompt='Saving results in', default='results.txt', help='Path to save results.')
#@click.option('-i', '--ignore_feat', 'unwanted_features', help='Features to ignore.', type=str, multiple=True)
#@click.option('-a', '--alg', 'algo_name', type=click.Choice(['kk', 'fa2'], case_sensitive=False), help="Drawing algorithm to use")

def perform_experiment_model(results_file: str, unwanted_features: list, algo_name: str):
    """Runs the experiments of Edge Relaxation Graph Drawing."""

    # Select dataframe that contains that was generated with the algorithm I chose
    df = pd.read_csv(f'../data/graph_train_experiment_{algo_name}.csv')

    # Select features I want to use in the experiment from the dataframe, drop the others
    df.drop([col for col in df.columns if col in unwanted_features],
            axis=1, inplace=True)
    # Train an xgb model with the selected features
    Xn, yn = preprocess_data(df)
    Xn_train, Xn_test, yn_train, yn_test = train_test_split(
        Xn, yn, shuffle=True)
    xgb = XGBClassifier(learning_rate=0.02, n_estimators=600,
                        objective='binary:logistic', silent=True, nthread=1)
    xgb.fit(Xn_train, yn_train)
    xgb.save_model(f'../data/xgb_{algo_name}.bin')
    # Evaluate the model and save the results in results_file
    y_res = make_predictions(xgb, Xn_test)
    f1, acc = evaluate_accuracy(yn_test, y_res)
    with open(results_file, 'w') as f:
        f.write(f'Algorithm: {algo_name} \n')
        f.write(f'Unwanted features: {unwanted_features} \n')
        f.write(f'F1 score: {f1} \n')
        f.write(f'Accuracy: {acc} \n')
        f.write(f'----------------------------------------\n')

    # Evaluate the model used for drawing
    # metrics_relax_one    = eval_relax_one(xgb, df, gdict, DRAW_FS[algo_name])
    # metrics_direct_relax = eval_direct_relax(xgb, df, gdict, DRAW_FS[algo_name])
    # metrics_relax_recomp = eval_relax_recomp(xgb, df, gdict, DRAW_FS[algo_name])
    # metrics_relax_block  = eval_relax_block(xgb, df, gdict, DRAW_FS[algo_name])

    # eval(xgb, df, gdict, relax_one, results_file, DRAW_FS[algo_name], 'relax_one')

#perform_experiment('results_experiment_fa2.txt', [], 'fa2')
if __name__ == '__main__':
    perform_experiment_model('results_model_fa2.txt', [], 'fa2')
    perform_experiment_model('results_model_kk.txt', [], 'kk')