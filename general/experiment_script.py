from xgb_model import preprocess_data, make_predictions, evaluate_accuracy
from data_generation import read_list_of_graphs, generate_data_from_list, draw_fa2, draw_kk
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


def perform_experiment(results_file: str, list_features: list, algo_name: str):

    # Select dataframe that contains that was generated with the algorithm I chose
    df = pd.read_csv('data/graph_train_experiment_'+algo_name+'.csv')

    # Select features I want to use in the experiment from the dataframe, drop the others
    df.drop([col for col in df.columns if col not in list_features],
            axis=1, inplace=True)
    # Train an xgb model with the selected features
    Xn, yn = preprocess_data(df)
    Xn_train, Xn_test, yn_train, yn_test = train_test_split(
        Xn, yn, shuffle=True)
    xgb = XGBClassifier(learning_rate=0.02, n_estimators=600,
                        objective='binary:logistic', silent=True, nthread=1)
    xgb.fit(Xn_train, yn_train)

    # Evaluate the model and save the results in results_file
    y_res = make_predictions(xgb, Xn_test)
    f1, acc = evaluate_accuracy(yn_test, y_res)
    with open(results_file, 'a') as f:
        f.write('Algorithm: '+algo_name + '\n')
        f.write('Features: ' + str(list_features) + '\n')
        f.write('F1 score: ' + str(f1) + '\n')
        f.write('Accuracy: ' + str(acc) + '\n')
        f.write('----------------------------------------\n')
