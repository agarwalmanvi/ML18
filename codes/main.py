import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np

from preprocess import reweighing_data
from adversarial_debiasing import adversarial
from prejudice_remover import prejudice
from neuralnet import nondebiased_classifier
from ensemble import ensemble



# Make predictions using four classifiers
def make_predictions(train, test, unprivileged_groups, privileged_groups):

    # Adversarial Debiasing
    pred_adversarial, accuracy_adversarial, metrics_adversarial = adversarial(train, test, privileged_groups, unprivileged_groups)

    # Prejudice Remover
    pred_prejudice, accuracy_prejudice, metrics_prejudice = prejudice(train, test, unprivileged_groups, privileged_groups)

    # Neural Network
    pred_nondebiasing, accuracy_nondebiasing, metrics_nondebiasing = nondebiased_classifier(train, test, privileged_groups, unprivileged_groups)

    # Ensemble
    accuracy_ensemble, metrics_ensemble = ensemble(test, pred_adversarial, pred_prejudice, pred_nondebiasing, unprivileged_groups, privileged_groups)

    # Store all accuracy scores and fairness scores
    accuracy_scores = [accuracy_adversarial, accuracy_prejudice, accuracy_nondebiasing, accuracy_ensemble]
    fairness_scores = [metrics_adversarial, metrics_prejudice, metrics_nondebiasing, metrics_ensemble]
    
    return accuracy_scores, fairness_scores


def run(data, runs, privileged_groups, unprivileged_groups, reweigh_option):

    # initialize cells to store accuracy and fairness scores of all runs
    matrix_accuracy = {}
    matrix_fairness_scores = {}

    # perform classification for several runs
    for i in range(0, runs):
        print('run =', i+1)

        # split data to training and testing sets
        train, test = data.split([0.7], shuffle=True)

        if reweigh_option == True:

            # transform training data with reweighing
            train_transformed = reweighing_data(train, unprivileged_groups, privileged_groups)
        else:

            # without reweighing
            train_transformed = train

        # calculate accuracy and fairness scores
        accuracy, metrics = make_predictions(train_transformed, test, unprivileged_groups, privileged_groups)
    
        # store values for each run
        matrix_accuracy[i] = accuracy
        matrix_fairness_scores[i] = metrics
    
    return matrix_accuracy, matrix_fairness_scores


def compile_metrics(runs, matrix_fairness):
    metrics_adversarial = []
    metrics_prejudice = []
    metrics_nondebiasing = []
    metrics_ensemble = []

    for i in range(0, runs):
        metrics_adversarial.append(matrix_fairness[i][0])
        metrics_prejudice.append(matrix_fairness[i][1])
        metrics_nondebiasing.append(matrix_fairness[i][2])
        metrics_ensemble.append(matrix_fairness[i][3])

    return metrics_adversarial, metrics_prejudice, metrics_nondebiasing, metrics_ensemble


def accuracy_dataframe(matrix_accuracy):

    # create data frame for all metrics
    columns = ['Adversarial Debiasing', 'Prejudice Remover', 'Nondebiasing', 'Ensemble']

    # accuracy
    accuracy_values = np.array(list(matrix_accuracy.values()))
    df_accuracy = pd.DataFrame(accuracy_values, columns=columns)

    return df_accuracy


def fairness_metrics_dataframe(runs, fairness_metrics_nonreweigh):

    metrics_adversarial, metrics_prejudice, metrics_nondebiasing, metrics_ensemble = compile_metrics(runs, fairness_metrics_nonreweigh)

    columns = ['Mean Difference', 'Disparate Impact', 'Equal Opportunity Difference', 'Average Odds Difference', 'Theil Index']
    df_adversarial = pd.DataFrame(metrics_adversarial, columns=columns)
    df_prejudice = pd.DataFrame(metrics_prejudice, columns=columns)
    df_neural_network = pd.DataFrame(metrics_nondebiasing, columns=columns)
    df_ensemble = pd.DataFrame(metrics_ensemble, columns=columns)

    return df_adversarial, df_prejudice, df_neural_network, df_ensemble