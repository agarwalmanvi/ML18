import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_compas
import pandas as pd
import os
import errno
from preprocess import reweighing_data
from main import run, accuracy_dataframe, fairness_metrics_dataframe

def save_output(df_accuracy_reweigh, adversarial_reweigh, prejudice_reweigh, neural_network_reweigh, ensemble_reweigh,
                df_accuracy_nonreweigh, adversarial_nonreweigh, prejudice_nonreweigh, neural_network_nonreweigh, ensemble_nonreweigh):

    try:
        os.makedirs('../results/compas/reweighed')
        os.makedirs('../results/compas/non-reweighed')
    except OSError as e:
        if e.errno != errno.EEXIST:
            pass
    df_accuracy_reweigh.to_csv("../results/compas/reweighed/accuracy.csv", encoding='utf-8')
    adversarial_reweigh.to_csv("../results/compas/reweighed/adversarial.csv", encoding='utf-8')
    prejudice_reweigh.to_csv("../results/compas/reweighed/prejudice.csv", encoding='utf-8')
    neural_network_reweigh.to_csv("../results/compas/reweighed/neural_net.csv", encoding='utf-8')
    ensemble_reweigh.to_csv("../results/compas/reweighed/ensemble.csv", encoding='utf-8')

    df_accuracy_nonreweigh.to_csv("../results/compas/non-reweighed/accuracy.csv", encoding='utf-8')
    adversarial_nonreweigh.to_csv("../results/compas/non-reweighed/adversarial.csv", encoding='utf-8')
    prejudice_nonreweigh.to_csv("../results/compas/non-reweighed/prejudice.csv", encoding='utf-8')
    neural_network_nonreweigh.to_csv("../results/compas/non-reweighed/neural_net.csv", encoding='utf-8')
    ensemble_nonreweigh.to_csv("../results/compas/non-reweighed/ensemble.csv", encoding='utf-8')
    print("Saved all outputs to csv")

def main():

    # load dataset
    data = load_preproc_data_compas()

    # define priviledged and unpriviledged groups
    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]

    # uncomment the following lines to test it with race
    # privileged_groups = [{'race': 1}]
    # unprivileged_groups = [{'race': 0}]

    # set the number of runs for testing
    runs = 10

    # run with reweighing
    accuracy_reweigh, fairness_metrics_reweigh = run(data, runs, privileged_groups, unprivileged_groups, True)

    # convert to dataframe
    df_accuracy_reweigh = accuracy_dataframe(accuracy_reweigh)
    adversarial_reweigh, prejudice_reweigh, neural_network_reweigh, ensemble_reweigh = fairness_metrics_dataframe(runs, fairness_metrics_reweigh)

    # run without reweighing
    accuracy_nonreweigh, fairness_metrics_nonreweigh = run(data, runs, privileged_groups, unprivileged_groups, False)

    # convert to dataframe
    df_accuracy_nonreweigh = accuracy_dataframe(accuracy_nonreweigh)
    adversarial_nonreweigh, prejudice_nonreweigh, neural_network_nonreweigh, ensemble_nonreweigh = fairness_metrics_dataframe(runs, fairness_metrics_nonreweigh)

    # save output to csv
    save_output(df_accuracy_reweigh, adversarial_reweigh, prejudice_reweigh, neural_network_reweigh, ensemble_reweigh,
                df_accuracy_nonreweigh, adversarial_nonreweigh, prejudice_nonreweigh, neural_network_nonreweigh, ensemble_nonreweigh)

if __name__ == "__main__":
    main()