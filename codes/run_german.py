import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import os
import errno
from preprocess import preprocess_germandataset, reweighing_data
from main import run, accuracy_dataframe, fairness_metrics_dataframe


def save_output(df_accuracy_reweigh, adversarial_reweigh, prejudice_reweigh, neural_network_reweigh, ensemble_reweigh,
                df_accuracy_nonreweigh, adversarial_nonreweigh, prejudice_nonreweigh, neural_network_nonreweigh, ensemble_nonreweigh):

    try:
        os.makedirs('../results/german/reweighed')
        os.makedirs('../results/german/non-reweighed')
    except OSError as e:
        if e.errno != errno.EEXIST:
            pass
    df_accuracy_reweigh.to_csv("../results/german/reweighed/accuracy.csv", encoding='utf-8')
    adversarial_reweigh.to_csv("../results/german/reweighed/adversarial.csv", encoding='utf-8')
    prejudice_reweigh.to_csv("../results/german/reweighed/prejudice.csv", encoding='utf-8')
    neural_network_reweigh.to_csv("../results/german/reweighed/neural_net.csv", encoding='utf-8')
    ensemble_reweigh.to_csv("../results/german/reweighed/ensemble.csv", encoding='utf-8')

    df_accuracy_nonreweigh.to_csv("../results/german/non-reweighed/accuracy.csv", encoding='utf-8')
    adversarial_nonreweigh.to_csv("../results/german/non-reweighed/adversarial.csv", encoding='utf-8')
    prejudice_nonreweigh.to_csv("../results/german/non-reweighed/prejudice.csv", encoding='utf-8')
    neural_network_nonreweigh.to_csv("../results/german/non-reweighed/neural_net.csv", encoding='utf-8')
    ensemble_nonreweigh.to_csv("../results/german/non-reweighed/ensemble.csv", encoding='utf-8')
    print("Saved all outputs to csv")


def main():

    # load dataset
    df = pd.read_csv('../dataset/german_credit.csv')

    # transform data to StandardDataset object
    data = preprocess_germandataset(df)

    # define priviledged and unpriviledged groups
    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]

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