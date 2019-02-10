import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
import pandas as pd
import os
import errno
from preprocess import reweighing_data
from preprocess import preprocess_adultdataset
from main import run, compile_metrics, accuracy_dataframe, fairness_metrics_dataframe


def save_output(df_accuracy_reweigh, adversarial_reweigh, prejudice_reweigh, neural_network_reweigh, ensemble_reweigh,
                df_accuracy_nonreweigh, adversarial_nonreweigh, prejudice_nonreweigh, neural_network_nonreweigh, ensemble_nonreweigh):

    try:
        os.makedirs('../results/adult/reweighed')
        os.makedirs('../results/adult/non-reweighed')
    except OSError as e:
        if e.errno != errno.EEXIST:
            pass

    df_accuracy_reweigh.to_csv("../results/adult/reweighed/accuracy.csv", encoding='utf-8')
    adversarial_reweigh.to_csv("../results/adult/reweighed/adversarial.csv", encoding='utf-8')
    prejudice_reweigh.to_csv("../results/adult/reweighed/prejudice.csv", encoding='utf-8')
    neural_network_reweigh.to_csv("../results/adult/reweighed/neural_net.csv", encoding='utf-8')
    ensemble_reweigh.to_csv("../results/adult/reweighed/ensemble.csv", encoding='utf-8')

    df_accuracy_nonreweigh.to_csv("../results/adult/non-reweighed/accuracy.csv", encoding='utf-8')
    adversarial_nonreweigh.to_csv("../results/adult/non-reweighed/adversarial.csv", encoding='utf-8')
    prejudice_nonreweigh.to_csv("../results/adult/non-reweighed/prejudice.csv", encoding='utf-8')
    neural_network_nonreweigh.to_csv("../results/adult/non-reweighed/neural_net.csv", encoding='utf-8')
    ensemble_nonreweigh.to_csv("../results/adult/non-reweighed/ensemble.csv", encoding='utf-8')
    print("Saved all outputs to csv")


def main():
    
    # load dataset
    data = load_preproc_data_adult()

    # uncomment the following lines to load data from the csv and uncomment line 9
    # df = pd.read_csv('../dataset/adult.csv')
    # data = preprocess_adultdataset(df)

    # define priviledged and unpriviledged groups
    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]

    # set the number of runs
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