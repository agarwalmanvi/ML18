from sklearn.metrics import accuracy_score
from scipy.stats import mode
import pandas as pd
import numpy as np

from preprocess import preprocess_data
from adversarial_debiasing import adversarial
from prejudice_remover import prejudice
from neuralnet import nondebiased_classifier

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# append the classified labels into the input data and save it as a csv file
def save_output(input, predicted, filename):
    output, _ = input.convert_to_dataframe()

    if filename == 'output/output_ensemble.csv':
        output['labels'] = predicted
    else:
        output['labels'] = predicted.labels

    output.to_csv(filename)

def make_predictions(train, test):
    # define priviledged and unpriviledged groups
    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]

    # apply the 3 classifiers
    pred_adversarial = adversarial(train, test, privileged_groups, unprivileged_groups)
    pred_prejudice = prejudice(train, test)
    pred_nondebiased = nondebiased_classifier(train, test, privileged_groups, unprivileged_groups)

    # ensemble classifier using majority vote
    final_pred = []
    for i in range (0, len(test.features)):
        arr = mode([pred_adversarial.labels[i], pred_prejudice.labels[i], pred_nondebiased.labels[i]])
        final_pred.append(arr[0][0])

    return pred_adversarial, pred_prejudice, pred_nondebiased, final_pred
    
# calculate prediction accuracy
def calculate_accuracy(test, test_adversarial, test_prejudice, test_nondebiased, final_pred):
    accuracy_adversarial = accuracy_score(y_true = test.labels, y_pred = test_adversarial.labels)
    accuracy_prejudice = accuracy_score(y_true = test.labels, y_pred = test_prejudice.labels)
    accuracy_nondebiased = accuracy_score(y_true = test.labels, y_pred = test_nondebiased.labels)
    accuracy_ensemble = accuracy_score(y_true = test.labels, y_pred = final_pred)

    print("Classification accuracy of adversarial debiasing = {:.2f}%".format(accuracy_adversarial * 100))
    print("Classification accuracy of prejudice remover = {:.2f}%".format(accuracy_prejudice * 100))
    print("Classification accuracy of non-debiasing classifier = {:.2f}%".format(accuracy_nondebiased * 100))
    print("Classification accuracy of ensemble classifier = {:.2f}%".format(accuracy_ensemble * 100))

def main():
    # load preprocessed dataset
    dataset = pd.read_csv('dataset/adult.csv')
    preproc_data = preprocess_data(dataset)

    # split dataset into training and testing set with a fraction 70/30
    data, _ = preproc_data.split([0.5], shuffle=True)
    train, test = data.split([0.7], shuffle=True)

    # perform classification
    pred_adversarial, pred_prejudice, pred_nondebiased, final_pred = make_predictions(train, test)

    # save output to csv files
    save_output(test, pred_adversarial, 'output/output_adversarial.csv')
    save_output(test, pred_prejudice, 'output/output_prejudice.csv')
    save_output(test, pred_nondebiased, 'output/output_nondebiased.csv')
    save_output(test, final_pred, 'output/output_ensemble.csv')

    # calculate classification accuracy
    calculate_accuracy(test, pred_adversarial, pred_prejudice, pred_nondebiased, final_pred)


if __name__ == "__main__":
    main()