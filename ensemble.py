import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.metrics import accuracy_score
from scipy.stats import mode
import pandas as pd
import numpy as np
import os
import errno

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult

from preprocess import preprocess_data
from adversarial_debiasing import adversarial
from prejudice_remover import prejudice
from neuralnet import nondebiased_classifier


def write_to_csv(df, predicted, filename):
    output, _ = df.convert_to_dataframe()

    if filename == 'output/adult/output_ensemble.csv' or filename == 'output/compas/output_ensemble.csv' or filename == 'output/german/output_ensemble.csv':
        output['labels'] = predicted
    else:
        output['labels'] = predicted.labels

    output.to_csv(filename)

# define protected attributes based on the data set
def protected_attributes(data, args):
    if data == 'adult' or data == 'compas':
        if args == 1:
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
        else:
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]
    
    if data == 'german':
        if args == 1:
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
        else:
            privileged_groups = [{'age': 1}]
            unprivileged_groups = [{'age': 0}]
    
    return privileged_groups, unprivileged_groups

def make_predictions(train, test):
    # define priviledged and unpriviledged groups
    privileged_groups, unprivileged_groups = protected_attributes('adult', 1)

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
    # load dataset
    # data1 = pd.read_csv('dataset/adult.csv')
    data2 = pd.read_csv('dataset/compas-scores-two-years.csv')
    data3 = pd.read_csv('dataset/german_credit.csv')

    # preprocessing
    # preproc_data1 = preprocess_data(data1, 'adult')

    # QUICK FIX
    preproc_data1 = load_preproc_data_adult()
    preproc_data2 = preprocess_data(data2, 'compas')
    preproc_data3 = preprocess_data(data3, 'german')

    # split dataset into training and testing set with a fraction of 70/30
    data_split, _ = preproc_data1.split([0.5], shuffle=True)
    train1, test1 = data_split.split([0.7], shuffle=True)
    train2, test2 = preproc_data2.split([0.7], shuffle=True)
    train3, test3 = preproc_data3.split([0.7], shuffle=True)

    # perform classification
    pred_adversarial1, pred_prejudice1, pred_nondebiased1, final_pred1 = make_predictions(train1, test1)
    pred_adversarial2, pred_prejudice2, pred_nondebiased2, final_pred2 = make_predictions(train2, test2)
    pred_adversarial3, pred_prejudice3, pred_nondebiased3, final_pred3 = make_predictions(train3, test3)

    # calculate classification accuracy
    print()
    print('       Classification using Adult dataset')
    calculate_accuracy(test1, pred_adversarial1, pred_prejudice1, pred_nondebiased1, final_pred1)
    print()

    print('       Classification using Compas dataset')
    calculate_accuracy(test2, pred_adversarial2, pred_prejudice2, pred_nondebiased2, final_pred2)
    print()

    print('      Classification using German dataset')
    calculate_accuracy(test3, pred_adversarial3, pred_prejudice3, pred_nondebiased3, final_pred3)
    print()

    # save output to csv files
    try:
        os.makedirs('output/adult')
        os.makedirs('output/compas')
        os.makedirs('output/german')
    except OSError as e:
        if e.errno != errno.EEXIST:
            pass

    write_to_csv(test1, pred_adversarial1, 'output/adult/output_adversarial.csv')
    write_to_csv(test1, pred_prejudice1, 'output/adult/output_prejudice.csv')
    write_to_csv(test1, pred_nondebiased1, 'output/adult/output_nondebiased.csv')
    write_to_csv(test1, final_pred1, 'output/adult/output_ensemble.csv')

    write_to_csv(test2, pred_adversarial2, 'output/compas/output_adversarial.csv')
    write_to_csv(test2, pred_prejudice2, 'output/compas/output_prejudice.csv')
    write_to_csv(test2, pred_nondebiased2, 'output/compas/output_nondebiased.csv')
    write_to_csv(test2, final_pred2, 'output/compas/output_ensemble.csv')

    write_to_csv(test3, pred_adversarial3, 'output/german/output_adversarial.csv')
    write_to_csv(test3, pred_prejudice3, 'output/german/output_prejudice.csv')
    write_to_csv(test3, pred_nondebiased3, 'output/german/output_nondebiased.csv')
    write_to_csv(test3, final_pred3, 'output/german/output_ensemble.csv')


if __name__ == "__main__":
    main()
