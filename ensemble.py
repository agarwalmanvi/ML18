import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.metrics import accuracy_score
from scipy.stats import mode
import pandas as pd
import numpy as np
import os
import errno

from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

from preprocess import preprocess_data
from adversarial_debiasing import adversarial
from prejudice_remover import prejudice
from neuralnet import nondebiased_classifier


# save output to csv file
def write_to_csv(df, predicted, filename):
    output, _ = df.convert_to_dataframe()
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

def make_predictions(train, test, privileged_groups, unprivileged_groups):
    # apply the 3 classifiers
    pred_adversarial = adversarial(train, test, privileged_groups, unprivileged_groups)
    pred_prejudice = prejudice(train, test)
    pred_nondebiased = nondebiased_classifier(train, test, privileged_groups, unprivileged_groups)

    # ensemble classifier using majority vote
    pred_labels = []
    for i in range (0, len(test.features)):
        arr = mode([pred_adversarial.labels[i], pred_prejudice.labels[i], pred_nondebiased.labels[i]])
        pred_labels.append(arr[0][0])

    pred_ensemble = test.copy()
    pred_ensemble.labels = np.array(pred_labels)

    return pred_adversarial, pred_prejudice, pred_nondebiased, pred_ensemble
    
# calculate prediction accuracy
def calculate_accuracy(test, pred_adversarial, pred_prejudice, pred_nondebiased, pred_ensemble):
    accuracy_adversarial = accuracy_score(y_true = test.labels, y_pred = pred_adversarial.labels)
    accuracy_prejudice = accuracy_score(y_true = test.labels, y_pred = pred_prejudice.labels)
    accuracy_nondebiased = accuracy_score(y_true = test.labels, y_pred = pred_nondebiased.labels)
    accuracy_ensemble = accuracy_score(y_true = test.labels, y_pred = pred_ensemble.labels)

    print("Classification accuracy of adversarial debiasing = {:.2f}%".format(accuracy_adversarial * 100))
    print("Classification accuracy of prejudice remover = {:.2f}%".format(accuracy_prejudice * 100))
    print("Classification accuracy of non-debiasing classifier = {:.2f}%".format(accuracy_nondebiased * 100))
    print("Classification accuracy of ensemble classifier = {:.2f}%".format(accuracy_ensemble * 100))

    # accuracy = [accuracy_adversarial, accuracy_prejudice, accuracy_nondebiased, accuracy_ensemble]

    # return accuracy

def fairness_metrics(test, pred, unprivileged_groups, privileged_groups, opt=False):
    metric1 = BinaryLabelDatasetMetric(pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    metric2 = ClassificationMetric(test, pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    print("Mean difference = {}".format(metric1.mean_difference()))
    print("Disparate impact = {}".format(metric2.disparate_impact()))

    if opt == True:
        print("Equal Opportunity Difference = {}".format(np.nan))
        print("Average Odds Difference = {}".format(np.nan))
    else:
        print("Equal Opportunity Difference = {}".format(metric2.equal_opportunity_difference()))
        print("Average Odds Difference = {}".format(metric2.average_odds_difference()))
    
    print("Theil Index = {}".format(metric2.theil_index()))

    # if opt == True:
    #     fairness_scores = fairness_scores = [metric1.mean_difference(), metric2.disparate_impact(), np.nan, np.nan, metric2.theil_index()]
    # else:
    #     fairness_scores = [metric1.mean_difference(), metric2.disparate_impact(), metric2.equal_opportunity_difference(), metric2.average_odds_difference(), metric2.theil_index()]

    # return fairness_scores


def main():
    # load dataset
    data1 = pd.read_csv('dataset/adult.csv')
    data2 = pd.read_csv('dataset/compas-scores-two-years.csv')
    data3 = pd.read_csv('dataset/german_credit.csv')

    # define priviledged and unpriviledged groups
    privileged_groups1, unprivileged_groups1 = protected_attributes('adult', 1)
    privileged_groups2, unprivileged_groups2 = protected_attributes('compas', 1)
    privileged_groups3, unprivileged_groups3 = protected_attributes('german', 1)

    # preprocessing
    train1, test1 = preprocess_data(data1, unprivileged_groups1, privileged_groups1, 'adult')
    train2, test2 = preprocess_data(data2, unprivileged_groups2, privileged_groups2, 'compas')
    train3, test3 = preprocess_data(data3, unprivileged_groups3, privileged_groups3, 'german')

    # perform classification
    pred_adversarial1, pred_prejudice1, pred_nondebiased1, pred_ensemble1 = make_predictions(train1, test1, privileged_groups1, unprivileged_groups1)
    pred_adversarial2, pred_prejudice2, pred_nondebiased2, pred_ensemble2 = make_predictions(train2, test2, privileged_groups2, unprivileged_groups2)
    pred_adversarial3, pred_prejudice3, pred_nondebiased3, pred_ensemble3 = make_predictions(train3, test3, privileged_groups3, unprivileged_groups3)

    # calculate classification accuracy
    print()
    print('       Classification using Adult dataset')
    calculate_accuracy(test1, pred_adversarial1, pred_prejudice1, pred_nondebiased1, pred_ensemble1)
    print()

    # calculate fairness metrics
    print('Fairness metrics for Adult dataset')
    print('    Adversarial Debiasing')
    fairness_metrics(test1, pred_adversarial1, unprivileged_groups1, privileged_groups1, False)
    print()
    print('    Prejudice Remover')
    fairness_metrics(test1, pred_prejudice1, unprivileged_groups1, privileged_groups1, True)
    print()
    print('    Nondebiasing')
    fairness_metrics(test1, pred_nondebiased1, unprivileged_groups1, privileged_groups1, False)
    print()
    print('    Ensemble')
    fairness_metrics(test1, pred_ensemble1, unprivileged_groups1, privileged_groups1, False)
    print()

    print('       Classification using Compas dataset')
    calculate_accuracy(test2, pred_adversarial2, pred_prejudice2, pred_nondebiased2, pred_ensemble2)
    print()
    
    # calculate fairness metrics
    print('Fairness metrics for Compas dataset')
    print('    Adversarial Debiasing')
    fairness_metrics(test2, pred_adversarial2, unprivileged_groups2, privileged_groups2, False)
    print()
    print('    Prejudice Remover')
    fairness_metrics(test2, pred_prejudice2, unprivileged_groups2, privileged_groups2, True)
    print()
    print('    Nondebiasing')
    fairness_metrics(test2, pred_nondebiased2, unprivileged_groups2, privileged_groups2, False)
    print()
    print('    Ensemble')
    fairness_metrics(test2, pred_ensemble2, unprivileged_groups2, privileged_groups2, False)
    print()

    print('      Classification using German dataset')
    calculate_accuracy(test3, pred_adversarial3, pred_prejudice3, pred_nondebiased3, pred_ensemble3)
    print()
    
    # calculate fairness metrics
    print('Fairness metrics for German dataset')
    print('    Adversarial Debiasing')
    fairness_metrics(test3, pred_adversarial3, unprivileged_groups3, privileged_groups3, False)
    print()
    print('    Prejudice Remover')
    fairness_metrics(test3, pred_prejudice3, unprivileged_groups3, privileged_groups3, True)
    print()
    print('    Nondebiasing')
    fairness_metrics(test3, pred_nondebiased3, unprivileged_groups3, privileged_groups3, False)
    print()
    print('    Ensemble')
    fairness_metrics(test3, pred_ensemble3, unprivileged_groups3, privileged_groups3, False)
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
    write_to_csv(test1, pred_ensemble1, 'output/adult/output_ensemble.csv')

    write_to_csv(test2, pred_adversarial2, 'output/compas/output_adversarial.csv')
    write_to_csv(test2, pred_prejudice2, 'output/compas/output_prejudice.csv')
    write_to_csv(test2, pred_nondebiased2, 'output/compas/output_nondebiased.csv')
    write_to_csv(test2, pred_ensemble2, 'output/compas/output_ensemble.csv')

    write_to_csv(test3, pred_adversarial3, 'output/german/output_adversarial.csv')
    write_to_csv(test3, pred_prejudice3, 'output/german/output_prejudice.csv')
    write_to_csv(test3, pred_nondebiased3, 'output/german/output_nondebiased.csv')
    write_to_csv(test3, pred_ensemble3, 'output/german/output_ensemble.csv')


if __name__ == "__main__":
    main()