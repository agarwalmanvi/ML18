from sklearn.metrics import accuracy_score
import pandas as pd
from preprocess import preprocess_data

from adversarial_debiasing import adversarial
from prejudice_remover import prejudice
from neuralnet import nondebiased_classifier

def write_output(input, predicted, filename):
    output, _ = input.convert_to_dataframe()
    output['labels'] = predicted.labels
    output.to_csv(filename)
    return output

def main():
    # load preprocessed dataset
    dataset = pd.read_csv('dataset/adult.csv')
    preproc_data = preprocess_data(dataset)

    # define priviledged and unpriviledged groups
    # for this dataset, the protected attribute is sex
    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]

    # split dataset into training and testing set with a fraction 70/30
    data, _ = preproc_data.split([0.5], shuffle=True)
    train, test = data.split([0.7], shuffle=True)

    # apply the 3 classifiers
    test_adversarial = adversarial(train, test, privileged_groups, unprivileged_groups)
    test_prejudice = prejudice(train, test)
    test_nondebiased = nondebiased_classifier(train, test, privileged_groups, unprivileged_groups)

    # write output to csv files
    output_adversarial = write_output(test, test_adversarial, 'output/output_adversarial.csv')
    output_prejudice = write_output(test, test_prejudice, 'output/output_prejudice.csv')
    output_nondebiased = write_output(test, test_nondebiased, 'output/output_nondebiased.csv')

    # calculate prediction accuracy
    accuracy_adversarial = accuracy_score(y_true = test.labels, y_pred = test_adversarial.labels)
    accuracy_prejudice = accuracy_score(y_true = test.labels, y_pred = test_prejudice.labels)
    accuracy_nondebiased = accuracy_score(y_true = test.labels, y_pred = test_nondebiased.labels)
    print("Classification accuracy of adversarial debiasing = {:.2f}%".format(accuracy_adversarial * 100))
    print("Classification accuracy of prejudice remover = {:.2f}%".format(accuracy_prejudice * 100))
    print("Classification accuracy of non-debiasing classifier = {:.2f}%".format(accuracy_nondebiased * 100))


if __name__ == "__main__":
    main()