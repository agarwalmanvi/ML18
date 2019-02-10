import numpy as np
from sklearn.metrics import accuracy_score
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from scipy.stats import mode
    
def ensemble(test, pred_adversarial, pred_prejudice, pred_nondebiased, unprivileged_groups, privileged_groups):
    pred_labels = []
    for i in range (0, len(test.features)):
        arr = mode([pred_adversarial.labels[i], pred_prejudice.labels[i], pred_nondebiased.labels[i]])
        pred_labels.append(arr[0][0])

    pred_ensemble = test.copy()
    pred_ensemble.labels = np.array(pred_labels)

    accuracy = accuracy_score(y_true = test.labels, y_pred = pred_ensemble.labels)

    metric_test = BinaryLabelDatasetMetric(pred_ensemble, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    acc_test = ClassificationMetric(test, pred_ensemble, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    # metrics = [metric_test.mean_difference(),acc_test.disparate_impact(), acc_test.equal_opportunity_difference(), acc_test.average_odds_difference(), acc_test.theil_index()]
    metrics = [metric_test.mean_difference(),acc_test.disparate_impact(), acc_test.theil_index()]

    return accuracy, metrics