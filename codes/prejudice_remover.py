from aif360.algorithms.inprocessing import PrejudiceRemover
from sklearn.metrics import accuracy_score
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
import math
from metrics import equal_opp_diff, avg_odds_diff

# apply prejudice remover algorithm
def prejudice(train, test, unprivileged_groups, privileged_groups):
    prejudice_model = PrejudiceRemover(eta=100, sensitive_attr='sex')
    prejudice_model.fit(train)

    # predict outcome using the test set
    pred_prejudice = prejudice_model.predict(test)

    # calculate accuracy
    accuracy = accuracy_score(y_true = test.labels, y_pred = pred_prejudice.labels)

    # calculate fairness metrics
    metric_test = BinaryLabelDatasetMetric(pred_prejudice, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    acc_test = ClassificationMetric(test, pred_prejudice, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    # equal_opportunity_difference = equal_opp_diff(test, test_prejudice, 'sex', privileged=1, unprivileged=0, favourable=1, unfavourable=0)
    # average_odds_difference = avg_odds_diff(test, test_prejudice, 'sex', privileged=1, unprivileged=0, favourable=1, unfavourable=0)
    
    if acc_test.disparate_impact() == math.inf:
        disparate_impact = 5.0
    else:
        disparate_impact = acc_test.disparate_impact()
    
    # metrics = [metric_test.mean_difference(), disparate_impact, equal_opportunity_difference, average_odds_difference, acc_test.theil_index()]
    metrics = [metric_test.mean_difference(), disparate_impact, acc_test.theil_index()]

    return pred_prejudice, accuracy, metrics