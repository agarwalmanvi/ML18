from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
from aif360.algorithms.inprocessing import AdversarialDebiasing, PrejudiceRemover

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from data import preprocessed_data

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# load preprocessed dataset
# data = load_preproc_data_adult()

data = preprocessed_data

# define priviledged and unpriviledged groups
# for this dataset, the protected attribute is sex
privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]

# split dataset into training and testing set with a fraction 70/30
train, test = data.split([0.7], shuffle=True)

# apply adversarial debiasing algorithm
sess = tf.Session()
adversarial_model = AdversarialDebiasing(privileged_groups, unprivileged_groups, scope_name='debiased_classifier', debias=True, sess=sess)
fit_adversarial_model = adversarial_model.fit(train)

# predict outcome using the debiased model
train_adversarial = adversarial_model.predict(train)
test_adversarial = adversarial_model.predict(test)

# compute accuracy and fairness metrics based on the debiased model
metric_adversarial = ClassificationMetric(test, test_adversarial, unprivileged_groups, privileged_groups)
print("Classification accuracy of adversarial debiasing = {:.2f}%".format(metric_adversarial.accuracy() * 100))
print()
print("Fairness metrics:")
print("Disparate impact = {}".format(metric_adversarial.disparate_impact()))
print("Equal Opportunity Difference = {}".format(metric_adversarial.equal_opportunity_difference()))
print("Average Odds Difference = {}".format(metric_adversarial.average_odds_difference()))
print("Theil index = {}".format(metric_adversarial.theil_index()))
print()

# apply prejudice remover algorithm
prejudice_model = PrejudiceRemover(eta=10, sensitive_attr='sex')
fit_prejudice_model = prejudice_model.fit(train)

# predict outcome using the fitted model
train_prejudice = prejudice_model.predict(train)
test_prejudice = prejudice_model.predict(test)

# compute accuracy
metric_prejudice = BinaryLabelDatasetMetric(test_prejudice, unprivileged_groups, privileged_groups)
accuracy_prejudice = accuracy_score(y_true = test.labels, y_pred = test_prejudice.labels)
print("Classification accuracy of prejudice remover = {:.2f}%".format(accuracy_prejudice * 100))
print()
print("Fairness metrics:")
print('Disparate impact = {}'.format(metric_prejudice.disparate_impact()))

# metric_prejudice = ClassificationMetric(test, test_prejudice, unprivileged_groups, privileged_groups)
# print('Equal Opportunity Difference = {}'.format(metric_prejudice.equal_opportunity_difference()))
# print('Average Odds Difference = {}'.format(metric_prejudice.average_odds_difference()))
# print('Theil index = {}'.format(metric_prejudice.theil_index()))

