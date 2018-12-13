from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
from aif360.algorithms.inprocessing import AdversarialDebiasing, PrejudiceRemover, ARTClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# load preprocessed dataset
data = load_preproc_data_adult()

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

# compute accuracy based on the debiased model
metric_adversarial = ClassificationMetric(test, test_adversarial, unprivileged_groups, privileged_groups)
accuracy_adversarial = metric_adversarial.accuracy()
print("Classification accuracy of adversarial debiasing = {:.2f}%".format(accuracy_adversarial * 100))

adv_di = metric_adversarial.disparate_impact()
adv_eod = metric_adversarial.equal_opportunity_difference()
adv_aod = metric_adversarial.average_odds_difference()
adv_theil = metric_adversarial.theil_index()
print("Disparate impact = {}".format(adv_di))
print("Equal Opportunity Difference = {}".format(adv_eod))
print("Average Odds Difference = {}".format(adv_aod))
print("Theil index = {}".format(adv_theil))
print()


prejudice_model = PrejudiceRemover(eta=10, sensitive_attr='sex')

fit_prejudice_model = prejudice_model.fit(train)
train_prejudice = prejudice_model.predict(train)

test_prejudice = prejudice_model.predict(test)

# print(test.features.shape)
print(test_prejudice.features)

# accuracy_prejudice = accuracy_score(y_true = test.labels, y_pred = test_prejudice.labels)
# print("Classification accuracy of prejudice remover = {:.2f}%".format(accuracy_prejudice * 100))


metric_prejudice = ClassificationMetric(test, test_prejudice.features, unprivileged_groups, privileged_groups)
print('{}'.format(metric_prejudice.disparate_impact()))
print('{}'.format(metric_prejudice.equal_opportunity_difference()))

