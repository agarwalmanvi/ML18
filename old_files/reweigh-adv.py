import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.metrics import accuracy_score
from scipy.stats import mode
import pandas as pd
import numpy as np
import os
import errno
import tensorflow as tf

from IPython.display import Markdown, display

from preprocess import preprocess_data
from adversarial_debiasing import adversarial
from prejudice_remover import prejudice
from neuralnet import nondebiased_classifier

from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector

from aif360.algorithms.preprocessing.reweighing import Reweighing


from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german

from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult

privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]

cycles = []
metric_dataset_debiasing_train = []
metric_dataset_debiasing_test = []
metric_dataset_reweigh_train = []
metric_dataset_reweigh_test = []
dataset_orig = load_preproc_data_adult()

for i in range(10):
    train1, test1 = dataset_orig.split([0.7], shuffle=True)
    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                   privileged_groups=privileged_groups)
    RW.fit(train1)
    dataset_transf_train = RW.transform(train1)
    sess = tf.Session()
    debiased_model = AdversarialDebiasing(privileged_groups = privileged_groups,
                              unprivileged_groups = unprivileged_groups,
                              scope_name='debiased_classifier',
                              debias=True,
                              sess=sess)
    debiased_model.fit(train1)
    dataset_debiasing_train = debiased_model.predict(train1)
    dataset_debiasing_test = debiased_model.predict(test1)
    metric_debiasing_train = BinaryLabelDatasetMetric(dataset_debiasing_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
    metric_debiasing_test = BinaryLabelDatasetMetric(dataset_debiasing_test, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
    metric_dataset_debiasing_train.append(metric_debiasing_train.mean_difference())
    metric_dataset_debiasing_test.append(metric_debiasing_test.mean_difference())
    
    
    
    sess.close()
    tf.reset_default_graph()
    sess = tf.Session()
    debiased_model = AdversarialDebiasing(privileged_groups = privileged_groups,
                              unprivileged_groups = unprivileged_groups,
                              scope_name='debiased_classifier',
                              debias=True,
                              sess=sess)
    debiased_model.fit(dataset_transf_train)
    dataset_reweigh_train = debiased_model.predict(dataset_transf_train)
    dataset_reweigh_test = debiased_model.predict(test1)
    
    metric_reweigh_train = BinaryLabelDatasetMetric(dataset_reweigh_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
    metric_reweigh_test = BinaryLabelDatasetMetric(dataset_reweigh_test, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
    
    metric_dataset_reweigh_train.append(metric_reweigh_train.mean_difference())
    metric_dataset_reweigh_test.append(metric_reweigh_test.mean_difference())
    sess.close()
    tf.reset_default_graph()
    cycles.append(i)
    
# Metrics for the dataset from model with debiasing
for i in range(10):
    print("Without reweighing: ")
    print(metric_dataset_debiasing_train[i], '\t\t',metric_dataset_debiasing_test[i])
    print("With reweighing: ")
    print(metric_dataset_reweigh_train[i], '\t\t',metric_dataset_reweigh_test[i])
    print('\n\n')
