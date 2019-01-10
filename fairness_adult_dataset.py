import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.metrics import accuracy_score
from scipy.stats import mode
import pandas as pd
import numpy as np
import os
import errno
import tensorflow as tf
import copy
from copy import deepcopy
import csv

from aif360.datasets import BinaryLabelDataset, StructuredDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector

from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german

from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult

from scipy.stats import mode

###########################################

def run_trial():
    
    # stores each run (4 algos without reweighing, 4 algos with reweighing) as a sublist
    # number of sublists = number of runs
    # each sublist has four elements
    # we ONLY predict on the testing data

    ########## WITHOUT REWEIGHING #############
    stat_par = []
    disp_imp = []
    eq_opp_diff = []
    avg_odds_diff = []
    theil = []
    acc = []

    ########## WITH REWEIGHING #############
    stat_par_reweigh = []
    disp_imp_reweigh = []
    eq_opp_diff_reweigh = []
    avg_odds_diff_reweigh = []
    theil_reweigh = []
    acc_reweigh = []

    ###########################################
    
    for i in range(10):

        ###########################################
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        dataset_orig = load_preproc_data_adult()
        # train1, test1 are the original dataset
        train1, test1 = dataset_orig.split([0.7], shuffle=True)
        RW = Reweighing(unprivileged_groups=unprivileged_groups,
                       privileged_groups=privileged_groups)
        RW.fit(train1)
        # dataset_transf_train, test1 are for the reweighed dataset
        dataset_transf_train = RW.transform(train1)

        ###########################################

        # change weights to whole numbers
        for i in range(dataset_transf_train.instance_weights.size):
            dataset_transf_train.instance_weights[i] = (round(dataset_transf_train.instance_weights[i] / 0.1) * 0.1) * 10
        weights = copy.deepcopy(dataset_transf_train.instance_weights)

        # change dataset_transf_train.features and dataset_transf_train.labels and dataset_transf_train.protected_attributes according to the weights of each instance
        sum_weights = 0
        for i in range(dataset_transf_train.features.shape[0]):
            row = copy.deepcopy(dataset_transf_train.features[i])
            row_label = copy.deepcopy(dataset_transf_train.labels[i])
            row_protected_attributes = copy.deepcopy(dataset_transf_train.protected_attributes[i])
            row_protected_attributes.resize(1,2)
            row.resize(1,18)
            row_label.resize(1,1)
            weight = int(weights[i])
            for j in range(weight-1):
                dataset_transf_train.features = np.concatenate((dataset_transf_train.features,row))
                dataset_transf_train.labels = np.concatenate((dataset_transf_train.labels,row_label))
                dataset_transf_train.protected_attributes = np.concatenate((dataset_transf_train.protected_attributes,row_protected_attributes))

        # change the dataset_transf_train to a numpy array of ones to match number of rows in features
        dataset_transf_train.instance_weights = np.ones(dataset_transf_train.features.shape[0])

        ################## without reweighing ##########################

        temp_stat_par = []
        temp_disp_imp = []
        temp_eq_opp_diff = []
        temp_avg_odds_diff = []
        temp_theil = []
        temp_acc = []

        ##################### adversarial debiasing #####################
        sess = tf.Session()
        debiased_model = AdversarialDebiasing(privileged_groups = privileged_groups,
                                  unprivileged_groups = unprivileged_groups,
                                  scope_name='debiased_classifier',
                                  debias=True,
                                  sess=sess)
        debiased_model.fit(train1)
        dataset_debiasing_test = debiased_model.predict(test1)
        sess.close()
        tf.reset_default_graph()

        ##################### metrics #####################

        metric_test = BinaryLabelDatasetMetric(dataset_debiasing_test, 
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
        acc_test = ClassificationMetric(test1,
                                        dataset_debiasing_test,
                                        unprivileged_groups=unprivileged_groups,
                                        privileged_groups=privileged_groups)
        temp_stat_par.append(metric_test.mean_difference())
        temp_disp_imp.append(metric_test.disparate_impact())
        temp_eq_opp_diff.append(metric_test.equal_opportunity_difference())
        temp_avg_odds_diff.append(metric_test.average_odds_difference())
        temp_theil.append(metric_test.theil_index())
        temp_acc.append(acc_test.accuracy())

        ##################### prejudice remover #####################

        prejudice_model = PrejudiceRemover(eta=100, sensitive_attr='sex')
        prejudice_model.fit(train1)
        dataset_prejudice_test = prejudice_model.predict(test1)

        ##################### metrics #####################

        metric_test = BinaryLabelDatasetMetric(dataset_prejudice_test, 
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
        acc_test = ClassificationMetric(test1,
                                        dataset_prejudice_test,
                                        unprivileged_groups=unprivileged_groups,
                                        privileged_groups=privileged_groups)
        temp_stat_par.append(metric_test.mean_difference())
        temp_disp_imp.append(metric_test.disparate_impact())
        temp_eq_opp_diff.append(metric_test.equal_opportunity_difference())
        temp_avg_odds_diff.append(metric_test.average_odds_difference())
        temp_theil.append(metric_test.theil_index())
        temp_acc.append(acc_test.accuracy())

        ##################### normal neural net #####################

        sess = tf.Session()
        neural_model = AdversarialDebiasing(privileged_groups = privileged_groups,
                                  unprivileged_groups = unprivileged_groups,
                                  scope_name='debiased_classifier',
                                  debias=False,
                                  sess=sess)
        neural_model.fit(train1)
        dataset_neural_test = neural_model.predict(test1)
        sess.close()
        tf.reset_default_graph()

        ##################### metrics #####################

        metric_test = BinaryLabelDatasetMetric(dataset_neural_test, 
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
        acc_test = ClassificationMetric(test1,
                                        dataset_neural_test,
                                        unprivileged_groups=unprivileged_groups,
                                        privileged_groups=privileged_groups)
        temp_stat_par.append(metric_test.mean_difference())
        temp_disp_imp.append(metric_test.disparate_impact())
        temp_eq_opp_diff.append(metric_test.equal_opportunity_difference())
        temp_avg_odds_diff.append(metric_test.average_odds_difference())
        temp_theil.append(metric_test.theil_index())
        temp_acc.append(acc_test.accuracy())

        ##################### ensemble #####################

        pred_labels_test = []
        for i in range(0, len(test1.features)):
            arr_test = mode([dataset_debiasing_test[i], dataset_prejudice_test[i], dataset_neural_test[i]])
            pred_labels_test.append(arr_test[0][0])
        dataset_ensemble_test = test1.copy()
        dataset_ensemble_test.labels = np.array(pred_labels_test)

        ##################### metrics #####################

        metric_test = BinaryLabelDatasetMetric(dataset_ensemble_train, 
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
        acc_test = ClassificationMetric(test1,
                                        dataset_ensemble_train,
                                        unprivileged_groups=unprivileged_groups,
                                        privileged_groups=privileged_groups)
        temp_stat_par.append(metric_test.mean_difference())
        temp_disp_imp.append(metric_test.disparate_impact())
        temp_eq_opp_diff.append(metric_test.equal_opportunity_difference())
        temp_avg_odds_diff.append(metric_test.average_odds_difference())
        temp_theil.append(metric_test.theil_index())
        temp_acc.append(acc_test.accuracy())

        ######### DUMP SHIT ###########

        stat_par.append(temp_stat_par)
        disp_imp.append(temp_disp_imp)
        eq_opp_diff.append(temp_eq_opp_diff)
        avg_odds_diff.append(temp_avg_odds_diff)
        theil.append(temp_theil)
        acc.append(temp_acc)

        ################## with reweighing ##########################

        temp_stat_par = []
        temp_disp_imp = []
        temp_eq_opp_diff = []
        temp_avg_odds_diff = []
        temp_theil = []
        temp_acc = []

        ################## adversarial debiasing ##################
        sess = tf.Session()
        debiased_model_reweighing = AdversarialDebiasing(privileged_groups = privileged_groups,
                                  unprivileged_groups = unprivileged_groups,
                                  scope_name='debiased_classifier',
                                  debias=True,
                                  sess=sess)
        debiased_model_reweighing.fit(dataset_transf_train)
        dataset_debiasing_test_reweighing = debiased_model_reweighing.predict(test1)
        sess.close()
        tf.reset_default_graph()

        ##################### metrics #####################

        metric_test = BinaryLabelDatasetMetric(dataset_debiasing_test_reweighing, 
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
        acc_test = ClassificationMetric(test1,
                                        dataset_debiasing_test_reweighing,
                                        unprivileged_groups=unprivileged_groups,
                                        privileged_groups=privileged_groups)
        temp_stat_par.append(metric_test.mean_difference())
        temp_disp_imp.append(metric_test.disparate_impact())
        temp_eq_opp_diff.append(metric_test.equal_opportunity_difference())
        temp_avg_odds_diff.append(metric_test.average_odds_difference())
        temp_theil.append(metric_test.theil_index())
        temp_acc.append(acc_test.accuracy())

        ##################### prejudice remover #####################
        prejudice_model_reweighing = PrejudiceRemover(eta=100, sensitive_attr='sex')
        prejudice_model_reweighing.fit(dataset_transf_train)
        dataset_prejudice_test_reweighing = prejudice_model_reweighing.predict(test1)

        ##################### metrics #####################

        metric_test = BinaryLabelDatasetMetric(dataset_prejudice_test_reweighing, 
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
        acc_test = ClassificationMetric(test1,
                                        dataset_prejudice_test_reweighing,
                                        unprivileged_groups=unprivileged_groups,
                                        privileged_groups=privileged_groups)
        temp_stat_par.append(metric_test.mean_difference())
        temp_disp_imp.append(metric_test.disparate_impact())
        temp_eq_opp_diff.append(metric_test.equal_opportunity_difference())
        temp_avg_odds_diff.append(metric_test.average_odds_difference())
        temp_theil.append(metric_test.theil_index())
        temp_acc.append(acc_test.accuracy())

        ##################### normal neural net #####################
        sess = tf.Session()
        neural_model = AdversarialDebiasing(privileged_groups = privileged_groups,
                                  unprivileged_groups = unprivileged_groups,
                                  scope_name='debiased_classifier',
                                  debias=False,
                                  sess=sess)
        neural_model.fit(dataset_transf_train)
        dataset_neural_test = neural_model.predict(test1)
        sess.close()
        tf.reset_default_graph()

        ##################### metrics #####################

        metric_test = BinaryLabelDatasetMetric(dataset_neural_test, 
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
        acc_test = ClassificationMetric(test1,
                                        dataset_neural_test,
                                        unprivileged_groups=unprivileged_groups,
                                        privileged_groups=privileged_groups)
        temp_stat_par.append(metric_test.mean_difference())
        temp_disp_imp.append(metric_test.disparate_impact())
        temp_eq_opp_diff.append(metric_test.equal_opportunity_difference())
        temp_avg_odds_diff.append(metric_test.average_odds_difference())
        temp_theil.append(metric_test.theil_index())
        temp_acc.append(acc_test.accuracy())

        ##################### ensemble #####################
        pred_labels_test = []
        for i in range(0, len(test1.features)):
            arr_test = mode([dataset_debiasing_test[i], dataset_prejudice_test[i], dataset_neural_test[i]])
            pred_labels_test.append(arr_test[0][0])
        dataset_ensemble_test = test1.copy()
        dataset_ensemble_test.labels = np.array(pred_labels_test)

        ##################### metrics #####################

        metric_test = BinaryLabelDatasetMetric(dataset_ensemble_test, 
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
        acc_test = ClassificationMetric(test1,
                                        dataset_ensemble_test,
                                        unprivileged_groups=unprivileged_groups,
                                        privileged_groups=privileged_groups)
        temp_stat_par.append(metric_test.mean_difference())
        temp_disp_imp.append(metric_test.disparate_impact())
        temp_eq_opp_diff.append(metric_test.equal_opportunity_difference())
        temp_avg_odds_diff.append(metric_test.average_odds_difference())
        temp_theil.append(metric_test.theil_index())
        temp_acc.append(acc_test.accuracy())

        ######### DUMP SHIT ###########

        stat_par_reweigh.append(temp_stat_par)
        disp_imp_reweigh.append(temp_disp_imp)
        eq_opp_diff_reweigh.append(temp_eq_opp_diff)
        avg_odds_diff_reweigh.append(temp_avg_odds_diff)
        theil_reweigh.append(temp_theil)
        acc_reweigh.append(temp_acc)
        
    without_reweighing = [stat_par,disp_imp,eq_opp_diff,avg_odds_diff,theil,acc]
    with_reweighing = [stat_par_reweigh,disp_imp_reweigh,eq_opp_diff_reweigh,avg_odds_diff_reweigh,theil_reweigh,acc_reweigh]
        
    for metric in range(len(without_reweighing)):
        name = "metric" + str(metric)
        sublist = without_reweighing[metric]
        with open(name, "wb") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(sublist)
            
    for metric in range(len(with_reweighing)):
        name = "metric" + str(metric) + "reweigh"
        sublist = with_reweighing[metric]
        with open(name, "wb") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(sublist)

            
            
            
            
            