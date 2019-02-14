import numpy as np

def output_rates(input_data, output_data, attribute_name, privileged=None, unprivileged=None, favourable=None, unfavourable=None):

    index_attribute = input_data.feature_names.index(attribute_name)
    privileged = float(privileged)
    unprivileged = float(unprivileged)
    
    input_priv = input_data.labels[np.where(input_data.features[:,index_attribute] == privileged)]
    input_priv = np.reshape(input_priv, (input_priv.shape[0],1))
    output_priv = output_data.labels[np.where(output_data.features[:,index_attribute] == privileged)]
    output_priv = np.reshape(output_priv, (output_priv.shape[0],1))
    priv_labels = np.concatenate((input_priv, output_priv), axis=1)
    
    input_unpriv = input_data.labels[np.where(input_data.features[:,index_attribute] == unprivileged)]
    input_unpriv = np.reshape(input_unpriv, (input_unpriv.shape[0],1))
    output_unpriv = output_data.labels[np.where(output_data.features[:,index_attribute] == unprivileged)]
    output_unpriv = np.reshape(output_unpriv, (output_unpriv.shape[0],1))
    unpriv_labels = np.concatenate((input_unpriv, output_unpriv), axis=1)
    
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    for i in range(priv_labels.shape[0]):
        input_label = priv_labels[i][0]
        output_label = priv_labels[i][1]
        if input_label == output_label:
            if input_label == unfavourable:
                tn = tn + 1
            else:
                tp = tp + 1
        else:
            if input_label == favourable and output_label == unfavourable:
                fn = fn + 1
            else:
                fp = fp + 1
    
    rates_privileged = [tp,fp,tn,fn]
    
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    for i in range(unpriv_labels.shape[0]):
        input_label = unpriv_labels[i][0]
        output_label = unpriv_labels[i][1]
        if input_label == output_label:
            if input_label == unfavourable:
                tn = tn + 1
            else:
                tp = tp + 1
        else:
            if input_label == favourable and output_label == unfavourable:
                fn = fn + 1
            else:
                fp = fp + 1
                
    rates_unprivileged = [tp,fp,tn,fn]  
    
    rates_list = [rates_privileged, rates_unprivileged]
    
    return rates_list

def equal_opp_diff(input_data, output_data, attribute_name, privileged, unprivileged, favourable, unfavourable):
    rates_both = output_rates(input_data, output_data, attribute_name, privileged, unprivileged, favourable, unfavourable)
    
    # [tp, fp, tn, fn]
    outcome_privileged = rates_both[0]
    outcome_unprivileged = rates_both[1]
    
    # true positive rate = tp / (tp + fn)
    tpr_privileged = outcome_privileged[0] / (outcome_privileged[0] + outcome_privileged[3])
    tpr_unprivileged = outcome_unprivileged[0] / (outcome_unprivileged[0] + outcome_unprivileged[3])

    equal_opportunity_difference = tpr_unprivileged - tpr_privileged
    
    return equal_opportunity_difference

def avg_odds_diff(input_data, output_data, attribute_name, privileged, unprivileged, favourable, unfavourable):
    rates_both = output_rates(input_data, output_data, attribute_name, privileged, unprivileged, favourable, unfavourable)
    
    # [tp, fp, tn, fn]
    outcome_privileged = rates_both[0]
    outcome_unprivileged = rates_both[1]
    
    # true positive rate = tp / (tp + fn)
    tpr_privileged = outcome_privileged[0] / (outcome_privileged[0] + outcome_privileged[3])
    tpr_unprivileged = outcome_unprivileged[0] / (outcome_unprivileged[0] + outcome_unprivileged[3])

    # false positive rate = fp / (fp + tn)
    fpr_privileged = outcome_privileged[1] / (outcome_privileged[1] + outcome_privileged[2])
    fpr_unprivileged = outcome_unprivileged[1] / (outcome_unprivileged[1] + outcome_unprivileged[2])
    
    fpr_diff = fpr_unprivileged - fpr_privileged
    tpr_diff = tpr_unprivileged - tpr_privileged
    
    average_odds_difference = (fpr_diff + tpr_diff) * 0.5
    
    return average_odds_difference
