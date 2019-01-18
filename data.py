from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
from aif360.datasets import AdultDataset
import pandas as pd
import csv
import sys
# from custom_preprocessing import custom_preprocessing

adult_data = pd.read_csv('dataset/adult.csv')

adult_data['Age'] = adult_data['age'].apply(lambda x: x//100)
adult_data['Age'] = adult_data['Age'].apply(lambda x: '>=70' if x >= 70 else x)

adult_data['Education_years'] = adult_data['education.num'].apply(lambda x: '<6' if x <= 5 else ('>12' if x >= 13 else x))
adult_data['Education_years'] = adult_data['Education_years'].astype('category')

adult_data['sex'] = adult_data['sex'].replace({'Female': 0.0, 'Male': 1.0})
adult_data['race'] = adult_data['race'].apply(lambda x: 1.0 if x == 'White' else 0.0)

protected_attribute = ['sex', 'race']
label_name = 'income'
categorical_features = ['Age', 'Education_years']
features = categorical_features + [label_name] + protected_attribute

privileged_class = {'sex': [1.0], 'race': [1.0]}
protected_attribute_map = {'sex': {1.0: 'Male', 0.0: 'Female'},
                           'race': {1.0: 'White', 0.0: 'Non-white'}}

preprocessed_data = StandardDataset(adult_data, label_name, favorable_classes=['>50K', '>50K.'],
                                    protected_attribute_names=protected_attribute,
                                    privileged_classes=[privileged_class[x] for x in protected_attribute],
                                    categorical_features=categorical_features,
                                    features_to_keep=features,
                                    metadata={'label_map': [{1.0: '>50K', 0.0: '<=50K'}],
                                    'protected_attribute_map': [protected_attribute_map[x] for x in protected_attribute]})

# print(preprocessed_data)

print(preprocessed_data.unfavorable_label)