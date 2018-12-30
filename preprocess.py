from aif360.datasets import StandardDataset
import pandas as pd
import numpy as np

def preprocess_adultdataset(df):
    df['Age'] = df['age'].apply(lambda x: x//100)
    df['Age'] = df['Age'].apply(lambda x: '>=70' if x >= 70 else x)

    df['Education_years'] = df['education.num'].apply(lambda x: '<6' if x <= 5 else ('>12' if x >= 13 else x))
    df['Education_years'] = df['Education_years'].astype('category')

    df['sex'] = df['sex'].replace({'Female': 0.0, 'Male': 1.0})
    df['race'] = df['race'].apply(lambda x: 1.0 if x == 'White' else 0.0)

    protected_attribute = ['sex', 'race']
    label_name = 'income'
    categorical_features = ['Age', 'Education_years']
    features = categorical_features + [label_name] + protected_attribute

    privileged_class = {'sex': [1.0], 'race': [1.0]}
    protected_attribute_map = {'sex': {1.0: 'Male', 0.0: 'Female'},
                            'race': {1.0: 'White', 0.0: 'Non-white'}}

    data = StandardDataset(df, label_name, favorable_classes=['>50K', '>50K.'],
                            protected_attribute_names=protected_attribute,
                            privileged_classes=[privileged_class[x] for x in protected_attribute],
                            categorical_features=categorical_features,
                            features_to_keep=features,
                            metadata={'label_map': [{1.0: '>50K', 0.0: '<=50K'}],
                                      'protected_attribute_map': [protected_attribute_map[x] for x in protected_attribute]})
    return data


def preprocess_germandataset(df):
    def group_credit_hist(x):
        if x in ['no credits taken/ all credits paid back duly', 'all credits at this bank paid back duly', 'existing credits paid back duly till now']:
            return 'None/Paid'
        elif x == 'delay in paying off in the past':
            return 'Delay'
        elif x == 'critical account/ other credits existing (not at this bank)':
            return 'Other'
        else:
            return 'NA'

    def group_employ(x):
        if x == 'unemployed':
            return 'Unemployed'
        elif x in ['... < 1 year ', '1 <= ... < 4 years']:
            return '1-4 years'
        elif x in ['4 <= ... < 7 years', '.. >= 7 years']:
            return '4+ years'
        else:
            return 'NA'

    def group_savings(x):
        if x in ['... < 100 DM', '100 <= ... < 500 DM']:
            return '<500'
        elif x in ['500 <= ... < 1000 DM ', '.. >= 1000 DM ']:
            return '500+'
        elif x == 'unknown/ no savings account':
            return 'Unknown/None'
        else:
            return 'NA'

    def group_status(x):
        if x in ['< 0 DM', '0 <= ... < 200 DM']:
            return '<200'
        elif x in ['>= 200 DM / salary assignments for at least 1 year']:
            return '200+'
        elif x == 'no checking account':
            return 'None'
        else:
            return 'NA'

    status_map = {'male : divorced/separated': 1.0,
                'male : single': 1.0,
                'male : married/widowed': 1.0,
                'female : divorced/separated/married': 0.0,
                'female : single': 0.0}

    df['personal_status_sex'] = df['personal_status_sex'].replace(status_map)
    df['credit_history'] = df['credit_history'].apply(lambda x: group_credit_hist(x))
    df['savings'] = df['savings'].apply(lambda x: group_savings(x))
    df['present_emp_since'] = df['present_emp_since'].apply(lambda x: group_employ(x))
    df['age'] = df['age'].apply(lambda x: np.float(x >= 25))
    df['account_check_status'] = df['account_check_status'].apply(lambda x: group_status(x))

    df = df.rename(columns = {'default': 'credit', 'present_emp_since': 'employment', 'account_check_status': 'status', 'personal_status_sex': 'sex'})

    protected_attribute = ['sex', 'age']
    label_name = 'credit'
    categorical_features = ['credit_history', 'savings', 'employment']
    features = categorical_features + [label_name] + protected_attribute

    privileged_class = {'sex': [1.0], 'age': [1.0]}

    protected_attribute_map = {"sex": {1.0: 'male', 0.0: 'female'},
                            "age": {1.0: 'old', 0.0: 'young'}}

    data = StandardDataset(df, label_name, favorable_classes=[1],
                            protected_attribute_names=protected_attribute,
                            privileged_classes=[privileged_class[x] for x in protected_attribute],
                            categorical_features=categorical_features,
                            features_to_keep=features,
                            metadata={'label_maps': [{1.0: 'Good Credit', 2.0: 'Bad Credit'}],
                                    'protected_attribute_maps': [protected_attribute_map[x] for x in protected_attribute]})

    return data

def preprocess_compasdataset(df):
    df = df[['age', 'c_charge_degree', 'race', 'age_cat', 'score_text',
                 'sex', 'priors_count', 'days_b_screening_arrest', 'decile_score',
                 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]

    # Indices of data samples to keep
    ix = df['days_b_screening_arrest'] <= 30
    ix = (df['days_b_screening_arrest'] >= -30) & ix
    ix = (df['is_recid'] != -1) & ix
    ix = (df['c_charge_degree'] != "O") & ix
    ix = (df['score_text'] != 'N/A') & ix
    df = df.loc[ix,:]
    df['length_of_stay'] = (pd.to_datetime(df['c_jail_out']) - pd.to_datetime(df['c_jail_in'])).apply(lambda x: x.days)

    # Restrict races to African-American and Caucasian
    df = df.loc[~df['race'].isin(['Native American','Hispanic','Asian','Other']),:]

    df = df[['sex','race','age_cat','c_charge_degree','score_text','priors_count','is_recid', 'two_year_recid','length_of_stay']]

    df['priors_count'] = df['priors_count'].apply(lambda x: 0 if x <= 0 else ('1 to 3' if 1 <= x <= 3 else 'More than 3'))
    df['length_of_stay'] = df['length_of_stay'].apply(lambda x: '<week' if x <= 7 else ('<3months' if 8 < x <= 93 else '>3months'))
    df['score_text'] = df['score_text'].apply(lambda x: 'MediumHigh' if (x == 'High')| (x == 'Medium') else x)
    df['age_cat'] = df['age_cat'].apply(lambda x: '25 to 45' if x == '25 - 45' else x)

    df['sex'] = df['sex'].replace({'Female': 1.0, 'Male': 0.0})
    df['race'] = df['race'].apply(lambda x: 1.0 if x == 'Caucasian' else 0.0)

    df = df[['two_year_recid', 'sex', 'race', 'age_cat', 'priors_count', 'c_charge_degree']]

    protected_attributes = ['sex', 'race']
    label_name = 'two_year_recid'
    categorical_features = ['age_cat', 'priors_count', 'c_charge_degree']
    features = categorical_features + [label_name] + protected_attributes

    # privileged classes
    privileged_classes = {"sex": [1.0], "race": [1.0]}

    # protected attribute maps
    protected_attribute_map = {"sex": {0.0: 'Male', 1.0: 'Female'},
                                "race": {1.0: 'Caucasian', 0.0: 'Not Caucasian'}}


    data = StandardDataset(df, label_name, favorable_classes=[0],
                           protected_attribute_names=protected_attributes,
                           privileged_classes=[privileged_classes[x] for x in protected_attributes],
                           categorical_features=categorical_features,
                           features_to_keep=features,
                           metadata={'label_maps': [{1.0: 'Did recid.', 0.0: 'No recid.'}],
                                     'protected_attribute_maps': [protected_attribute_map[x] for x in protected_attributes]})

    return data



def preprocess_data(df, name):
    if name == 'adult':
        data = preprocess_adultdataset(df)
    if name == 'german':
        data = preprocess_germandataset(df)
    if name == 'compas':
        data = preprocess_compasdataset(df)
    
    return data
