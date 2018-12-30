from aif360.algorithms.inprocessing import AdversarialDebiasing

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# apply classification algorithm without debiasing
def nondebiased_classifier(train, test, privileged_groups, unprivileged_groups):
    sess = tf.Session()
    NN_model = AdversarialDebiasing(privileged_groups, unprivileged_groups, scope_name='nondebiased_classifier', debias=False, sess=sess)
    NN_model.fit(train)

    # predict outcome using the test set
    test_NNmodel = NN_model.predict(test)
    tf.reset_default_graph()

    return test_NNmodel