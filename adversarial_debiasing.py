from aif360.algorithms.inprocessing import AdversarialDebiasing

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# apply adversarial debiasing algorithm
def adversarial(train, test, privileged_groups, unprivileged_groups):
    sess = tf.Session()
    adversarial_model = AdversarialDebiasing(privileged_groups, unprivileged_groups, scope_name='debiased_classifier', debias=True, sess=sess)
    adversarial_model.fit(train)

    # predict outcome using the test set
    test_adversarial = adversarial_model.predict(test)

    return test_adversarial
