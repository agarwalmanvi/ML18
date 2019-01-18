from aif360.algorithms.inprocessing import PrejudiceRemover

# apply prejudice remover algorithm
def prejudice(train, test):
    prejudice_model = PrejudiceRemover(eta=100, sensitive_attr='sex')
    prejudice_model.fit(train)

    # predict outcome using the test set
    test_prejudice = prejudice_model.predict(test)

    return test_prejudice