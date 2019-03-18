from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import dataloader


def find_best_num_of_hidden_units(X, Y):
    print("--------------K-Fold Training-------------------")
    scores = []
    max_scores = 0
    max_h = 1
    for i in range(1, 11):
        '''
        Find the best learning rate first
        '''
        clf = MLPClassifier(hidden_layer_sizes=(i,), max_iter=5000, solver='sgd', tol=0.000000001)
        parameters = {'alpha': 10.0 ** -np.arange(1, 10)}
        clf = GridSearchCV(clf, parameters, n_jobs=-1, cv=3, verbose=True)
        clf.fit(X, Y)
        clf = clf.best_estimator_
        '''
        Cross validation to find out the mean score
        '''
        score = np.mean(cross_val_score(clf, X, Y, cv=5))
        scores.append(score)
        print("Hidden Layer:", i, " scores = ", score)
        if score > max_scores:
            max_h = i
            max_scores = score
    print("The best number of hidden units used is ", max_h)
    print("--------------------------------------------------")
    return MLPClassifier(hidden_layer_sizes=(max_h,), tol=0.000000001)


def data_set_by_neural_network(table_trainX, table_trainY, table_testX, table_testY):
    mlp_classifier = find_best_num_of_hidden_units(table_trainX, table_trainY)
    '''
        Training avoid over fitting
    '''
    print("--- Training ---")
    iterations = [200, 500, 1000, 1500, 2000, 2500, 3000]
    best_test_set_accuracy = 0
    training_set_accruacy = 0
    for i in iterations:
        mlp_classifier.max_iter = i
        mlp_classifier.fit(table_trainX, table_trainY)
        ac_train = accuracy_score(mlp_classifier.predict(table_trainX), table_trainY)
        ac_test = accuracy_score(mlp_classifier.predict(table_testX), table_testY)
        print("Accuracy of training set with iterations: ", i, ac_train)
        print("Accuracy of test set with iterations: ", i, ac_test)
        if ac_test > best_test_set_accuracy:
            training_set_accruacy = ac_train
            best_test_set_accuracy = ac_test

    '''
        Results and Visualize
    '''

    print("Accuracy of training set : ", training_set_accruacy)
    print("Accuracy of test set : ", best_test_set_accuracy)


fifa_dataset, finance_dataset, orbits_dataset = dataloader.load_data()
train_x, test_x, c_train_y, c_test_y, _, _ = dataloader.dataset_to_table(fifa_dataset)
data_set_by_neural_network(train_x, c_train_y,
                           test_x, c_test_y)
