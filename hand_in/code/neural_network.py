from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from warnings import filterwarnings
import numpy as np
import time
import matplotlib.pyplot as plt
import dataloader

filterwarnings('ignore')


def find_best_num_of_hidden_units(X, Y):
    print("--------------K-Fold Training-------------------")
    scores = []
    best_alpha = 0
    max_scores = 0
    max_h = 1

    for i in range(1, 11):
        '''
        Find the best learning rate first
        '''
        clf = MLPClassifier(hidden_layer_sizes=(i,), max_iter=8000, solver='sgd', tol=0.000000001)
        parameters = {'alpha': 10.0 ** -np.arange(1, 10)}
        clf = GridSearchCV(clf, parameters, n_jobs=-1, cv=3, verbose=True)
        reg = clf.fit(X, Y)
        '''
        Cross validation to find out the mean score
        '''
        print("Best Alpha: ", clf.best_params_['alpha'])
        score = np.mean(cross_val_score(reg, X, Y, cv=5, n_jobs=-1))
        scores.append(score)
        print("Hidden Layer:", i, " scores = ", score)
        if score > max_scores:
            best_alpha = clf.best_params_['alpha']
            max_h = i
            max_scores = score
    print("The best number of hidden units used is ", max_h, "with max score : ", max_scores)
    print("with alpha", best_alpha)
    print("--------------------------------------------------")
    best_classifier = MLPClassifier(hidden_layer_sizes=(max_h,), tol=0.000000001)
    best_classifier.alpha = best_alpha
    return best_classifier


def data_set_by_neural_network(table_trainX, table_trainY, table_testX, table_testY, name_classifier):
    mlp_classifier = find_best_num_of_hidden_units(table_testX, table_testY)
    '''
        Training avoid over fitting
    '''
    print("--- Training ---")
    iterations = [500, 1000, 1500, 2000, 2500, 3000,
                  3500, 4000, 4500, 5000, 5500, 6000,
                  6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000]
    best_test_set_loss = 999999999
    best_iteration = 0
    for i in iterations:
        mlp_classifier.max_iter = i
        mlp_classifier.fit(table_testX, table_testY)
        if mlp_classifier.loss_ < best_test_set_loss:
            best_test_set_loss = mlp_classifier.loss_
            best_iteration = i
        else:
            break

    print("Number of iterations before overfit", best_iteration)
    '''
        Results and Visualize
    '''

    mlp_classifier.max_iter = best_iteration
    mlp_classifier.fit(table_testX, table_testY)

    print("^^^^^^^Training Results^^^^^^^^^")
    print("Test set Loss ", mlp_classifier.loss_)

    mlp_classifier.fit(table_trainX, table_trainY)

    '''
        Learning Curve
    '''
    plt.xlabel('Loss')
    plt.ylabel('Iterations')
    plt.plot(mlp_classifier.loss_curve_)
    plt.show()

    '''
        Results and Visualize
    '''
    train_predicted = mlp_classifier.predict(table_trainX)
    test_predicted = mlp_classifier.predict(table_testX)

    ac_train = accuracy_score(train_predicted, table_trainY)
    ac_test = accuracy_score(test_predicted, table_testY)

    print("Accuracy score of training set: ", ac_train)
    print("Accuracy score of test set: ", ac_test)
    print("Training set Loss ", mlp_classifier.loss_)

    '''
        Confusion Matrix Visualization for training set
    '''
    labels = ['postive', 'negative']
    cm = confusion_matrix(table_trainY, train_predicted)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of training set of ' + name_classifier + ' logistic regression')
    fig.colorbar(cax)

    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('Real')
    plt.show()

    '''
        Confusion Matrix Visualization for test set
    '''
    labels = ['postive', 'negative']
    cm = confusion_matrix(table_testY, test_predicted)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of test set of ' + name_classifier + ' logistic regression')
    fig.colorbar(cax)

    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('Real')
    plt.show()

    '''
        AUC Curve Visualization for training set
    '''
    preds = mlp_classifier.predict_proba(table_trainX)[:, 1]
    fpr, tpr, threshold = roc_curve(table_trainY, preds)
    roc_auc = auc(fpr, tpr)
    plt.title('AUC Curve of training set of ' + name_classifier + ' dataset')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    '''
        AUC Curve Visualization for testing set
    '''
    preds = mlp_classifier.predict_proba(table_testX)[:, 1]
    fpr, tpr, threshold = roc_curve(table_testY, preds)
    roc_auc = auc(fpr, tpr)
    plt.title('AUC Curve of test set of ' + name_classifier + ' dataset')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def main():
    start_time = time.time()
    print("*********Single Layer Neural Network**********")
    fifa_dataset, finance_dataset, orbits_dataset = dataloader.load_data()
    print("-----------------fifa-----------------")
    train_x, test_x, c_train_y, c_test_y, _, _ = dataloader.dataset_to_table(fifa_dataset)
    data_set_by_neural_network(train_x, c_train_y,
                               test_x, c_test_y, "fifa")
    print("----------------finance---------------")
    train_x, test_x, c_train_y, c_test_y, _, _ = dataloader.dataset_to_table(finance_dataset)
    data_set_by_neural_network(train_x, c_train_y,
                               test_x, c_test_y, "finance")
    print("-----------------orbits---------------")
    train_x, test_x, c_train_y, c_test_y, _, _ = dataloader.dataset_to_table(orbits_dataset)
    data_set_by_neural_network(train_x, c_train_y,
                               test_x, c_test_y, "orbits")
    print("------------------------------------")
    elapsed_time = time.time() - start_time
    print(elapsed_time, " seconds to complete the task")
