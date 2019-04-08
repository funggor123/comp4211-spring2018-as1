from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import dataloader
import numpy as np
import time


def train(X, Y):
    clf = SGDClassifier(loss="log", max_iter=5000, tol=0.000000001)
    '''
        Use Grid Search to find out the suitable step size for logistic regression
    '''
    parameters = {'alpha': 10.0 ** -np.arange(1, 10)}
    clf = GridSearchCV(clf, parameters, cv=5)
    reg = clf.fit(X, Y)
    print("^^^^^^^Training Parameters^^^^^^^")
    print("Best Alpha: ", clf.best_params_['alpha'])
    print("tol: ", 0.000000001)
    print("max_iteration: ", 5000)
    return reg


def data_set_by_logistic_regression(table_trainX, table_trainY, table_testX, table_testY, name_classifier):
    reg = train(table_trainX, table_trainY)
    train_predicted = reg.predict(table_trainX)
    test_predicted = reg.predict(table_testX)
    '''
        Results and Visualize
    '''
    ac_train = accuracy_score(table_trainY, train_predicted)
    ac_test = accuracy_score(table_testY, test_predicted)

    rc_train = recall_score(table_trainY, train_predicted)
    rc_test = recall_score(table_testY, test_predicted)

    roc_train = roc_auc_score(table_trainY, reg.predict_proba(table_trainX)[:, 1])
    roc_test = roc_auc_score(table_testY, reg.predict_proba(table_testX)[:, 1])

    cm_train = confusion_matrix(table_trainY, train_predicted)
    cm_test = confusion_matrix(table_testY, test_predicted)

    print("^^^^^^^Training Results^^^^^^^^^")
    print("Accuracy score of training set : ", ac_train)
    print("Accuracy score of test set : ", ac_test)

    print("Recall score of training set : ", rc_train)
    print("Recall score of test set : ", rc_test)

    print("AUC score of training set : ", roc_train)
    print("AUC score of test set : ", roc_test)

    print("Confusion Matrix of training set : ", cm_train)
    print("Confusion Matrix of test set : ", cm_test)

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
    preds = reg.predict_proba(table_trainX)[:, 1]
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
    preds = reg.predict_proba(table_testX)[:, 1]
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
    print("*********Logistic Regression**********")
    fifa_dataset, finance_dataset, orbits_dataset = dataloader.load_data()
    print("-----------------fifa-----------------")
    train_x, test_x, c_train_y, c_test_y, _, _ = dataloader.dataset_to_table(fifa_dataset)
    data_set_by_logistic_regression(train_x, c_train_y,
                                    test_x, c_test_y, "fifa")

    print("----------------finance---------------")
    train_x, test_x, c_train_y, c_test_y, _, _ = dataloader.dataset_to_table(finance_dataset)
    data_set_by_logistic_regression(train_x, c_train_y,
                                    test_x, c_test_y, "finance")

    print("-----------------orbits---------------")
    train_x, test_x, c_train_y, c_test_y, _, _ = dataloader.dataset_to_table(orbits_dataset)
    data_set_by_logistic_regression(train_x, c_train_y,
                                    test_x, c_test_y, "orbits")
    print("------------------------------------")
    elapsed_time = time.time() - start_time
    print(elapsed_time, " seconds to complete the task")
