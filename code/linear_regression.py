from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import dataloader
import matplotlib.pyplot as plt


def train(X, Y):
    reg = LinearRegression().fit(X, Y)
    return reg


def data_set_by_linear_regression(table_trainX, table_trainY, table_testX, table_testY, name_classifier):
    reg = train(table_trainX, table_trainY)

    msq_train = mean_squared_error(reg.predict(table_trainX), table_trainY)
    msq_test = mean_squared_error(reg.predict(table_testX), table_testY)

    '''
        Results and Visualize
    '''
    print("^^^^^^^Training Results^^^^^^^^^")
    print("Sum of Squared Error of training set : ", msq_train * table_trainX.shape[0])
    print("Sum of Squared Error of test set : ", msq_test * table_testX.shape[0])

    sq_for_test = (reg.predict(table_testX) - table_testY) ** 2

    print("Mean Square Error of each data point for test set : ", msq_train)
    print("Mean Square Error of each data point for test set : ", msq_test)

    plt.hist(sq_for_test, bins=50)
    plt.xlabel('Square Error')
    plt.title('Square Error Histogram for test set of ' + name_classifier + ' linear regression')
    plt.ylabel('Amounts of data')
    plt.show()


print("*********Linear Regression**********")
fifa_dataset, finance_dataset, orbits_dataset = dataloader.load_data()
print("----------------fifa----------------")
train_x, test_x, c_train_y, c_test_y, _, _ = dataloader.dataset_to_table(fifa_dataset)
data_set_by_linear_regression(train_x, c_train_y,
                              test_x, c_test_y, "fifa")
print("----------------finance-------------")
train_x, test_x, c_train_y, c_test_y, _, _ = dataloader.dataset_to_table(finance_dataset)
data_set_by_linear_regression(train_x, c_train_y,
                              test_x, c_test_y, "finance")
print("-----------------orbits-------------")
train_x, test_x, c_train_y, c_test_y, _, _ = dataloader.dataset_to_table(orbits_dataset)
data_set_by_linear_regression(train_x, c_train_y,
                              test_x, c_test_y, "orbits")
