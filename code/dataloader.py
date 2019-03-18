import numpy as np

data_set_path = "./dataset/"
data_set_1_name = "fifa.npz"
data_set_2_name = "finance.npz"
data_set_3_name = "orbits.npz"
show_data = False


def load_data():
    dataset_1 = np.load(data_set_path + data_set_1_name)
    dataset_2 = np.load(data_set_path + data_set_2_name)
    dataset_3 = np.load(data_set_path + data_set_3_name)
    return dataset_1, dataset_2, dataset_3


def dataset_to_table(dataset):
    train_x = dataset['train_X']
    test_x = dataset['test_X']
    c_train_y = dataset['classification_train_Y']
    c_test_y = dataset['classification_test_Y']
    r_train_y = dataset['regression_train_Y']
    r_test_y = dataset['regression_test_Y']
    return train_x, test_x, c_train_y, c_test_y, r_train_y, r_test_y


def show_dataset(dataset):
    train_shape = dataset['train_X']
    test_shape = dataset['test_X']
    classification_train_y = dataset['classification_train_Y']
    classification_test_y = dataset['classification_test_Y']
    regression_train_y = dataset['regression_train_Y']
    regression_test_y = dataset['regression_test_Y']

    show_table(train_shape)
    show_table(test_shape)
    show_table(classification_train_y)
    show_table(classification_test_y)
    show_table(regression_train_y)
    show_table(regression_test_y)


def show_table(table):
    tshape = table.shape
    top_num = 1
    top = table[0:top_num]
    print("The shape of the table:", tshape)
    print("The top items of table:", top)


if show_data:
    fifa_dataset, finance_dataset, orbits_dataset = load_data()

    print("*************Start of showing data*******************")
    print("------------------fifa----------------")
    show_dataset(fifa_dataset)
    print("------------------finance----------------")
    show_dataset(finance_dataset)
    print("------------------orbits----------------")
    show_dataset(orbits_dataset)
    print("*************End of showing data*******************")
