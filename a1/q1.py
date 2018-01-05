from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
from math import sqrt
import pandas as pd

np.random.seed(0)


def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X, y, features


def normalize_data(X):
    matrix_X_substract_mean = X - np.mean(X, axis=0, keepdims=True)
    matrix_X_std = np.power(np.var(X, axis=0, keepdims=True), 0.5)
    normal_X = matrix_X_substract_mean/matrix_X_std
    return normal_X


def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]
    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        # Plot feature i against y
        plt.scatter(X[:, i], y)
        plt.xlabel(features[i])
        plt.ylabel("House Price")
    plt.tight_layout()
    plt.show()


def fit_regression(X,Y):
    # implement linear regression
    # Remember to use np.linalg.solve instead of inverting!
    # add bias term
    feature_count = X.shape[1]
    X = np.insert(X, feature_count, 1, axis=1)
    # (X^T*X)W = X^T*y
    left = np.dot(X.T, X)
    right = np.dot(X.T, Y)
    # solve w
    return np.linalg.solve(left, right)


def split(X, y, train_split_rate):
    feature_count, data_X_count, train_X_count = \
        X.shape[1], X.shape[0], ceil(X.shape[0] * train_split_rate)
    train_X = []
    test_X = []
    train_y = []
    test_y = []
    training_index = np.random.choice(data_X_count, int(train_X_count),
                                      replace=False)
    for index in range(data_X_count):
        if index in training_index:
            train_X.append(X[index])
            train_y.append(y[index])
        else:
            test_X.append(X[index])
            test_y.append(y[index])

    train_X = np.array(train_X)
    train_y = np.array(train_y)
    test_X = np.array(test_X)
    test_y = np.array(test_y)
    print("train set of X is: " + str(train_X.shape))
    print("train set of y is: " + str(train_y.shape))
    print("test set of X is: " + str(test_X.shape))
    print("test set of y is: " + str(test_y.shape))
    print()
    return train_X, train_y, test_X, test_y


def get_predict_value(w, X):
    feature_count = X.shape[1]
    # add bias term
    X = np.insert(X, feature_count, 1, axis=1)
    return np.dot(X, w)


def mse(predict_value, test_value):
    # print(predict_value, test_value)
    return np.mean(np.power(test_value-predict_value, 2))


def norm1_loss(test_value, predict_value):
    loss = np.linalg.norm(test_value - predict_value)
    return loss


def r_square_coeff(y, y_predict):
    ss_total = np.var(y)*int(y.shape[0])
    ss_res = mse(y, y_predict)*int(y.shape[0])
    return 1 - np.divide(ss_res, ss_total)


def rmse_loss(mse):
    return sqrt(mse)


def main():
    # Load the data
    X, y, features = load_data()
    print("Features: {}".format(features))

    X = normalize_data(X)

    # Split data into train and test
    train_X, train_y, test_X, test_y = split(X, y, 0.8)

    # Data summary
    print("Sample Of X Summary")
    print(pd.DataFrame(X).describe())
    print("----------------------------")
    print("Sample Of y Summary")
    print(pd.DataFrame(y).describe())
    print("----------------------------")
    print("Training Sample Of X Summary")
    print(pd.DataFrame(train_X).describe())
    print("----------------------------")
    print("Training Sample Of y Summary")
    print(pd.DataFrame(train_y).describe())
    print("----------------------------")
    print("Testing Sample Of X Summary")
    print(pd.DataFrame(test_X).describe())
    print("----------------------------")
    print("Testing Sample Of y Summary")
    print(pd.DataFrame(test_y).describe())
    print("----------------------------")
    print()
    # Fit regression model
    w = fit_regression(train_X, train_y)
    features = features.tolist()

    # Add Bias term
    features.insert(X.shape[1], "BIAS")

    # Show tabulate for weights
    data_frame = pd.DataFrame(w, features)
    print(data_frame)
    print()
    print("The sign of 'INDUS' is positive, which is: " + str(w[2]) +
          ", which makes sense because more proportion of non-retail "
          " business acres per town means less bussiness acres per town, "
          "which let business acres become more valued, therefore the retail "
          "housing price will increase ")
    # Compute fitted values, MSE, etc.
    predict_value = get_predict_value(w, test_X)
    print()
    # MSE
    MSE = mse(predict_value, test_y)
    print("MSE of model is: " + str(MSE) + ".")
    print()
    print("Proposal : use norm 1 loss, r square coefficient and RMSE "
          "to measure the error.")
    print()
    # Norm1 loss
    norm_1_loss = norm1_loss(test_y, predict_value)
    print("Norm 1 loss of model is: " + str(norm_1_loss) + ".")
    print()
    # r square coefficient
    r_square = r_square_coeff(test_y, predict_value)
    print("R square coefficient of model is: " + str(r_square))
    print()
    # RMSE losss
    RMSE = rmse_loss(MSE)
    print("RMSE loss is: " + str(RMSE) + ".")
    print()
    print("Most significant feature to predict the price is LSTAT "
          "after normalization, which makes sense because "
          "more % lower status of the population in town means less quality "
          "of population in town, less opportunity that businessmen investment "
          "in that area, less infrastructure and less security in that area "
          "therefore the housing price will decrease in that area.")
    print()

    # print("Most significant feature to predict the price before normalization
    # is NOX,  which makes sense because more NOX in town "
    # "means more pollution in town, and the retail housing
    # price will decrease. ")
    # print()

    # Visualize the features
    visualize(X, y, features)
    indices = np.arange(1, 15)
    plt.plot(indices, w)
    plt.ylabel("Weight")
    plt.xlabel("Indices")
    plt.show()

if __name__ == "__main__":
    main()

