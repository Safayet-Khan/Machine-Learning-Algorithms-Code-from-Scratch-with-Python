# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 17:19:43 2020

@author: safayet_khan
"""


def normal_equation(x_train, y_train):
    """
    Normal Equation to find the optimized theta.
    Formula, theta = pinv(x.T, x) * x.T * y
    Return value - optimized theta
    """

    # Initializing theta to all zero vector
    theta = np.zeros((np.size(x_train, 1), 1))

    # Computing theta using normal equation
    theta = np.matmul(
        np.linalg.pinv(np.matmul(np.transpose(x_train), x_train)),
        np.matmul(np.transpose(x_train), y_train),
    )

    return theta


def main():
    """
    This is the main function
    """

    # Read data from a .txt file that is separated by comma(,)
    # data = pd.read_csv("ex1data1.txt", sep=",", header=None)
    data = pd.read_csv("ds5_train.csv", sep=",", header=0)

    # Number of training examples and features in the file
    num_ex = np.size(data.iloc[:, 1], 0)
    feature = np.size(data.iloc[1, :], 0)

    # After separating made data type Array and reshaped variables
    x_train = data.iloc[:, 0 : (feature - 1)].to_numpy().reshape(num_ex, (feature - 1))
    y_train = data.iloc[:, (feature - 1)].to_numpy().reshape(num_ex, 1)

    # Added a new coloum of 'Ones' vector to X_train variable
    x_train = np.hstack((np.ones((num_ex, 1)), x_train))

    # Computing optimized theta using the normal_quation function
    theta = normal_equation(x_train, y_train)
    print(theta)


if __name__ == "__main__":

    import pandas as pd
    import numpy as np

    # qt5(Pop up graph) and inline(Embedded graph).
    # Run one time and comment the line again
    # %matplotlib qt5

    # Calling the main function
    main()
