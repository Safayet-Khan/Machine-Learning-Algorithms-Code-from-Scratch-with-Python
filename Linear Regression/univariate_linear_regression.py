# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 17:19:43 2020

@author: safayet_khan
"""


def compute_cost(x_train, y_train, theta):
    """
    Compute cost function of in each iteration of gradient descent
    Return cost function value for each iteration
    """

    # Number of training examples in the file
    num_ex = np.size(y_train, 0)

    # Vectorized implementation to compute cost function value
    j_cost = (1 / (2 * num_ex)) * np.sum((np.matmul(x_train, theta) - y_train) ** 2)

    return j_cost


def gradient_descent(x_train, y_train, theta, learning_rate, num_iter):
    """
    Gradient descent algorithm
    Minimize(global/local minima) the cost function
    Return 2 values - optimized variable theta and variable J_history
    that contains value for every interations of cost function
    """

    # Number of training examples
    num_ex = np.size(y_train, 0)
    j_history = list()

    # Batch gradient descent algorithm
    for _ in range(num_iter):

        # Vectorized implementation of batch gradient descent
        delta = (1 / num_ex) * np.matmul(
            np.transpose(x_train), (np.matmul(x_train, theta) - y_train)
        )
        # Updating theta using all the examples in every iterations
        theta -= learning_rate * delta
        j_history.append(compute_cost(x_train, y_train, theta))

    return theta, np.array(j_history)


def main():
    """
    This is the main function
    """

    # Read data from a .txt file that is separated by comma(,)
    data = pd.read_csv("ex1data1.txt", sep=",", header=None)
    # data = pd.read_csv("ds5_train.csv", sep=",", header=0)

    # Number of training examples in the file
    num_ex = np.size(data.iloc[:, 1], 0)

    # After separating made data type Array and reshaped variables
    x_train = data.iloc[:, 0].to_numpy().reshape(num_ex, 1)
    y_train = data.iloc[:, 1].to_numpy().reshape(num_ex, 1)

    # Ploting X and Y. 'Hold On' applied automatically
    plt.figure(num="Univariate Linear Regression")
    plt.scatter(x_train, y_train, color="r", marker="*", s=50, label="Trainin Data")
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")

    # Added a new coloum of 'Ones' vector to X_train variable
    x_train = np.hstack((np.ones((num_ex, 1)), x_train))

    # Initializing theta -> all zero for feeding to gradient descent
    theta = np.zeros((x_train.shape[1], 1))

    # Setting up total num of iter and learning rate(alpha)
    iterations = 1500
    learning_rate = 0.01

    # Computing final theta and cost function value in every iteration
    theta, j_history = gradient_descent(
        x_train, y_train, theta, learning_rate, iterations
    )
    print(theta)

    # Plot the final prediction of our model
    plt.plot(x_train[:, 1], np.matmul(x_train, theta), label="Linear Regression")
    plt.legend()
    plt.title("Univariate Linear Regression Fit")
    plt.show()

    # Make an array like ([[1], [2], [3], ....., [iterations]])
    iter_range = np.linspace(1, iterations, num=iterations).reshape(iterations, 1)

    # Cost funtions vs iteration graph
    plt.figure(num="Cost Function")
    plt.plot(iter_range, j_history)
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost function value for each iteration")
    plt.title("Univariate Linear Regression Cost Function")
    plt.show()


if __name__ == "__main__":

    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # qt5(Pop up graph) and inline(Embedded graph).
    # Run one time and comment the line again
    # %matplotlib qt5

    # Calling the main function
    main()
