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
        # Updating theta using all the examples in one iterations
        theta -= learning_rate * delta
        j_history.append(compute_cost(x_train, y_train, theta))

    return theta, np.array(j_history)


def feature_normalization(x_train):
    """
    feature_normalization Normalizes the features in X_train
    feature_normalization(X) returns a normalized version of X_train
    where the mean value of each feature is 0 and the
    standard deviation is 1. This is often a good preprocessing
    step to do when working with learning algorithms.
    Return 3 values - X_norm(main objective of the function),
    sample_mean and sample_std so that we can use this value to
    normalize X_test in order to process the test case.
    """

    # Computing mean, standard deviation  and normalized value of X_train.
    # Note that in 'np.std' function parameter 'ddof=1' is added to compute
    # standard  deviation of sample.
    # Without this parameter 'np.std' compute population standard  deviation.
    sample_mean = np.mean(x_train, axis=0).reshape(1, np.size(x_train, 1))
    sample_std = np.std(x_train, axis=0, ddof=1).reshape(1, np.size(x_train, 1))
    x_norm = (x_train - sample_mean) / sample_std

    return x_norm, sample_mean, sample_std


def main():
    """
    This is the main function
    """

    # Read data from a .txt file that is separated by comma(,)
    data = pd.read_csv("ex1data2.txt", sep=",", header=None)
    # data = pd.read_csv("ds5_train.csv", sep=",", header=0)

    # Number of training examples and features in the file
    num_ex = np.size(data.iloc[:, 1], 0)
    feature = np.size(data.iloc[1, :], 0)

    # After separating made data type Array and reshaped variables
    x_train = data.iloc[:, 0 : (feature - 1)].to_numpy().reshape(num_ex, (feature - 1))
    y_train = data.iloc[:, (feature - 1)].to_numpy().reshape(num_ex, 1)

    # Computing mean, std and normalized value of X_train
    x_train, _, _ = feature_normalization(x_train)
    # Added a new coloum of 'Ones' vector to X_train variable
    x_train = np.hstack((np.ones((num_ex, 1)), x_train))
    # Initializing theta -> all zero for feeding to gradient descent
    theta = np.zeros((feature, 1))

    # Setting up total num of iter and learning rate(alpha)
    iterations = 400
    learning_rate = 0.01

    # Computing final theta and cost function value in every iteration
    theta, j_history = gradient_descent(
        x_train, y_train, theta, learning_rate, iterations
    )
    print(theta)

    # Make a array([[1], [2], [3], ....., [iterations]])
    iter_range = np.linspace(1, iterations, num=iterations).reshape(iterations, 1)

    # Cost funtions vs iteration. Comment to see the previous graph
    plt.figure(num="Cost Function")
    plt.plot(iter_range, j_history)
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost function value for each iteration")
    plt.title("Multivariate Linear Regression Cost Function Graph")
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
