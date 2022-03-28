# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 17:19:43 2020

@author: safayet_khan
"""

# Import libraries including the 3rd party libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# qt5(Pop up graph) and inline(Embedded graph).
# Run one time and comment the line again
# %matplotlib qt5


def compute_cost(x_train, y_train, theta):
    """
    Compute cost function of in each iteration of gradient descent
    Return cost function value for each iteration
    """
    # Number of training examples in the file
    num_ex = np.size(y_train, 0)

    # compute the value of hypothesis function
    hypothesis = sigmoid(np.dot(x_train, theta))

    # Vectorized implementation to compute cost function value
    j_cost = (-1 / num_ex) * (
        np.matmul(y_train.T, np.log(hypothesis))
        + np.matmul((1 - y_train.T), np.log(1 - hypothesis))
    )
    return j_cost


def sigmoid(z_value):
    """
    Return the value of hypothesis function using sigmoid function
    """
    hypothesis = 1 / (1 + np.exp(-z_value))
    return hypothesis


def predict(x_train, theta):
    """
    predict y_train using computed theta
    """
    # Number of training examples in the file
    num_ex = np.size(x_train, 0)

    # Initializing pred variable to Zeros
    pred = np.zeros((num_ex, 1))

    # Compute the prediction using optimized theta and sigmoid function
    # As it returns just 'True' and 'False', we need to convert it
    # into '1' and '0' using '.astype()' method
    pred = (sigmoid(np.matmul(x_train, theta)) >= 0.5).astype(int)

    return pred


def accuracy(y_train, pred):
    """
    Calculate accuracy of the model.
    Return accuracy in precentage
    """
    acc = np.mean((pred == y_train).astype(float)) * 100
    return acc


def gradient_descent(x_train, y_train, theta, learning_rate, num_iter):
    """
    Gradient descent algorithm
    Minimize(global/local minima) the cost function
    Return 2 values - optimized variable theta and variable J_history
    that contains value for every interations of cost function
    """
    # Number of training examples
    num_ex = np.size(y_train, 0)

    # Initializing J_history variable to Zeros
    j_history = np.zeros((num_iter, 1))

    # Batch gradient descent algorithm
    for each_iter in range(num_iter):

        # Compute the value of hypothesis function
        hypothesis = sigmoid(np.dot(x_train, theta))

        # Vectorized implementation of batch gradient descent
        delta = (1 / num_ex) * (np.matmul(x_train.T, (hypothesis - y_train)))

        # Updating theta using all the examples in one iterations
        theta -= learning_rate * delta

        # Re-writing J_history in every iterations
        j_history[each_iter] = compute_cost(x_train, y_train, theta)

    return theta, j_history


def plot_data(x_train, y_train):
    """
    Plot a graph 'Exam 2 score' vs 'Exam 1 score' with the x_train values
    """
    # Create figure (will only create new window if needed)
    plt.figure(num="Data Visulization")

    # Extract the positions of zeros and ones in y_train array
    pos_zeros = list(np.where(y_train == 0))
    pos_ones = list(np.where(y_train == 1))

    # Based on the postion array extract
    # the x_train_zeros and x_train_ones array
    x_train_zeros = x_train[pos_zeros[0], :]
    x_train_ones = x_train[pos_ones[0], :]

    # First plot scatter diagram with x_train_zeros array denoted by 'o'
    # Then plot scatter diagram with x_train_ones array denoted by '+'
    plt.scatter(
        x_train_zeros[:, 0],
        x_train_zeros[:, 1],
        color="r",
        marker="o",
        s=40,
        label="Not admitted",
    )
    plt.scatter(
        x_train_ones[:, 0],
        x_train_ones[:, 1],
        color="g",
        marker="+",
        s=75,
        label="Admitted",
    )

    # Label x-axis and y-axis also add legend
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.title("Data Visulization for Logistic Regression")

    # Show the plot in non-blocking mode
    plt.show(block=False)


def plot_decision_boundary(x_train, y_train, theta):
    """
    Plot a graph 'Exam 2 score' vs 'Exam 1 score' with the x_train values
    Also plot the decision boundary in this graph
    """
    # Create figure (will only create new window if needed)
    plt.figure(num="Decision Boundary Visulization")

    # Extract the positions of zeros and ones in y_train array
    pos_zeros = list(np.where(y_train == 0))
    pos_ones = list(np.where(y_train == 1))

    # Based on the postion array extract
    # the x_train_zeros and x_train_ones array
    x_train_zeros = x_train[pos_zeros[0], 1:3]
    x_train_ones = x_train[pos_ones[0], 1:3]

    # First plot scatter diagram with x_train_zeros array denoted by 'o'
    # Then plot scatter diagram with x_train_ones array denoted by '+'
    plt.scatter(
        x_train_zeros[:, 0],
        x_train_zeros[:, 1],
        color="r",
        marker="o",
        s=40,
        label="Not admitted",
    )
    plt.scatter(
        x_train_ones[:, 0],
        x_train_ones[:, 1],
        color="g",
        marker="+",
        s=75,
        label="Admitted",
    )

    # From max and min values in x_train(specifically from Exam 1 score)
    # compute the Exam 2 score using the optimized theta
    plot_x = np.array([np.min(x_train[:, 1] - 2), np.max(x_train[:, 1] + 2)])
    # fmt:off
    plot_y = np.multiply((-1 / theta[2]), (np.multiply(theta[1],
                                                       plot_x) + theta[0]))
    # fmt:on
    plt.plot(plot_x, plot_y, label="Decision Boundary")

    # Label x-axis and y-axis also add legend
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.title("Decision Boundary Visulization for Logistic Regression")

    # Show the plot in non-blocking mode
    plt.show(block=False)


def plot_cost_function(j_history, iterations):
    """
    Plot a graph of cost function (j_history vs iterations)
    """
    # Create figure (will only create new window if needed)
    plt.figure(num="Cost Function Visualization")

    # Make a array([[1], [2], [3], ....., [iterations]])
    # fmt:off
    iter_range = np.linspace(1, iterations,
                             num=iterations).reshape(iterations, 1)
    # fmt:on

    # Cost funtions vs iteration. Comment to see the previous graph
    plt.plot(iter_range, j_history, linewidth=2)

    # Label x-axis and y-axis
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost function value for each iteration")
    plt.grid(True, alpha=0.25)
    plt.title("Cost Function Visualization for Logistic Regression")

    # Show the plot in non-blocking mode
    plt.show(block=False)


def main():
    """
    This is the main function
    """
    # Read data from a .txt file that is separated by comma(,)
    data = pd.read_csv("ex2data1.txt", sep=",", header=None)

    # Number of training examples and features in the file
    num_ex = np.size(data.iloc[:, 1], 0)
    feature = np.size(data.iloc[1, :], 0)

    # After separating made data type Array and reshaped variables
    # fmt:off
    x_train = data.iloc[:, 0 : (feature - 1)].to_numpy().reshape(num_ex,
                                                                 (feature - 1))
    # fmt:on
    y_train = data.iloc[:, (feature - 1)].to_numpy().reshape(num_ex, 1)

    # Plot a graph with the x_train values
    plot_data(x_train, y_train)

    # Added a new coloum of 'Ones' vector to X_train variable
    x_train = np.hstack((np.ones((num_ex, 1)), x_train))

    # Initializing theta -> all zero for feeding to gradient descent
    theta = np.zeros((feature, 1))

    # Setting up total num of iter and learning rate(alpha)
    iterations = 500000
    learning_rate = 0.001

    # Computing final theta and cost function value in every iteration
    theta, j_history = gradient_descent(
        x_train, y_train, theta, learning_rate, iterations
    )
    print("Value of theta:\n ", theta)

    # Plot decision boundary using the optimized theta
    plot_decision_boundary(x_train, y_train, theta)

    # Plot a graph cost function (j_history vs iterations)
    plot_cost_function(j_history, iterations)

    # Compute the accuracy
    acc = accuracy(y_train, predict(x_train, theta))
    print("\nValue of accuracy: ", acc)

    plt.show()


if __name__ == "__main__":
    # Calling the main function
    main()
