# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 17:43:10 2020

@author: safayet_khan
"""

# Import libraries including the 3rd party libraries
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np

# qt5(Pop up graph) and inline(Embedded graph).
# Run one time and comment the line again
# %matplotlib qt5


def compute_cost(x_train, y_train, theta, lamda):
    """
    Compute cost function of in each iteration of gradient descent
    Return cost function value for each iteration
    """
    # Number of training examples in the file
    num_ex = np.size(y_train, 0)

    # Compute the value of hypothesis function
    hypothesis = sigmoid(np.matmul(x_train, theta))

    # Vectorized implementation to compute cost function value
    j_cost = (-1 / num_ex) * (
        np.matmul(y_train.T, np.log(hypothesis))
        + np.matmul((1 - y_train.T), np.log(1 - hypothesis))
    )

    # Adding regularization to cost function value
    j_cost = j_cost + (lamda / (2 * num_ex)) * np.matmul(theta[1:].T, theta[1:])

    # Computing the gradient with regularization
    grad = (1 / num_ex) * (np.matmul(x_train.T, (hypothesis - y_train)))
    grad[1:] = grad[1:] + ((lamda / num_ex) * theta[1:])

    return j_cost, grad


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
    acc = (
        np.mean((np.argmax(pred, axis=1).reshape(5000, 1) == y_train).astype(float))
        * 100
    )
    return acc


def gradient_descent(x_train, y_train, theta, learning_rate, num_iter, lamda):
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

        # compute the value of hypothesis function
        hypothesis = sigmoid(np.matmul(x_train, theta))

        # Vectorized implementation of batch gradient descent with
        # regularization to prevent overfitting
        delta = (1 / num_ex) * (np.matmul(x_train.T, (hypothesis - y_train)))
        delta[1:] = delta[1:] + np.multiply((lamda / num_ex), theta[1:])

        # Updating theta using all the examples in one iterations
        theta -= learning_rate * delta

        # Re-writing J_history in every iterations
        j_history[each_iter], _ = compute_cost(x_train, y_train, theta, lamda)

    return theta, j_history


def plot_cost_function(j_history, iterations, class_label):
    """
    Plot a graph of cost function (j_history vs iterations)
    """
    # Create figure (will only create new window if needed)
    plt.figure(num="Cost Function Visualization")

    # Make a array([[1], [2], [3], ....., [iterations]])
    iter_range = np.linspace(1, iterations, num=iterations).reshape(iterations, 1)

    # Cost funtions vs iteration. Comment to see the previous graph
    # fmt:off
    plt.plot(iter_range, j_history,
             label="Class {} vs All".format(class_label), linewidth=2)
    # fmt:on

    # Label x-axis and y-axis
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost function value for each iteration")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.title("Cost Function for Multiclass Logistic Regression")

    # Show the plot in non-blocking mode
    plt.show(block=False)


def main():
    """
    This is the main function
    """
    # Read data from .mat file. Return type is 'dict { }'
    data = loadmat("ex3data1.mat")

    # After separating made data type Array and reshaped variables
    x_train = data["X"]
    y_train = data["y"]

    # Set the zero digit to 0, because its mapped 10 in this dataset
    y_train[y_train == 10] = 0

    # Number of training examples, features and labels in the file
    num_ex = np.size(x_train, 0)
    x_train = np.hstack((np.ones((num_ex, 1)), x_train))
    feature = np.size(x_train, 1)  # 20x20 Input Images of Digits
    num_labels = 10  # 10 labels, from 0 to 9

    # Initializing theta -> all zero for feeding to gradient descent
    all_theta = np.zeros((num_labels, feature))
    initial_theta = np.zeros((feature, 1))

    # Setting up total num of iter and learning rate(alpha)
    iterations = 2500
    learning_rate = 0.5
    lamda = 0.1

    # Computing final theta and cost function value in every iteration
    for i in range(num_labels):
        initial_theta, j_history = gradient_descent(
            x_train,
            (y_train == i).astype(int),
            initial_theta,
            learning_rate,
            iterations,
            lamda,
        )

        # Save initial theta to all theta
        all_theta[i, :] = initial_theta.T

        # Plot cost function for every labels
        plot_cost_function(j_history, iterations, class_label=i)

        # Initialize theta to zero again
        initial_theta = np.zeros((feature, 1))

    # Compute the accuracy
    acc = accuracy(y_train, predict(x_train, all_theta.T))
    print("\nValue of accuracy: ", acc)


if __name__ == "__main__":
    # Calling the main function
    main()
