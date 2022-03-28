# -*- coding: utf-8 -*-
"""
Created on Thu Jul 01 17:19:43 2020

@author: safayet_khan
"""

# Import libraries including the 3rd party libraries
import pandas as pd
import matplotlib.pyplot as plt
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
    hypothesis = sigmoid(np.dot(x_train, theta))

    # Vectorized implementation to compute cost function value
    j_cost = (-1 / num_ex) * (
        np.matmul(y_train.T, np.log(hypothesis))
        + np.matmul((1 - y_train.T), np.log(1 - hypothesis))
    )

    # Adding regularization to cost function value
    j_cost = j_cost + (lamda / (2 * num_ex)) * np.matmul(theta[1:].T, theta[1:])

    # Computing the gradient with regularization
    grad = (1 / num_ex) * (np.matmul(x_train.T, (hypothesis - y_train)))
    grad[1:] = grad[1:] + np.multiply((lamda / num_ex), theta[1:])

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
    acc = np.mean((pred == y_train).astype(float)) * 100
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
        hypothesis = sigmoid(np.dot(x_train, theta))

        # Vectorized implementation of batch gradient descent with
        # regularization to prevent overfitting
        delta = (1 / num_ex) * (np.matmul(x_train.T, (hypothesis - y_train)))
        delta[1:] = delta[1:] + np.multiply((lamda / num_ex), theta[1:])

        # Updating theta using all the examples in one iterations
        theta -= learning_rate * delta

        # Re-writing J_history in every iterations
        j_history[each_iter], _ = compute_cost(x_train, y_train, theta, lamda)

    return theta, j_history


def map_feature(x_1, x_2):
    """
    Feature mapping function to polynomial features
    map_feature(X1, X2) maps the two input features to quadratic features
    Returns a new feature array with more features, comprising of
    X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    Note that: Inputs X1, X2 must be the same size
    """
    # Set maximum degree of the polynomial
    degree = 6

    # Initialize to zero. This take care of the dynamic array problem
    count = 0

    # Initialize function output to 1.
    out = np.ones((np.size(x_1), 28))

    # This loop is responsible for producing polynomial features
    for i in range(1, degree + 1):
        for j in range(i + 1):
            count += 1
            out[:, count] = np.multiply(x_1 ** (i - j), x_2 ** j)

    return out


def plot_data(x_train, y_train):
    """
    Plot a graph 'Exam 2 score' vs 'Exam 1 score' with the x_train values
    """
    # Create figure (will only create new window if needed)
    plt.figure()

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
        label="y = 0",
    )
    plt.scatter(
        x_train_ones[:, 0],
        x_train_ones[:, 1],
        color="g",
        marker="+",
        s=75,
        label="y = 1",
    )

    # Label x-axis and y-axis also add legend
    plt.xlabel("Microchip Test 1")
    plt.ylabel("Microchip Test 2")
    plt.grid(True, alpha=0.25)
    plt.legend()

    # Show the plot in non-blocking mode
    plt.show(block=False)


def plot_decision_boundary(x_train, y_train, theta):
    """
    Plot a graph 'Exam 2 score' vs 'Exam 1 score' with the x_train values
    Also plot the decision boundary in this graph
    """
    # Call the plot_data function to plot the data
    plot_data(x_train[:, 1:3], y_train)

    # If it is a linear decision boundary the first condition is applied
    if np.size(x_train, 1) <= 3:

        # From max and min values in x_train(specifically from Exam 1 score)
        # compute the Exam 2 score using the optimized theta
        plot_x = np.array([np.min(x_train[:, 1] - 2), np.max(x_train[:, 1] + 2)])
        plot_y = np.multiply(
            (-1 / theta[2]), (np.multiply(theta[1], plot_x) + theta[0])
        )

        plt.plot(plot_x, plot_y)

    # If it is a non-linear decision boundary the second condition is applied
    else:

        # U_space and V_space correspond to X axis and Y axis
        u_space = np.linspace(-1, 1.5, 50)
        v_space = np.linspace(-1, 1.5, 50)

        # Initializing Z_space with zeros
        z_space = np.zeros((len(u_space), len(v_space)))

        # This loop is responsible for computing Z_space value
        for i, u_i in enumerate(u_space):
            for j, v_j in enumerate(v_space):
                z_space[i, j] = np.matmul(map_feature(u_i, v_j), theta)

        # Transpose the Z_space value
        z_space = z_space.T

        # Plot the contour graph
        plt.contour(u_space, v_space, z_space, levels=0, linewidths=2, colors="b")

    # Show the plot in non-blocking mode
    plt.show(block=False)


def plot_cost_function(j_history, iterations):
    """
    Plot a graph of cost function (j_history vs iterations)
    """
    # Create figure (will only create new window if needed)
    plt.figure(num="Cost Function")

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
    plt.title("Cost Function for Logistic Regression")
    # Show the plot in non-blocking mode
    plt.show(block=False)


def main():
    """
    This is the main function
    """
    # Read data from a .txt file that is separated by comma(,)
    data = pd.read_csv("ex2data2.txt", sep=",", header=None)

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

    # Separate x_train to x_1 and x_2 to feed it to map_feature funtion
    x_1 = x_train[:, 0]
    x_2 = x_train[:, 1]
    x_train = map_feature(x_1, x_2)

    # Initializing theta -> all zero for feeding to gradient descent
    theta = np.zeros((np.size(x_train, 1), 1))

    # Setting up total num of iter and learning rate(alpha)
    iterations = 150000
    learning_rate = 0.001
    lamda = 2

    # Computing final theta and cost function value in every iteration
    theta, j_history = gradient_descent(
        x_train, y_train, theta, learning_rate, iterations, lamda
    )

    # # Print the value of optimized theta
    # print('Value of theta:\n ', theta)

    # Plot decision boundary using the optimized theta
    plot_decision_boundary(x_train, y_train, theta)

    # Plot a graph cost function (j_history vs iterations)
    plot_cost_function(j_history, iterations)

    # Compute the accuracy
    acc = accuracy(y_train, predict(x_train, theta))
    print("\nValue of accuracy: ", acc)


if __name__ == "__main__":
    # Calling the main function
    main()
