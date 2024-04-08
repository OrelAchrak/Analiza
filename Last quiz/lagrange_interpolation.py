from colors import bcolors
import sympy as sp


def lagrange_interpolation(x_data, y_data, x):
    """
    Lagrange Interpolation

    Parameters:
    x_data (list): List of x-values for data points.
    y_data (list): List of y-values for data points.
    x (float): The x-value where you want to evaluate the interpolated polynomial.

    Returns:
    float: The interpolated y-value at the given x.
    Correct!
    """
    n = len(x_data)
    result = 0.0

    for i in range(n):
        term = y_data[i]
        for j in range(n):
            if i != j:
                term *= (x - x_data[j]) / (x_data[i] - x_data[j])
        result += term

    return result

import numpy as np

if __name__ == '__main__':
    x_data = [1.2 , 1.3 , 1.4 , 1.5 , 1.6]
    y_data = [1.31, 2.69, 1.30, -1.25, -2.1 ]
    x_interpolate = [1.47, 1.65]  # The x-values where you want to interpolate
    for item in x_interpolate:
        y_interpolate = lagrange_interpolation(x_data, y_data, item)
        print("Interpolated value at x =", item, "is y =", y_interpolate)




