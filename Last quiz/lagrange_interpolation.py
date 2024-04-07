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

def lagrange_interpolation_error(x, f, interpolation_points):
    """
    Calculate the error in Lagrange interpolation at a point x.

    Parameters:
    x (float): The point at which to calculate the error.
    f (function): The function being interpolated.
    interpolation_points (list): List of interpolation points.

    Returns:
    float: The error in Lagrange interpolation at the point x.
    """
    n = len(interpolation_points) - 1
    x_sym = sp.symbols('x')
    lagrange_poly = lagrange_interpolation(interpolation_points, [f(xi) for xi in interpolation_points], x_sym)
    derivative = lagrange_poly.diff(x_sym, n + 1)
    error = abs(derivative.subs(x_sym, x))
    return error

# Example usage
f = lambda x: np.sin(x)  # Example function
interpolation_points = [0, np.pi/2, np.pi]  # Example interpolation points
x = np.pi/4  # Example point to calculate the error at
error = lagrange_interpolation_error(x, f, interpolation_points)
print("Error at x =", x, ":", error)


if __name__ == '__main__':
    x_data = [1, 2, 5]
    y_data = [1, 0, 2]
    x_interpolate = 3  # The x-value where you want to interpolate
    y_interpolate = lagrange_interpolation(x_data, y_data, x_interpolate)
    actual_value = 1.5  # Assuming the actual value at x=3 is 1.5
    error = lagrange_interpolation_error(x_interpolate, f, x_data)
    print("Interpolated value at x =", x_interpolate, "is y =", y_interpolate)
    print("Actual value at x =", x_interpolate, "is y =", actual_value)
    print("Error is", error)



