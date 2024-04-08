import numpy as np
import sympy as sp
import math
from colors import bcolors
"""
Correct!
"""
x = sp.symbols('x')

def romberg_integration(func, a, b, max_iterations=10, tol=1e-6):
    """
    Romberg Integration

    Parameters:
    func (function): The function to be integrated.
    a (float): The lower limit of integration.
    b (float): The upper limit of integration.
    n (int): The number of iterations (higher value leads to better accuracy).

    Returns:
    float: The approximate definite integral of the function over [a, b].
    """
    h = b - a
    R = np.zeros((n, n), dtype=float)

    R[0, 0] = 0.5 * h * (func(a) + func(b))

    for i in range(1, n):
        h /= 2
        sum_term = 0

        for k in range(1, 2 ** i, 2):
            sum_term += func(a + k * h)

        R[i, 0] = 0.5 * R[i - 1, 0] + h * sum_term

        for j in range(1, i + 1):
            R[i, j] = R[i, j - 1] + (R[i, j - 1] - R[i - 1, j - 1]) / ((4 ** j) - 1)

    return R[n - 1, n - 1]




def romberg_integration2(f, a, b, n):
    h = (b - a)
    R = np.zeros((n, n))
    R[0, 0] = 0.5 * h * (f(a) + f(b))

    power_of_2 = 1
    for i in range(1, n):
        h /= 2
        R[i, 0] = 0.5 * R[i - 1, 0] + h * sum(f(a + (2 * j - 1) * h) for j in range(1, power_of_2 + 1))
        power_of_2 *= 2
        for k in range(1, i + 1):
            R[i, k] = R[i, k - 1] + (R[i, k - 1] - R[i - 1, k - 1]) / (4 ** k - 1)

    return R[n-1, n-1]


def f(x):
    return (2*x**2 + sp.cos(2*math.e**(-2*x))) / (2*x**3 + x**2 - 6)


if __name__ == '__main__':

    a = -0.6
    b = -0.5
    n = 2
    integral = romberg_integration2(f, a, b, n)
    if integral < 0:
        integral = abs(integral)

    print( f" Division into n={n} sections ")
    print(bcolors.OKBLUE, f"Approximate integral in range [{a},{b}] is {integral:.5f}", bcolors.ENDC)

