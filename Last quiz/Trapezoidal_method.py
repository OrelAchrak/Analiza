import math
import sympy as sp

from colors import bcolors

x= sp.symbols('x')

def trapezoidal_rule(f, a, b, n):

    h = (b - a) / n
    T = f(a) + f(b)
    integral = 0.5 * T  # Initialize with endpoints

    for i in range(1, n):
        x_i = a + i * h
        integral += f(x_i)

    integral *= h
    return integral

def calculate_error_trapezoidal(f, a, b, n):
    h = (b - a) / n
    x_values = [a + i * h for i in range(n+1)]
    max_derivative_2 = max(abs(f(x).diff(x, 2).subs(x, xi)) for xi in x_values)
    error = (1/12) * h**2 * (b - a) * max_derivative_2
    return error


if __name__ == '__main__':
    f = lambda x:math.e ** (x ** 2 )
    result = trapezoidal_rule(f, 0, 1, 2)
    error = calculate_error_trapezoidal(f, 0, 1, 2)
    print(bcolors.OKBLUE,"Approximate integral:", result, bcolors.ENDC)
    print(bcolors.OKBLUE,"Error:", error, bcolors.ENDC)
