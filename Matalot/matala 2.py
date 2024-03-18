import numpy as np
from numpy.linalg import norm

from colors import bcolors

def is_diagonally_dominant(mat):
    if mat is None:
        return False

    d = np.diag(np.abs(mat))  # Find diagonal coefficients
    s = np.sum(np.abs(mat), axis=1) - d  # Find row sum without diagonal
    return np.all(d > s)

def jacobi_iterative(A, b, X0, TOL=0.001, N=100):
    n = len(A)
    k = 1

    if is_diagonally_dominant(A):
        print('Matrix is diagonally dominant - preforming jacobi algorithm\n')

    print( "Iteration" + "\t\t\t".join([" {:>12}".format(var) for var in ["x{}".format(i) for i in range(1, len(A) + 1)]]))
    print("-----------------------------------------------------------------------------------------------")

    while k <= N:
        x = np.zeros(n, dtype=np.double)
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * X0[j]
            x[i] = (b[i] - sigma) / A[i][i]

        print("{:<15} ".format(k) + "\t\t".join(
            ["{:<15} ".format(round(val)) if val % 1 < 1e-5 else "{:<15.3f} ".format(val) for val in x]))

        if all(abs(val - round(val)) < 1e-4 for val in x):
            print(bcolors.OKBLUE + "All values are close to whole numbers. Stopping iterations.")
            return tuple(round(val) for val in x)

        k += 1
        X0 = x.copy()

    print("Maximum number of iterations exceeded")
    return tuple(x)

def gauss_seidel(A, b, X0, TOL=0.0001, N=200):
    n = len(A)
    k = 1

    if is_diagonally_dominant(A):
        print('Matrix is diagonally dominant - preforming gauss seidel algorithm\n')

    print( "Iteration" + "\t\t\t".join([" {:>12}".format(var) for var in ["x{}".format(i) for i in range(1, len(A) + 1)]]))
    print("-----------------------------------------------------------------------------------------------")
    x = np.zeros(n, dtype=np.double)
    while k <= N:

        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * x[j]
            x[i] = (b[i] - sigma) / A[i][i]

        print("{:<15} ".format(k) + "\t\t".join(["{:<15} ".format(round(val)) if val % 1 < 1e-5 else "{:<15.3f} ".format(val) for val in x]))

        if norm(x - X0, np.inf) < TOL:
            for i in range(len(x)):
                x[i]= np.ceil(x[i])
            return tuple(x)

        k += 1
        X0 = x.copy()

    print("Maximum number of iterations exceeded")
    return tuple(x)



if __name__ == "__main__":

    def option1():
        print("***** Gauss-Seidel Iterative Method (3x3)******")
        A = np.array([input("Enter values for row {}: ".format(i + 1)).split() for i in range(3)], dtype=np.float64)
        b = np.array(input("Enter 3 values for b: ").split(), dtype=np.float64)
        x0 = np.zeros_like(b, dtype=np.double)
        solution = gauss_seidel(A, b, x0)
        print(bcolors.OKBLUE, "\nApproximate solution:", solution )
        print(bcolors.ENDC)

    def option2():
        print("***** Jacobi Iterative Method (3x3)******")
        A = np.array([input("Enter values for row {}: ".format(i + 1)).split() for i in range(3)], dtype=np.float64)
        b = np.array(input("Enter 3 values for b: ").split(), dtype=np.float64)
        x0 = np.zeros_like(b, dtype=np.double)
        solution = jacobi_iterative(A, b, x0)
        print(bcolors.OKBLUE, "\nApproximate solution:")
        for i in range(len(solution)):
            print("x{} = {}".format(i + 1, solution[i]))
        print(bcolors.ENDC)
    def exit_program():
        print("Exiting the program")
        exit()

    # Other functions remain the same

    # Menu options and loop
    menu_options = {
        '1': option1,
        '2': option2,
        '3': exit_program
    }

    while True:
        print("Choose an option:")
        print("1. Gauss-Seidel Iterative Method")
        print("2. Jacobi Iterative Method")
        print("3. Exit Program")

        choice = input("Enter your choice: ")

        if choice in menu_options:
            menu_options[choice]()
        else:
            print("Invalid choice. Please try again.")


