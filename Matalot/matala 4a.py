from colors import bcolors
import sympy as sp
from matrix_utility import *

#Lagrange Interpolation
def lagrange_interpolation(x_data, y_data, x):
    n = len(x_data)
    result = 0.0

    for i in range(n):
        term = y_data[i]
        for j in range(n):
            if i != j:
                term *= (x - x_data[j]) / (x_data[i] - x_data[j])
        result += term

    return result

#Linear interpolation
def linearInterpolation(table_points, point):
    p = []
    result = 0
    flag = 1
    for i in range(len(table_points)):
        p.append(table_points[i][0])
    for i in range(len(p) - 1):
        if i <= point <= i + 1:
            x1 = table_points[i][0]
            x2 = table_points[i + 1][0]
            y1 = table_points[i][1]
            y2 = table_points[i + 1][1]
            result = (((y1 - y2) / (x1 - x2)) * point) + ((y2 * x1) - (y1 * x2)) / (x1 - x2)
            print(bcolors.OKGREEN, "\nThe approximation (interpolation) of the point ", point, " is: ", bcolors.ENDC,
                  round(result, 4))
            flag = 0
    if flag:
        x1 = table_points[0][0]
        x2 = table_points[1][0]
        y1 = table_points[0][1]
        y2 = table_points[1][1]
        m = (y1 - y2) / (x1 - x2)
        result = y1 + m * (point - x1)
        print(bcolors.OKGREEN, "\nThe approximation (extrapolation) of the point ", point, " is: ", bcolors.ENDC,
              round(result, 4))

#Polynomial Interpolation
def GaussJordanElimination(matrix, vector):
    """
    Function for solving a linear equation using gauss's elimination method
    :param matrix: Matrix nxn
    :param vector: Vector n
    :return: Solve Ax=b -> x=A(-1)b
    """
    # Pivoting process
    matrix, vector = RowXchange(matrix, vector)
    # Inverse matrix calculation
    invert = InverseMatrix(matrix, vector)
    return MulMatrixVector(invert, vector)

def UMatrix(matrix, vector):
    """
    :param matrix: Matrix nxn
    :return:Disassembly into a  U matrix
    """
    # result matrix initialized as singularity matrix
    U = MakeIMatrix(len(matrix), len(matrix))
    # loop for each row
    for i in range(len(matrix[0])):
        # pivoting process
        matrix, vector = RowXchageZero(matrix, vector)
        for j in range(i + 1, len(matrix)):
            elementary = MakeIMatrix(len(matrix[0]), len(matrix))
            # Finding the M(ij) to reset the organs under the pivot
            elementary[j][i] = -(matrix[j][i]) / matrix[i][i]
            matrix = MultiplyMatrix(elementary, matrix)
    # U matrix is a doubling of elementary matrices that we used to reset organs under the pivot
    U = MultiplyMatrix(U, matrix)
    return U

def LMatrix(matrix, vector):
    """
       :param matrix: Matrix nxn
       :return:Disassembly into a  L matrix
       """
    # Initialize the result matrix
    L = MakeIMatrix(len(matrix), len(matrix))
    # loop for each row
    for i in range(len(matrix[0])):
        # pivoting process
        matrix, vector = RowXchageZero(matrix, vector)
        for j in range(i + 1, len(matrix)):
            elementary = MakeIMatrix(len(matrix[0]), len(matrix))
            # Finding the M(ij) to reset the organs under the pivot
            elementary[j][i] = -(matrix[j][i]) / matrix[i][i]
            # L matrix is a doubling of inverse elementary matrices
            L[j][i] = (matrix[j][i]) / matrix[i][i]
            matrix = MultiplyMatrix(elementary, matrix)

    return L

def SolveLU(matrix, vector):
    """
    Function for deconstructing a linear equation by ungrouping LU
    :param matrix: Matrix nxn
    :param vector: Vector n
    :return: Solve Ax=b -> x=U(-1)L(-1)b
    """
    matrixU = UMatrix(matrix)
    matrixL = LMatrix(matrix)
    return MultiplyMatrix(InverseMatrix(matrixU), MultiplyMatrix(InverseMatrix(matrixL), vector))

def solveMatrix(matrixA, vectorb):
    detA = Determinant(matrixA, 1)
    print(bcolors.YELLOW, "\nDET(A) = ", detA)

    if detA != 0:
        print("CondA = ", Cond(matrixA, InverseMatrix(matrixA, vectorb)), bcolors.ENDC)
        print(bcolors.OKBLUE, "\nnon-Singular Matrix - Perform GaussJordanElimination", bcolors.ENDC)
        result = GaussJordanElimination(matrixA, vectorb)
        print(np.array(result))
        return result
    else:
        print("Singular Matrix - Perform LU Decomposition\n")
        print("Matrix U: \n")
        print(np.array(UMatrix(matrixA, vectorb)))
        print("\nMatrix L: \n")
        print(np.array(LMatrix(matrixA, vectorb)))
        print("\nMatrix A=LU: \n")
        result = MultiplyMatrix(LMatrix(matrixA, vectorb), UMatrix(matrixA, vectorb))
        print(np.array(result))
        return result

def polynomialInterpolation(table_points, x):
    matrix = [[point[0] ** i for i in range(len(table_points))] for point in table_points]  # Makes the initial matrix

    b = [[point[1]] for point in table_points]

    print(bcolors.OKBLUE, "The matrix obtained from the points: ", bcolors.ENDC, '\n', np.array(matrix))
    print(bcolors.OKBLUE, "\nb vector: ", bcolors.ENDC, '\n', np.array(b))
    matrixSol = solveMatrix(matrix, b)

    result = sum([matrixSol[i][0] * (x ** i) for i in range(len(matrixSol))])
    print(bcolors.OKBLUE, "\nThe polynom:", bcolors.ENDC)
    print('P(X) = ' + '+'.join(['(' + str(matrixSol[i][0]) + ') * x^' + str(i) + ' ' for i in range(len(matrixSol))]))
    print(bcolors.OKGREEN, f"\nThe Result of P(X={x}) is:", bcolors.ENDC)
    print(result)
    return result

def main_menu():
    def option_1():
        table_points = []
        print(bcolors.OKBLUE, "How many points do you want to enter? ", bcolors.ENDC)
        n = int(input())
        print(bcolors.OKBLUE, "Enter the points: ", bcolors.ENDC)
        for i in range(n):
            print(bcolors.OKBLUE, "Enter the x", i + 1, " point: ", bcolors.ENDC)
            x = float(input())
            print(bcolors.OKBLUE, "Enter the y", i + 1, " point: ", bcolors.ENDC)
            y = float(input())
            table_points.append((x, y))
        print(bcolors.OKBLUE, "Enter the point you want to interpolate: ", bcolors.ENDC)
        point = float(input())
        # Assuming linearInterpolation is defined somewhere
        linearInterpolation(table_points, point)

    def option_2():
        table_points = []
        print(bcolors.OKBLUE, "How many points do you want to enter? ", bcolors.ENDC)
        n = int(input())
        print(bcolors.OKBLUE, "Enter the points: ", bcolors.ENDC)
        for i in range(n):
            print(bcolors.OKBLUE, "Enter the x", i + 1, " point: ", bcolors.ENDC)
            x = float(input())
            print(bcolors.OKBLUE, "Enter the y", i + 1, " point: ", bcolors.ENDC)
            y = float(input())
            table_points.append((x, y))
        print(bcolors.OKBLUE, "Enter the point you want to interpolate: ", bcolors.ENDC)
        point = float(input())
        polynomialInterpolation(table_points, point)
    while True:
        print(bcolors.OKBLUE , "Main menu :" , bcolors.ENDC)
        print("1. Linear interpolation")
        print("2. Polynomial interpolation")
        print("3. Exit")
        choice = int(input("Enter your choice: "))
        if choice == 1:
            option_1()
        elif choice == 2:
            option_2()
        elif choice == 3:
            exit()

if __name__ == '__main__':
    print(bcolors.OKBLUE, "----------------- Interpolation Methods -----------------", bcolors.ENDC)
    main_menu()
