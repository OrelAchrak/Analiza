import sympy as sp
from sympy.utilities.lambdify import lambdify
from colors import bcolors



def newton_raphson(f, a , b, TOL):
    print("---------------------------------------------------- \n"
          "          Newton's Raphson method \n"
          "---------------------------------------------------- \n")
    N = 100
    range_flag = False
    x = sp.symbols('x')
    df = f.diff(x)
    df = lambdify(x, df)
    f = lambdify(x, f)
    p0 = a
    print("{:<10} {:<15} {:<15} ".format("Iteration", "po", "p1"))
    for i in range(N):
        if df(p0) == 0:
            print("Derivative is zero at p0, method cannot continue.")
            return None
        p = p0 - f(p0) / df(p0)
        if abs(p - p0) < TOL:
            if a <= p <= b:
                return p  # Procedure completed successfully
            elif not range_flag:
                print("\n Trying to find the root at b. \n")
                i, p0 = 0, b
                range_flag = True
                p = p0
            else:
                return None
        print("{:<10} {:<15.9f} {:<15.9f} ".format(i, p0, p))
        p0 = p
    return None

def secant_method(f,a, b, TOL):
    print("---------------------------------------------------- \n"
          "                 secant method \n"
          "---------------------------------------------------- \n")
    N = 100
    x0, x1 = a, b
    x = sp.symbols('x')
    f = lambdify(x, f)
    print("{:<10} {:<15} {:<15} {:<15}".format("Iteration", "xo", "x1", "p"))
    for i in range(N):
        if f(x1) - f(x0) == 0:
            print(" method cannot continue.\n")
            return
        p = x0 - f(x0) * ((x1 - x0) / (f(x1) - f(x0)))
        if abs(p - x1) < TOL:
            return p  # Procedure completed successfully
        print("{:<10} {:<15.6f} {:<15.6f} {:<15.6f}".format(i, x0, x1,p))
        x0 = x1
        x1 = p
    return p


if __name__ == '__main__':
    x = sp.symbols('x')
    f = x**2 -1
    a = int(input("Enter number (a): "))
    b = int(input("Enter number (b) bigger then (a): "))
    if a>b:
        print("a is more than b! \nTry again! \n")
    else:
        root = secant_method(f,a,b,TOL=1e-10)
        if root:
            print(bcolors.OKBLUE, f"\n The equation f(x) has an approximate root at x = {root}", bcolors.ENDC)
            exit()
        else:
            root = newton_raphson(f, a, b, TOL=1e-10)
            if root:
                print(bcolors.OKBLUE, f"\n The equation f(x) has an approximate root at x = {root}", bcolors.ENDC)