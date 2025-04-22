#By:        Shane Billingsley, Sawyer Kuvin, Hannah Priddy
#Class:     APPM4600 Numerical Analysis and Scientific Computing
#Date:      5-06-2025
#File:      Final Project
#           Quadrature for Weakly Singular Integrals

import numpy as np
import math
from numpy import linalg as la
import matplotlib.pyplot as plt
import scipy
import scipy.special as scipy_sp
import scipy.integrate as scipy_int
from scipy.integrate import quad
from scipy.integrate import fixed_quad as fix
from numpy.polynomial.legendre import leggauss as lgwt
vrb = True

def driver_1(vrb):
    #define functions under consideration and plot for future reference
    f = lambda x:   1/np.sqrt(x)
    g = lambda x:   np.log(x)
    a = 0;  b = 1;
    x = np.linspace(b, a, 100, endpoint=False)[::-1]
    y1 = f(x);  y2 = g(x);
    if vrb:
        plt.figure()
        plt.plot(x,y1)
        plt.title(r"$f(x) = \frac{1}{\sqrt{x}}$ on [0,1]")
        plt.ylim(0, 5)
        plt.xlim(0, 1)
        plt.grid(True)
        plt.show()

    if vrb:
        plt.figure()
        plt.plot(x, y2)
        plt.title(r"$f(x) = ln(x)$ on [0,1]")
        plt.ylim(-5, 1)
        plt.xlim(0, 1)
        plt.grid(True)
        plt.show()

def driver_2(vrb):
    #integrate example functions using fixed_quad to demonstrate crappiness
    f = lambda x: 1 / np.sqrt(x)
    h = lambda x: np.exp(x)
    a = 0;  b = 1;
    #integrate f using fixed order quadrature
    Fq_1 = fix(f,a,b,n=5)
    Fq_2 = fix(f, a, b, n=20)
    Fq_3 = fix(f, a, b, n=50)
    Fq_4 = fix(f, a, b, n=100)
    Fq_5 = fix(f, a, b, n=200)
    # integrate h using fixed order quadrature
    Hq_1 = fix(h, a, b, n=5)
    Hq_2 = fix(h, a, b, n=20)
    Hq_3 = fix(h, a, b, n=50)
    Hq_4 = fix(h, a, b, n=100)
    Hq_5 = fix(h, a, b, n=200)
    if vrb:
        # Define column widths and title
        col1_width = 10
        col2_width = 10
        table_width = col1_width + col2_width + 7  # 7 accounts for borders and spacing
        title = "Fixed-Quad Integration of 1/sqrt(x)"

        # Create a horizontal line
        line = "+" + "-" * (col1_width + 2) + "+" + "-" * (col2_width + 2) + "+"

        # Print the title centered
        print(title.center(table_width))
        print(line)

        # Print header
        print(f"| {'Quad Order':<{col1_width}} | {'Result':<{col2_width}} |")
        print(line)

        # Print rows
        print(f"| {'n=5':<{col1_width}} | {f'{2.-Fq_1[0]}':<{col2_width}} |")
        print(f"| {'n=20':<{col1_width}} | {f'{2.-Fq_2[0]}':<{col2_width}} |")
        print(f"| {'n=50':<{col1_width}} | {f'{2.-Fq_3[0]}':<{col2_width}} |")
        print(f"| {'n=100':<{col1_width}} | {f'{2.-Fq_4[0]}':<{col2_width}} |")
        print(f"| {'n=200':<{col1_width}} | {f'{2.-Fq_5[0]}':<{col2_width}} |")
        print(line)

    if vrb:
        # Define column widths and title
        col1_width = 10
        col2_width = 10
        table_width = col1_width + col2_width + 7  # 7 accounts for borders and spacing
        title = "Fixed-Quad Integration of exp(x)"

        # Create a horizontal line
        line = "+" + "-" * (col1_width + 2) + "+" + "-" * (col2_width + 2) + "+"

        # Print the title centered
        print(title.center(table_width))
        print(line)

        # Print header
        print(f"| {'Quad Order':<{col1_width}} | {'Result':<{col2_width}} |")
        print(line)

        # Print rows
        print(f"| {'n=5':<{col1_width}} | {f'{(np.e-1)-Hq_1[0]}':<{col2_width}} |")
        print(f"| {'n=20':<{col1_width}} | {f'{(np.e-1)-Hq_2[0]}':<{col2_width}} |")
        print(f"| {'n=50':<{col1_width}} | {f'{(np.e-1)-Hq_3[0]}':<{col2_width}} |")
        print(f"| {'n=100':<{col1_width}} | {f'{(np.e-1)-Hq_4[0]}':<{col2_width}} |")
        print(f"| {'n=200':<{col1_width}} | {f'{(np.e-1)-Hq_5[0]}':<{col2_width}} |")
        print(line)

def driver_3(vrb):
    #integrate example function using better algorithms to demonstrate goals
    f = lambda x: 1 / np.sqrt(x)
    h = lambda x: np.exp(x)
    a = 0;  b = 1;
    F_n1, F_n1_err, Fn1_info = quad(f, a, b,full_output=1)
    Fn1_neval = Fn1_info["neval"]
    Fn1_int = Fn1_info["last"]
    H_n1, _, Hn1_info,_ = quad(f, a, b, full_output=1,limit=4)
    Hn1_neval = Hn1_info["neval"]
    Hn1_int = Hn1_info["last"]
    if vrb:
        print(f"Scipy Quad for 1/sqrt(x) produced error {np.abs(2-F_n1)} with {Fn1_neval} function evaluations")
        print(f"with {Fn1_int} intervals")
        print(f"Scipy Quad for 1/sqrt(x) produced error {np.abs(2-H_n1)} with {Hn1_neval} function evaluations")
        print(f"with {Hn1_int} intervals")

def BetterQuadWeights(n):
    wk = np.zeros(n)
    for k in range(n):
        if k == 0:
            gk = 1
        elif k == n - 1:
            gk = n
        else:
            gk = 2

        sum = 1
        for j in range(int(n / 2)):
            if j == (n / 2) - 1:
                bj = 1
            else:
                bj = 2

            sum = sum - ((bj / (4 * ((j + 1) ** 2) - 1))) * np.cos(2 * (j + 1) * (k + 1) * np.pi / n)

        wk[k] = ((gk / n) * sum) / 2

    return (wk)


def JacobiMethod(n, alpha, beta, a, b):
    p_n = scipy_sp.jacobi(n, alpha, beta)
    p_n_1 = scipy_sp.jacobi(n + 1, alpha, beta)
    # print(p_n)
    gammaterm = math.gamma(alpha + beta + n + 1 + 1) / (2 * scipy_sp.gamma(alpha + beta + n + 1))
    j_prime_term = scipy_sp.jacobi(n - 1, alpha + 1, beta + 1)
    p_n_prime = j_prime_term * gammaterm

    # print(p_n_prime)
    x_k, w = scipy_sp.roots_jacobi(n, alpha, beta)  # roots Jacobi gives us the weights, we calculate it
    # print(x_k)

    w_k = np.zeros(n)
    for k in range(n):
        # Jacobi Method 1:
        '''weight_int = lambda x: p_n(x)/((x-x_k[k])*p_n_prime(x_k[k])) * (1-x)**alpha * (1+x)**beta
        #print(weight_int(1))
        w_k[k],_ = scipy_int.quad(weight_int,a,b)
        #print(w_k)'''

        # Jacobi Method 2:
        '''frac1_num = math.gamma(n+alpha+1) * math.gamma(n+beta+1)
        frac1_den = math.gamma(n+alpha+beta+1)
        frac1 = frac1_num/frac1_den

        V_n_prime = lambda x: (-1)**(n-1) * p_n_prime(x) * (2**n) * math.factorial(n)

        frac2_num = 2**(2*n + alpha+ beta+1) * math.factorial(n)
        frac2_den = (1-x_k[k]**2) * (V_n_prime(x_k[k]))**2

        frac2 = frac2_num/frac2_den'''

        # Jacobi Method 3
        frac1_num = -(2 * n + alpha + beta + 2)
        frac1_den = n + alpha + beta + 1
        frac1 = frac1_num / frac1_den

        frac2_num = math.gamma(n + alpha + 1) * math.gamma(n + beta + 1)
        frac2_den = math.gamma(n + alpha + beta + 1) * math.factorial(n + 1)
        frac2 = frac2_num / frac2_den

        frac3_num = 2 ** (alpha + beta)
        frac3_den = p_n_prime(x_k[k]) * p_n_1(x_k[k])
        frac3 = frac3_num / frac3_den

        w_k[k] = frac1 * frac2 * frac3

    return (x_k, w_k)


def Test():
    n = 200
    k = np.linspace(1, n, n)
    xk = (1 - np.cos(((2 * k - 1) * np.pi) / (2 * n + 2))) / 4

    # print(xk)
    f = lambda x: x

    wk = BetterQuadWeights(n)

    test = np.sum(f(xk) * wk) + 0.25

    print(test)


def JacobiTest():
    Nmax = 4
    n_vec = np.arange(1, Nmax + 1)
    jac_soln = np.zeros(Nmax)
    err = np.zeros(Nmax)
    soln = 2
    for i in range(len(n_vec)):
        n = n_vec[i]
        # a term:
        a = 0  # start of interval
        beta = -1 / 2

        # b term:
        b = 1  # end of interval
        alpha = 0

        _, wk = JacobiMethod(n, alpha, beta, a, b)

        jac_soln[i] = (((b - a) / 2) ** (1 + alpha + beta)) * sum(wk)
        err[i] = abs(soln - jac_soln[i]) / soln
        print("The number of Jacobi Evals is: ", n, " and the error is: ", err[i])

    plt.figure
    plt.semilogy(n_vec, err)
    plt.grid(True)
    plt.show()

def compJacobiTest():
    Nmax = 10
    n_vec = np.arange(1, Nmax + 1)
    jac_soln = np.zeros(Nmax)
    err = np.zeros(Nmax)
    soln = 18.9622351876;
    f = lambda x: np.exp(x)/(np.cbrt(x-1)**2)
    g = lambda x: np.exp(x)/(np.sign(x+6)*((np.abs(x+6))**(1/4)))
    h = lambda x: np.exp(x)/((np.sign(x)*((np.abs(x))**(4/5)))*((np.sign(x+6)*((np.abs(x+6))**(1/4)))))

    for i in range(len(n_vec)):
        n = n_vec[i]

        #first integral
        # a term:
        a = -6  # start of interval
        beta = -(1/4)

        # b term:
        b = 0  # end of interval
        alpha = -(4/5)

        xk_1, wk_1 = JacobiMethod(n, alpha, beta, a, b)
        #transform xk to be on the apprpriate interval
        xk_1_new = ((1-xk_1)*a + (1+xk_1)*b)/2
        jac_soln_1 = (((b - a) / 2) ** (1 + alpha + beta)) * sum(wk_1*f(xk_1_new))

        #second integral
        # a term:
        a = 0  # start of interval
        beta = -(4 / 5)

        # b term:
        b = 1  # end of interval
        alpha = -(2 / 3)

        xk_2, wk_2 = JacobiMethod(n, alpha, beta, a, b)
        # transform xk to be on the apprpriate interval
        xk_2_new = ((1 - xk_2) * a + (1 + xk_2) * b) / 2
        jac_soln_2 = (((b - a) / 2) ** (1 + alpha + beta)) * sum(wk_2 * g(xk_2_new))

        # third integral
        # a term:
        a = 1  # start of interval
        beta = -(2 / 3)

        # b term:
        b = 4  # end of interval
        alpha = 0

        xk_3, wk_3 = JacobiMethod(n, alpha, beta, a, b)
        # transform xk to be on the apprpriate interval
        xk_3_new = ((1 - xk_3) * a + (1 + xk_3) * b) / 2
        jac_soln_3 = (((b - a) / 2) ** (1 + alpha + beta)) * sum(wk_3 * h(xk_3_new))


        jac_soln[i] = jac_soln_1 + jac_soln_2 + jac_soln_3
        err[i] = abs(soln - jac_soln[i]) / soln
        print("The number of Jacobi Evals is: ", n, " and the error is: ", err[i])

    plt.figure
    plt.semilogy(n_vec, err)
    plt.grid(True)
    plt.show()

#JacobiTest()
compJacobiTest()
#driver_1(vrb)
#driver_2(vrb)
#driver_3(vrb)