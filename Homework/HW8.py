#By:        Shane Billingsley
#Class:     APPM4600 Numerical Analysis and Scientific Computing
#Date:      3-14-2025
#File:      HW 8 assignment

import numpy as np
from numpy import linalg as lg
import matplotlib.pyplot as plt
import scipy.special as sp
vrb = True

def prob1a(vrb):
    f = lambda x: 1 / (1 + x**2)    #function of interest
    a = -5; b = 5;                  #interval of interest
    N = np.array([5,10,15,20])      #number of nodes
    x = [None] * len(N);    y = [None] * len(N);  # preallocate storage
    xeval = [None] * len(N);    yeval = [None] * len(N);
    error = [None] * len(N);
    for n in range(len(N)):  # loop through increasing number of nodes
        x[n] = equispaced(a,b,N[n])
        y[n] = f(x[n])  # find function values at node locations
        xeval[n] = np.linspace(a, b, 1001)  # create interpolation points
        yeval[n] = lagrange_eval(xeval[n], x[n], y[n])  # interpolate values at nodes
        error[n] = np.abs(yeval[n] - f(xeval[n]))  # find error as |p(x) - f(x)|
    if vrb:  # plotting
        fig1, axs = plt.subplots(2, 2)
        axs[0, 0].plot(x[0], y[0], 'o')
        axs[0, 0].plot(xeval[0], f(xeval[0]), 'g')
        axs[0, 0].plot(xeval[0], yeval[0], 'r')
        axs[0, 0].set_title('N=5')
        axs[0, 1].plot(x[1], y[1], 'o')
        axs[0, 1].plot(xeval[1], f(xeval[1]), 'g')
        axs[0, 1].plot(xeval[1], yeval[1], 'r')
        axs[0, 1].set_title('N=10')
        axs[1, 0].plot(x[2], y[2], 'o')
        axs[1, 0].plot(xeval[2], f(xeval[2]), 'g')
        axs[1, 0].plot(xeval[2], yeval[2], 'r')
        axs[1, 0].set_title('N=15')
        axs[1, 1].plot(x[3], y[3], 'o')
        axs[1, 1].plot(xeval[3], f(xeval[3]), 'g')
        axs[1, 1].plot(xeval[3], yeval[3], 'r')
        axs[1, 1].set_title('N=20')
        for ax in axs.flat:
            ax.label_outer()
        fig2, axs = plt.subplots(2, 2)
        axs[0, 0].plot(xeval[0], np.log10(error[0]), 'b')
        axs[0, 0].set_title('$Log_{10}$ Error for N=5')
        axs[0, 1].plot(xeval[1], np.log10(error[1]), 'b')
        axs[0, 1].set_title('$Log_{10}$ Error for N=10')
        axs[1, 0].plot(xeval[2], np.log10(error[2]), 'b')
        axs[1, 0].set_title('$Log_{10}$ Error for N=15')
        axs[1, 1].plot(xeval[3], np.log10(error[3]), 'b')
        axs[1, 1].set_title('$Log_{10}$ Error for N=20')
        for ax in axs.flat:
            ax.label_outer()

def prob1b(vrb):
    f = lambda x: 1 / (1 + x ** 2)  # function of interest
    df = lambda x: (-2 * x)/((1 + x**2)**2) #function derivative
    a = -5;
    b = 5;  # interval of interest
    N = np.array([5, 10, 15, 20])  # number of nodes
    x = [None] * len(N);    y = [None] * len(N);  # preallocate storage
    xeval = [None] * len(N);    yeval = [None] * len(N);
    error = [None] * len(N);    dy = [None] * len(N);
    for n in range(len(N)):  # loop through increasing number of nodes
        x[n] = equispaced(a,b,N[n])
        y[n] = f(x[n])  # find function values at node locations
        dy[n] = df(x[n])    #find derivative values at node locations
        xeval[n] = np.linspace(a, b, 1001)  # create interpolation points
        yeval[n] = np.zeros(len(xeval[n]))  #preallocate
        for i in range(len(xeval[n])):
            yeval[n][i] = eval_hermite(xeval[n][i],x[n],y[n],dy[n],N[n]-1)
        error[n] = np.abs(yeval[n] - f(xeval[n]))  # find error as |p(x) - f(x)|

    if vrb:  # plotting
        fig1, axs = plt.subplots(2, 2)
        axs[0, 0].plot(x[0], y[0], 'o')
        axs[0, 0].plot(xeval[0], f(xeval[0]), 'g')
        axs[0, 0].plot(xeval[0], yeval[0], 'r')
        axs[0, 0].set_title('N=5')
        axs[0, 1].plot(x[1], y[1], 'o')
        axs[0, 1].plot(xeval[1], f(xeval[1]), 'g')
        axs[0, 1].plot(xeval[1], yeval[1], 'r')
        axs[0, 1].set_title('N=10')
        axs[1, 0].plot(x[2], y[2], 'o')
        axs[1, 0].plot(xeval[2], f(xeval[2]), 'g')
        axs[1, 0].plot(xeval[2], yeval[2], 'r')
        axs[1, 0].set_title('N=15')
        axs[1, 1].plot(x[3], y[3], 'o')
        axs[1, 1].plot(xeval[3], f(xeval[3]), 'g')
        axs[1, 1].plot(xeval[3], yeval[3], 'r')
        axs[1, 1].set_title('N=20')
        for ax in axs.flat:
            ax.label_outer()
        fig2, axs = plt.subplots(2, 2)
        axs[0, 0].plot(xeval[0], np.log10(error[0]), 'b')
        axs[0, 0].set_title('$Log_{10}$ Error for N=5')
        axs[0, 1].plot(xeval[1], np.log10(error[1]), 'b')
        axs[0, 1].set_title('$Log_{10}$ Error for N=10')
        axs[1, 0].plot(xeval[2], np.log10(error[2]), 'b')
        axs[1, 0].set_title('$Log_{10}$ Error for N=15')
        axs[1, 1].plot(xeval[3], np.log10(error[3]), 'b')
        axs[1, 1].set_title('$Log_{10}$ Error for N=20')
        for ax in axs.flat:
            ax.label_outer()

def prob1c(vrb):
    f = lambda x: 1 / (1 + x ** 2)  # function of interest
    a = -5;
    b = 5;  # interval of interest
    N = np.array([5, 10, 15, 20])  # number of nodes
    x = [None] * len(N);    y = [None] * len(N);  # preallocate storage
    xeval = [None] * len(N);    yeval = [None] * len(N);
    error = [None] * len(N);    dy = [None] * len(N);
    for n in range(len(N)):  # loop through increasing number of nodes
        x[n] = equispaced(a, b, N[n])
        y[n] = f(x[n])  # find function values at node locations
        xeval[n] = np.linspace(a, b, 1001)  # create interpolation points
        #create and evaluate natural cubic spline
        (M, C, D) = create_natural_spline(y[n], x[n], N[n] -1)
        yeval[n] = eval_cubic_spline(xeval[n], len(xeval[n])-1, x[n], N[n]-1, M, C, D)
        error[n] = np.abs(yeval[n] - f(xeval[n]))  # find error as |p(x) - f(x)|
    if vrb:  # plotting
        fig1, axs = plt.subplots(2, 2)
        axs[0, 0].plot(x[0], y[0], 'o')
        axs[0, 0].plot(xeval[0], f(xeval[0]), 'g')
        axs[0, 0].plot(xeval[0], yeval[0], 'r')
        axs[0, 0].set_title('N=5')
        axs[0, 1].plot(x[1], y[1], 'o')
        axs[0, 1].plot(xeval[1], f(xeval[1]), 'g')
        axs[0, 1].plot(xeval[1], yeval[1], 'r')
        axs[0, 1].set_title('N=10')
        axs[1, 0].plot(x[2], y[2], 'o')
        axs[1, 0].plot(xeval[2], f(xeval[2]), 'g')
        axs[1, 0].plot(xeval[2], yeval[2], 'r')
        axs[1, 0].set_title('N=15')
        axs[1, 1].plot(x[3], y[3], 'o')
        axs[1, 1].plot(xeval[3], f(xeval[3]), 'g')
        axs[1, 1].plot(xeval[3], yeval[3], 'r')
        axs[1, 1].set_title('N=20')
        for ax in axs.flat:
            ax.label_outer()
        fig2, axs = plt.subplots(2, 2)
        axs[0, 0].plot(xeval[0], np.log10(error[0]), 'b')
        axs[0, 0].set_title('$Log_{10}$ Error for N=5')
        axs[0, 1].plot(xeval[1], np.log10(error[1]), 'b')
        axs[0, 1].set_title('$Log_{10}$ Error for N=10')
        axs[1, 0].plot(xeval[2], np.log10(error[2]), 'b')
        axs[1, 0].set_title('$Log_{10}$ Error for N=15')
        axs[1, 1].plot(xeval[3], np.log10(error[3]), 'b')
        axs[1, 1].set_title('$Log_{10}$ Error for N=20')
        for ax in axs.flat:
            ax.label_outer()


def prob2a(vrb):
    f = lambda x: 1 / (1 + x ** 2)    #function of interest
    a = -5; b = 5;                    #interval of interest
    N = np.array([5, 10, 15, 20])     #number of nodes
    x = [None] * len(N);    y = [None] * len(N);  # preallocate storage
    xeval = [None] * len(N);    yeval = [None] * len(N);
    error = [None] * len(N);
    for n in range(len(N)):  # loop through increasing number of nodes
        x[n] = chebyshev(a, b, N[n])
        y[n] = f(x[n])  # find function values at node locations
        xeval[n] = np.linspace(a, b, 1001)  # create interpolation points
        yeval[n] = lagrange_eval(xeval[n], x[n], y[n])  # interpolate values at nodes
        error[n] = np.abs(yeval[n] - f(xeval[n]))  # find error as |p(x) - f(x)|
    if vrb:  # plotting
        fig1, axs = plt.subplots(2, 2)
        axs[0, 0].plot(x[0], y[0], 'o')
        axs[0, 0].plot(xeval[0], f(xeval[0]), 'g')
        axs[0, 0].plot(xeval[0], yeval[0], 'r')
        axs[0, 0].set_title('N=5')
        axs[0, 1].plot(x[1], y[1], 'o')
        axs[0, 1].plot(xeval[1], f(xeval[1]), 'g')
        axs[0, 1].plot(xeval[1], yeval[1], 'r')
        axs[0, 1].set_title('N=10')
        axs[1, 0].plot(x[2], y[2], 'o')
        axs[1, 0].plot(xeval[2], f(xeval[2]), 'g')
        axs[1, 0].plot(xeval[2], yeval[2], 'r')
        axs[1, 0].set_title('N=15')
        axs[1, 1].plot(x[3], y[3], 'o')
        axs[1, 1].plot(xeval[3], f(xeval[3]), 'g')
        axs[1, 1].plot(xeval[3], yeval[3], 'r')
        axs[1, 1].set_title('N=20')
        for ax in axs.flat:
            ax.label_outer()
        fig2, axs = plt.subplots(2, 2)
        axs[0, 0].plot(xeval[0], np.log10(error[0]), 'b')
        axs[0, 0].set_title('$Log_{10}$ Error for N=5')
        axs[0, 1].plot(xeval[1], np.log10(error[1]), 'b')
        axs[0, 1].set_title('$Log_{10}$ Error for N=10')
        axs[1, 0].plot(xeval[2], np.log10(error[2]), 'b')
        axs[1, 0].set_title('$Log_{10}$ Error for N=15')
        axs[1, 1].plot(xeval[3], np.log10(error[3]), 'b')
        axs[1, 1].set_title('$Log_{10}$ Error for N=20')
        for ax in axs.flat:
            ax.label_outer()


def prob2b(vrb):
    f = lambda x: 1 / (1 + x ** 2)  # function of interest
    df = lambda x: (-2 * x) / ((1 + x ** 2) ** 2)
    a = -5;
    b = 5;  # interval of interest
    N = np.array([5, 10, 15, 20])  # number of nodes
    x = [None] * len(N);
    y = [None] * len(N);  # preallocate storage
    xeval = [None] * len(N);
    yeval = [None] * len(N);
    error = [None] * len(N);
    dy = [None] * len(N);
    for n in range(len(N)):  # loop through increasing number of nodes
        x[n] = chebyshev(a, b, N[n])
        y[n] = f(x[n])  # find function values at node locations
        dy[n] = df(x[n])  # find derivative values at node locations
        xeval[n] = np.linspace(a, b, 1001)  # create interpolation points
        yeval[n] = np.zeros(len(xeval[n]))
        for i in range(len(xeval[n])):
            yeval[n][i] = eval_hermite(xeval[n][i], x[n], y[n], dy[n], N[n]-1)
        error[n] = np.abs(yeval[n] - f(xeval[n]))  # find error as |p(x) - f(x)|

    if vrb:  # plotting
        fig1, axs = plt.subplots(2, 2)
        axs[0, 0].plot(x[0], y[0], 'o')
        axs[0, 0].plot(xeval[0], f(xeval[0]), 'g')
        axs[0, 0].plot(xeval[0], yeval[0], 'r')
        axs[0, 0].set_title('N=5')
        axs[0, 1].plot(x[1], y[1], 'o')
        axs[0, 1].plot(xeval[1], f(xeval[1]), 'g')
        axs[0, 1].plot(xeval[1], yeval[1], 'r')
        axs[0, 1].set_title('N=10')
        axs[1, 0].plot(x[2], y[2], 'o')
        axs[1, 0].plot(xeval[2], f(xeval[2]), 'g')
        axs[1, 0].plot(xeval[2], yeval[2], 'r')
        axs[1, 0].set_title('N=15')
        axs[1, 1].plot(x[3], y[3], 'o')
        axs[1, 1].plot(xeval[3], f(xeval[3]), 'g')
        axs[1, 1].plot(xeval[3], yeval[3], 'r')
        axs[1, 1].set_title('N=20')
        for ax in axs.flat:
            ax.label_outer()
        fig2, axs = plt.subplots(2, 2)
        axs[0, 0].plot(xeval[0], np.log10(error[0]), 'b')
        axs[0, 0].set_title('$Log_{10}$ Error for N=5')
        axs[0, 1].plot(xeval[1], np.log10(error[1]), 'b')
        axs[0, 1].set_title('$Log_{10}$ Error for N=10')
        axs[1, 0].plot(xeval[2], np.log10(error[2]), 'b')
        axs[1, 0].set_title('$Log_{10}$ Error for N=15')
        axs[1, 1].plot(xeval[3], np.log10(error[3]), 'b')
        axs[1, 1].set_title('$Log_{10}$ Error for N=20')
        for ax in axs.flat:
            ax.label_outer()
def prob2c(vrb):
    f = lambda x: 1 / (1 + x ** 2)  # function of interest
    a = -5;
    b = 5;  # interval of interest
    N = np.array([5, 10, 15, 20])  # number of nodes
    x = [None] * len(N);    y = [None] * len(N);  # preallocate storage
    xeval = [None] * len(N);    yeval = [None] * len(N);
    error = [None] * len(N);    dy = [None] * len(N);
    for n in range(len(N)):  # loop through increasing number of nodes
        x[n] = chebyshev(a, b, N[n])
        y[n] = f(x[n])  # find function values at node locations
        xeval[n] = np.linspace(a, b, 1001)  # create interpolation points
        #create and evaluate natural cubic spline
        (M, C, D) = create_natural_spline(y[n], x[n], N[n] -1)
        yeval[n] = eval_cubic_spline(xeval[n], len(xeval[n])-1, x[n], N[n]-1, M, C, D)
        error[n] = np.abs(yeval[n] - f(xeval[n]))  # find error as |p(x) - f(x)|
    if vrb:  # plotting
        fig1, axs = plt.subplots(2, 2)
        axs[0, 0].plot(x[0], y[0], 'o')
        axs[0, 0].plot(xeval[0], f(xeval[0]), 'g')
        axs[0, 0].plot(xeval[0], yeval[0], 'r')
        axs[0, 0].set_title('N=5')
        axs[0, 1].plot(x[1], y[1], 'o')
        axs[0, 1].plot(xeval[1], f(xeval[1]), 'g')
        axs[0, 1].plot(xeval[1], yeval[1], 'r')
        axs[0, 1].set_title('N=10')
        axs[1, 0].plot(x[2], y[2], 'o')
        axs[1, 0].plot(xeval[2], f(xeval[2]), 'g')
        axs[1, 0].plot(xeval[2], yeval[2], 'r')
        axs[1, 0].set_title('N=15')
        axs[1, 1].plot(x[3], y[3], 'o')
        axs[1, 1].plot(xeval[3], f(xeval[3]), 'g')
        axs[1, 1].plot(xeval[3], yeval[3], 'r')
        axs[1, 1].set_title('N=20')
        for ax in axs.flat:
            ax.label_outer()
        fig2, axs = plt.subplots(2, 2)
        axs[0, 0].plot(xeval[0], np.log10(error[0]), 'b')
        axs[0, 0].set_title('$Log_{10}$ Error for N=5')
        axs[0, 1].plot(xeval[1], np.log10(error[1]), 'b')
        axs[0, 1].set_title('$Log_{10}$ Error for N=10')
        axs[1, 0].plot(xeval[2], np.log10(error[2]), 'b')
        axs[1, 0].set_title('$Log_{10}$ Error for N=15')
        axs[1, 1].plot(xeval[3], np.log10(error[3]), 'b')
        axs[1, 1].set_title('$Log_{10}$ Error for N=20')
        for ax in axs.flat:
            ax.label_outer()

#subroutines
def equispaced(a,b,n):
    #use a formula to generate n equispaced points between a and b
    x = np.zeros(n)
    for i in range(1, n + 1):
        x[i - 1] = a + (i - 1) * ((b-a) / (n - 1))
    return x

def chebyshev(a,b,n):
    #use a formula to generate n Chebyshev nodes between a and b
    x = np.zeros(n)
    for i in range(1, n + 1):
        x[i - 1] = (a+b)/2 + ((b-a)/2)*np.cos(((2 * i -1) * np.pi) / (2 * n))
    x = np.sort(x)
    return x

def weights(x):
    # this function calculates the normalized weights for Lagrange interpolation
    w = np.ones(len(x))
    for i in range(len(x)):
        for j in range(len(x)):
            if i == j:
                continue
            w[i] = w[i] * (1 / (x[i] - x[j]))
    return w

def lagrange_eval(xeval, x, y):
    # this function evaluates a polynomial interpolant using barycentric Lagrange
    # INPUTS:    xeval   an array of points to interpolate the polynomial
    #           x       an array of interpolation nodes
    #           y       an array of function values at interpolation nodes
    yeval = np.zeros(len(xeval))  # preallocate
    w = weights(x);  # find normalized weights
    for i in range(len(xeval)):  # loop over evaluation points for interpolation
        num = 0;
        denom = 0;
        flag = 0;
        for j in range(len(x)):  # loop over interpolant nodes for each point
            if xeval[i] == x[j]:
                flag = 1
                yeval[i] = y[j]
            if flag != 1:
                step = w[j] / (xeval[i] - x[j])
                num = num + step * y[j]
                denom = denom + step
        if flag != 1:
            yeval[i] = num / denom
    return yeval


def eval_hermite(xeval, xint, yint, ypint, N):
    #this function was provided by professor.  It evaluates a Hermite interpolation
    #polynomial at a specified point
    ''' Evaluate all Lagrange polynomials'''

    lj = np.ones(N + 1)
    for count in range(N + 1):
        for jj in range(N + 1):
            if (jj != count):
                lj[count] = lj[count] * (xeval - xint[jj]) / (xint[count] - xint[jj])

    ''' Construct the l_j'(x_j)'''
    lpj = np.zeros(N + 1)
    #    lpj2 = np.ones(N+1)
    for count in range(N + 1):
        for jj in range(N + 1):
            if (jj != count):
                #              lpj2[count] = lpj2[count]*(xint[count] - xint[jj])
                lpj[count] = lpj[count] + 1. / (xint[count] - xint[jj])

    yeval = 0.

    for jj in range(N + 1):
        Qj = (1. - 2. * (xeval - xint[jj]) * lpj[jj]) * lj[jj] ** 2
        Rj = (xeval - xint[jj]) * lj[jj] ** 2
        #       if (jj == 0):
        #         print(Qj)

        #         print(Rj)
        #         print(Qj)
        #         print(xeval)
        #        return
        yeval = yeval + yint[jj] * Qj + ypint[jj] * Rj

    return (yeval)


def create_natural_spline(yint, xint, N):
    #    create the right  hand side for the linear system
    b = np.zeros(N + 1)
    #  vector values
    h = np.zeros(N + 1)
    for i in range(1, N):
        hi = xint[i] - xint[i - 1]
        hip = xint[i + 1] - xint[i]
        b[i] = (yint[i + 1] - yint[i]) / hip - (yint[i] - yint[i - 1]) / hi
        h[i - 1] = hi
        h[i] = hip

    #  create matrix so you can solve for the M values
    # This is made by filling one row at a time
    A = np.zeros((N + 1, N + 1))
    A[0][0] = 1.0
    for j in range(1, N):
        A[j][j - 1] = h[j - 1] / 6
        A[j][j] = (h[j] + h[j - 1]) / 3
        A[j][j + 1] = h[j] / 6
    A[N][N] = 1

    #Ainv = inv(A)

    #M = Ainv.dot(b)
    M = lg.solve(A,b)
    #  Create the linear coefficients
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
        C[j] = yint[j] / h[j] - h[j] * M[j] / 6
        D[j] = yint[j + 1] / h[j] - h[j] * M[j + 1] / 6
    return (M, C, D)


def eval_local_spline(xeval, xi, xip, Mi, Mip, C, D):
    # Evaluates the local spline as defined in class
    # xip = x_{i+1}; xi = x_i
    # Mip = M_{i+1}; Mi = M_i

    hi = xip - xi
    yeval = (Mi * (xip - xeval) ** 3 + (xeval - xi) ** 3 * Mip) / (6 * hi) \
            + C * (xip - xeval) + D * (xeval - xi)
    return yeval


def eval_cubic_spline(xeval, Neval, xint, Nint, M, C, D):
    yeval = np.zeros(Neval + 1)

    for j in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        atmp = xint[j]
        btmp = xint[j + 1]

        #   find indices of values of xeval in the interval
        ind = np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]

        # evaluate the spline
        yloc = eval_local_spline(xloc, atmp, btmp, M[j], M[j + 1], C[j], D[j])
        #        print('yloc = ', yloc)
        #   copy into yeval
        yeval[ind] = yloc

    return (yeval)

#prob1a(vrb)
#prob1b(vrb)
#prob1c(vrb)
#prob2a(vrb)
#prob2b(vrb)
prob2c(vrb)
if vrb:
    plt.show()