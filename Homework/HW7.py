#By:        Shane Billingsley
#Class:     APPM4600 Numerical Analysis and Scientific Computing
#Date:      3-07-2025
#File:      HW 7 assignment

import numpy as np
from numpy import linalg as lg
import matplotlib.pyplot as plt
import scipy.special as sp
vrb = True

def prob1(vrb):
    #evaluate and plot a given function using monomial expansion
    f = lambda x: 1 / (1 + (10 * x) ** 2)   #
    N = np.linspace(2,20,19)    #range of N(number of nodes)
    x = [None]*len(N);  y = [None]*len(N);              #preallocate storage
    xeval = [None]*len(N);  yeval = [None]*len(N);
    error = [None]*len(N);
    for n in N:                     #loop through increasing number of nodes
        n = int(n)                  #cast to integer for indexing
        x[n-2] = np.zeros(n)        #preallocate for node locations
        for i in range(1,n+1):      #create node locations using given formula
            x[n-2][i-1] = -1 + (i-1)*(2/(n-1))
        y[n-2] = f(x[n-2])          #find function values at node locations
        xeval[n-2] = np.linspace(-1,1,1001)     #create interpolation points
        coeffs = mono_exp(x[n-2],y[n-2],False)      #define polynomial coefficients
        if vrb:
            print(coeffs)
        yeval[n-2] = poly_eval(xeval[n-2],coeffs)       #interpolate values at nodes
        error[n-2] = np.abs(yeval[n-2] - f(xeval[n-2])) #find error as |p(x) - f(x)|
    if vrb:     #plotting
        fig1, axs = plt.subplots(2, 2)
        axs[0,0].plot(x[0],y[0],'o')
        axs[0,0].plot(xeval[0],f(xeval[0]),'g')
        axs[0,0].plot(xeval[0],yeval[0],'r')
        axs[0, 0].set_title('N=2')
        axs[0, 1].plot(x[1], y[1], 'o')
        axs[0, 1].plot(xeval[1], f(xeval[1]), 'g')
        axs[0, 1].plot(xeval[1], yeval[1], 'r')
        axs[0, 1].set_title('N=3')
        axs[1, 0].plot(x[4], y[4], 'o')
        axs[1, 0].plot(xeval[4], f(xeval[4]), 'g')
        axs[1, 0].plot(xeval[4], yeval[4], 'r')
        axs[1, 0].set_title('N=6')
        axs[1, 1].plot(x[8], y[8], 'o')
        axs[1, 1].plot(xeval[8], f(xeval[8]), 'g')
        axs[1, 1].plot(xeval[8], yeval[8], 'r')
        axs[1, 1].set_title('N=10')
        for ax in axs.flat:
            ax.label_outer()

        fig2, axs = plt.subplots(2, 2)
        axs[0, 0].plot(x[8], y[8], 'o')
        axs[0, 0].plot(xeval[8], error[8], 'b')
        axs[0, 0].set_title('Error for N=10')
        axs[0, 1].plot(x[12], y[12], 'o')
        axs[0, 1].plot(xeval[12], error[12], 'b')
        axs[0, 1].set_title('Error for N=14')
        axs[1, 0].plot(x[14], y[14], 'o')
        axs[1, 0].plot(xeval[14], error[14], 'b')
        axs[1, 0].set_title('Error for N=16')
        axs[1, 1].plot(x[16], y[16], 'o')
        axs[1, 1].plot(xeval[16], error[16], 'b')
        axs[1, 1].set_title('Error for N=18')
        for ax in axs.flat:
            ax.label_outer()

        plt.figure(3)
        plt.plot(x[17],y[17],'o')
        plt.plot(xeval[17], f(xeval[17]), 'g')
        plt.plot(xeval[17], yeval[17], 'r')
        plt.title('N=19')

def prob2(vrb):
    # evaluate and plot a given function using barycentric Lagrange
    f = lambda x: 1 / (1 + (10 * x) ** 2)  #
    N = np.linspace(2, 20, 19)  # range of N(number of nodes)
    x = [None] * len(N);    y = [None] * len(N);  # preallocate storage
    xeval = [None] * len(N);    yeval = [None] * len(N);
    error = [None] * len(N);
    for n in N:  # loop through increasing number of nodes
        n = int(n)  # cast to integer for indexing
        x[n - 2] = np.zeros(n)  # preallocate for node locations
        for i in range(1, n + 1):  # create node locations using given formula
            x[n - 2][i - 1] = -1 + (i - 1) * (2 / (n - 1))
        y[n - 2] = f(x[n - 2])  # find function values at node locations
        xeval[n - 2] = np.linspace(-1, 1, 1001)  # create interpolation points
        yeval[n - 2] = lagrange_eval(xeval[n - 2],x[n-2],y[n-2])  # interpolate values at nodes
        error[n - 2] = np.abs(yeval[n - 2] - f(xeval[n - 2]))  # find error as |p(x) - f(x)|
    if vrb:  # plotting
        fig1, axs = plt.subplots(2, 2)
        axs[0, 0].plot(x[0], y[0], 'o')
        axs[0, 0].plot(xeval[0], f(xeval[0]), 'g')
        axs[0, 0].plot(xeval[0], yeval[0], 'r')
        axs[0, 0].set_title('N=2')
        axs[0, 1].plot(x[1], y[1], 'o')
        axs[0, 1].plot(xeval[1], f(xeval[1]), 'g')
        axs[0, 1].plot(xeval[1], yeval[1], 'r')
        axs[0, 1].set_title('N=3')
        axs[1, 0].plot(x[4], y[4], 'o')
        axs[1, 0].plot(xeval[4], f(xeval[4]), 'g')
        axs[1, 0].plot(xeval[4], yeval[4], 'r')
        axs[1, 0].set_title('N=6')
        axs[1, 1].plot(x[8], y[8], 'o')
        axs[1, 1].plot(xeval[8], f(xeval[8]), 'g')
        axs[1, 1].plot(xeval[8], yeval[8], 'r')
        axs[1, 1].set_title('N=10')
        for ax in axs.flat:
            ax.label_outer()

        fig2, axs = plt.subplots(2, 2)
        axs[0, 0].plot(x[8], y[8], 'o')
        axs[0, 0].plot(xeval[8], np.log10(error[8]), 'b')
        axs[0, 0].set_title('$Log_{10}$ Error for N=10')
        axs[0, 1].plot(x[12], y[12], 'o')
        axs[0, 1].plot(xeval[12], np.log10(error[12]), 'b')
        axs[0, 1].set_title('$Log_{10}$ Error for N=14')
        axs[1, 0].plot(x[14], y[14], 'o')
        axs[1, 0].plot(xeval[14], np.log10(error[14]), 'b')
        axs[1, 0].set_title('$Log_{10}$ Error for N=16')
        axs[1, 1].plot(x[16], y[16], 'o')
        axs[1, 1].plot(xeval[16], np.log10(error[16]), 'b')
        axs[1, 1].set_title('$Log_{10}$ Error for N=18')
        for ax in axs.flat:
            ax.label_outer()
        #rerun interpolation for large N
        N=200
        x = np.zeros(N);   y = np.zeros(N);
        for i in range(1, N + 1):  # create node locations using given formula
            x[i - 1] = -1 + (i - 1) * (2 / (N - 1))     #define x-values for nodes
        y = f(x)                                        #define y-values for nodes
        xeval = np.linspace(-1, 1, 1001)     #define evaluation points
        yeval = lagrange_eval(xeval, x, y)  # interpolate values at nodes
        error = np.abs(yeval - f(xeval))  # find error as |p(x) - f(x)|
        plt.figure(3)
        plt.plot(x, y, 'o')
        plt.plot(xeval, np.log10(error), 'b')
        plt.title('$Log_{10}$ Error for N=200')

def prob3(vrb):
    #repeat problem 2 using Chebyshev nodes
    f = lambda x: 1 / (1 + (10 * x) ** 2)  #
    N = np.linspace(2, 20, 19)  # range of N(number of nodes)
    x = [None] * len(N);  y = [None] * len(N);# preallocate storage
    xeval = [None] * len(N);  yeval = [None] * len(N);
    error = [None] * len(N);
    for n in N:
        n = int(n)  # cast to integer for indexing
        x[n-2] = np.zeros(n)        #preallocate for node locations
        for i in range(1,n+1):      #generate Chebyshev nodes
            x[n-2][i-1] = np.cos(((2*i -1)*np.pi)/(2*n))
        y[n - 2] = f(x[n - 2])  # find function values at node locations
        xeval[n - 2] = np.linspace(-1, 1, 1001)  # create interpolation points
        yeval[n - 2] = lagrange_eval(xeval[n - 2], x[n - 2], y[n - 2])  # interpolate values at nodes
        error[n - 2] = np.abs(yeval[n - 2] - f(xeval[n - 2]))  # find error as |p(x) - f(x)|
    if vrb:  # plotting
        fig1, axs = plt.subplots(2, 2)
        axs[0, 0].plot(x[0], y[0], 'o')
        axs[0, 0].plot(xeval[0], f(xeval[0]), 'g')
        axs[0, 0].plot(xeval[0], yeval[0], 'r')
        axs[0, 0].set_title('N=2')
        axs[0, 1].plot(x[1], y[1], 'o')
        axs[0, 1].plot(xeval[1], f(xeval[1]), 'g')
        axs[0, 1].plot(xeval[1], yeval[1], 'r')
        axs[0, 1].set_title('N=3')
        axs[1, 0].plot(x[4], y[4], 'o')
        axs[1, 0].plot(xeval[4], f(xeval[4]), 'g')
        axs[1, 0].plot(xeval[4], yeval[4], 'r')
        axs[1, 0].set_title('N=6')
        axs[1, 1].plot(x[8], y[8], 'o')
        axs[1, 1].plot(xeval[8], f(xeval[8]), 'g')
        axs[1, 1].plot(xeval[8], yeval[8], 'r')
        axs[1, 1].set_title('N=10')
        for ax in axs.flat:
            ax.label_outer()

        fig2, axs = plt.subplots(2, 2)
        axs[0, 0].plot(x[8], y[8], 'o')
        axs[0, 0].plot(xeval[8], np.log10(error[8]), 'b')
        axs[0, 0].set_title('$Log_{10}$ Error for N=10')
        axs[0, 1].plot(x[12], y[12], 'o')
        axs[0, 1].plot(xeval[12], np.log10(error[12]), 'b')
        axs[0, 1].set_title('$Log_{10}$ Error for N=14')
        axs[1, 0].plot(x[14], y[14], 'o')
        axs[1, 0].plot(xeval[14], np.log10(error[14]), 'b')
        axs[1, 0].set_title('$Log_{10}$ Error for N=16')
        axs[1, 1].plot(x[16], y[16], 'o')
        axs[1, 1].plot(xeval[16], np.log10(error[16]), 'b')
        axs[1, 1].set_title('$Log_{10}$ Error for N=18')
        for ax in axs.flat:
            ax.label_outer()
        # rerun interpolation for large N
        N = 200
        x = np.zeros(N);
        y = np.zeros(N);
        for i in range(1, N + 1):  # create node locations using given formula
            x[i-1] = np.cos(((2*i -1)*np.pi)/(2*N)) #define Chebyshev nodes
        y = f(x)  # define y-values for nodes
        xeval = np.linspace(-1, 1, 1001)  # define evaluation points
        yeval = lagrange_eval(xeval, x, y)  # interpolate values at nodes
        error = np.abs(yeval - f(xeval))  # find error as |p(x) - f(x)|
        plt.figure(3)
        plt.plot(x, y, 'o')
        plt.plot(xeval, np.log10(error), 'b')
        plt.title('$Log_{10}$ Error for N=200')


#subroutines
def vandermonde(x):
    #this function returns the vandermonde matrix
    #x is expected as a 1D numpy array
    n = len(x)
    V = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            V[j,i] = x[j]**(i)
    return V

def mono_exp(x,y,vrb):
    #this function returns a vector of coefficients of a monomial expansion
    #x and y are expected as numpy arrays of equal dimension
    V = vandermonde(x)
    if vrb:
        print(V)
    return lg.solve(V,y)

def poly_eval(x,coeff):
    #this function evaluates a polynomial at points x using given coefficients
    y = np.zeros(len(x))
    for i in range(len(x)):
        for j in range(len(coeff)):
            y[i] = y[i] + (x[i]**j)*coeff[j]
    return y

def weights(x):
    #this function calculates the normalized weights for Lagrange interpolation
    w = np.ones(len(x))
    for i in range(len(x)):
        for j in range(len(x)):
            if i == j:
                continue
            w[i] = w[i]*(1/(x[i] - x[j]))
    return w

def lagrange_eval(xeval,x,y):
    #this function evaluates a polynomial interpolant using barycentric Lagrange
    #INPUTS:    xeval   an array of points to interpolate the polynomial
    #           x       an array of interpolation nodes
    #           y       an array of function values at interpolation nodes
    yeval = np.zeros(len(xeval))    #preallocate
    w = weights(x);                 #find normalized weights
    for i in range(len(xeval)):     #loop over evaluation points for interpolation
        num = 0;
        denom = 0;
        flag = 0;
        for j in range(len(x)):     #loop over interpolant nodes for each point
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


#prob1(vrb)
#prob2(vrb)
prob3(vrb)
if vrb:
    plt.show()