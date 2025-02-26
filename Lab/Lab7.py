#By:        Shane Billingsley (some example code provided by instructor)
#Class:     APPM4600 Numerical Analysis and Scientific Computing
#Date:      2-24-2025
#File:      Lab 7 assignment

import numpy as np
from numpy import linalg as lg
import matplotlib.pyplot as plt
import scipy.special as sp
vrb = True

def prelab(vrb):
    x = np.array([1,2,3,4,5])
    y = np.array([2,1,4,3,10])
    x2 = np.array([1.5,2.5,3.5,4.5,5.5])
    V = vandermonde(x)
    A = lg.solve(V,y)
    V2 = vandermonde(x2)
    y2 = V2@A
    if vrb:
        plt.plot(x2,y2)

def exercise3(vrb):
    f = lambda x: 1/(1 + (10*x)**2)
    N = 4           #set number of interpolation nodes
    x_dat = points(N)        #generate x-values for nodes
    y_dat = f(x_dat)            #generate y-values for nodes
    #generate Vandermonde coefficients
    V = vandermonde(x_dat)
    A = lg.solve(V,y_dat)
    #generate vector of interpolation points
    x = np.linspace(-1,1,1000)
    y = eval_monomial(x,A,N)
    y_real = f(x)
    if vrb:
        plt.plot(x,y_real)
        plt.plot(x,y)


def points(N):
    x = np.zeros(N)
    h = 2/(N-1)
    for j in range(N):
        x[j] = -1 +(j-1)*h
    return x

#routines
def vandermonde(x):
    #this function returns the coefficients of a vandermonde polynomial
    #x is expected as a 1D numpy array
    n = len(x)
    V = np.ones(n)
    for i in range(n-1):
        V = np.concatenate((V,x**(i+1)))
    V.shape = (n,n)
    V = np.transpose(V)
    return V

#instructor provided function to evaluate interpolant
def eval_monomial(xeval, coef, N, Neval):
    yeval = coef[0] * np.ones(Neval + 1)

    #    print('yeval = ', yeval)

    for j in range(1, N + 1):
        for i in range(Neval + 1):
            #        print('yeval[i] = ', yeval[i])
            #        print('a[j] = ', a[j])
            #        print('i = ', i)
            #        print('xeval[i] = ', xeval[i])
            yeval[i] = yeval[i] + coef[j] * xeval[i] ** j

    return yeval

exercise3(vrb)
#prelab(vrb)
if vrb:
    plt.show()