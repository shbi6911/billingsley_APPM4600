#By:        Shane Billingsley (some example code provided by instructor)
#Class:     APPM4600 Numerical Analysis and Scientific Computing
#Date:      3-18-2025
#File:      Lab 10 assignment

import numpy as np
import math
from numpy import linalg as la
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.integrate import quad
vrb = True


def driver():
    #  function you want to approximate
    #f = lambda x: math.exp(x)
    f = lambda x: 1/(1+x**2)
    # Interval of interest
    a = -1
    b = 1
    # weight function
    w = lambda x: 1.

    # order of approximation
    n = 4

    #  Number of points you want to sample in [a,b]
    N = 1000
    xeval = np.linspace(a, b, N + 1)
    pval = np.zeros(N + 1)

    for kk in range(N + 1):
        pval[kk] = eval_legendre_expansion(f, a, b, w, n, xeval[kk])

    ''' create vector with exact values'''
    fex = np.zeros(N + 1)
    for kk in range(N + 1):
        fex[kk] = f(xeval[kk])

    plt.figure()
    plt.plot(xeval, fex, 'ro-', label='f(x)')
    plt.plot(xeval, pval, 'bs--', label='Expansion')
    plt.legend()
    plt.show()

    err = abs(pval - fex)
    plt.semilogy(xeval, err_l, 'ro--', label='error')
    plt.legend()
    plt.show()


# subroutines
def eval_legendre_expansion(f, a, b, w, n, x):
    #   This subroutine evaluates the Legendre expansion

    #  Evaluate all the Legendre polynomials at x that are needed
    # by calling your code from prelab
    p = eval_legendre(n,x)
    # initialize the sum to 0
    pval = 0.0
    for j in range(0, n + 1):
        # make a function handle for evaluating phi_j(x)
        phi_j = lambda x: j_leg(j,x)
        # make a function handle for evaluating phi_j^2(x)*w(x)
        phi_j_sq = lambda x: (j_leg(j,x)**2)*w(x)
        # use the quad function from scipy to evaluate normalizations
        norm_fac, err = quad(phi_j_sq,a,b)
        # make a function handle for phi_j(x)*f(x)*w(x)/norm_fac
        func_j = lambda x: (phi_j(x)*f(x)*w(x))/norm_fac
        # use the quad function from scipy to evaluate coeffs
        aj, err = quad(func_j,a,b)
        # accumulate into pval
        pval = pval + aj * p[j]

    return pval

def eval_legendre(N,x):
    #this function evaluates Legendre polynomials at a point x
    #n is the order of polynomials to evaluate up to and x is the evaluation point
    #it returns a vector p with values of all Legendre polynomials up to order n at point x
    p = np.zeros(N+1)
    for i in range(len(p)):
        if i == 0:
            p[i] = 1
        elif i == 1:
            p[i] = x
        else:
            n = i-1
            p[i] = (1/(n+1))*((((2*n)+1)*x*p[n]) - n*p[n-1])
    return p

def j_leg(j,x):
    p = eval_legendre(j,x)
    return(p[j])

driver()

