#By:        Shane Billingsley (some example code provided by instructor)
#Class:     APPM4600 Numerical Analysis and Scientific Computing
#Date:      4-11-2025
#File:      Homework 11 assignment

import numpy as np
import math
from numpy import linalg as la
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.integrate import quad
from numpy.polynomial.legendre import leggauss as lgwt
vrb = True

def prob1(vrb):
    #integrate a given function using trapezoidal, simpson's, and scipy quad integration
    #compare error
    f = lambda x:   1/(1+x**2)      #function of interest
    m_trap = 1291;         #number of integration points
    m_simp = 54;
    a = -5;     b = 5;      #interval of interest
    T_n = trapz(f,a,b,m_trap)       #integrate with trapezoidal approx
    S_n = simpz(f,a,b,m_simp)       #integrate with simpson's approx
    Q_n1,_,Q1_info = quad(f,a,b,full_output=1,epsabs=1e-4);  Q1_neval = Q1_info["neval"]
    Q_n2,_,Q2_info = quad(f, a, b, full_output=1, epsabs=1e-6); Q2_neval = Q2_info["neval"]
    if vrb:
        print(f"Trapezoidal produced {T_n} with {m_trap+1} function evaluations")
        print(f"Simpson's produced {S_n} with {2*m_simp+1} function evaluations")
        print(f"Scipy Quad with epsabs 10^-4 produced {Q_n1} with {Q1_neval} function evaluations")
        print(f"Scipy Quad with epsabs 10^-6 produced {Q_n2} with {Q2_neval} function evaluations")
def prob2(vrb):
    f = lambda x:   x*np.cos(1/x)
    a = 0;  b = 1;  m = 2
    N = 2 * m + 1  # number of points is 2 times number of intervals +1
    x = np.linspace(a, b, N)
    h = (b - a) / (2 * m)
    w = np.ones(N)
    for i in range(1, (N - 1)):
        if i % 2 == 0:
            w[i] = 2
        elif i % 2 == 1:
            w[i] = 4
    w = (h / 3) * w
    x[1:] = f(x[1:])
    S_n = np.sum(w*x)
    if vrb:
        print(S_n)

#subroutines
def trapz(f,a,b,m):
    # calculates composite trapezoidal approximation to the integral on a,b
    # using m subintervals
    N = m+1     #number of points is one more than number of intervals
    x = np.linspace(a,b,N)
    h = (b-a)/(m)
    w = np.ones(N)
    w[0] = 0.5; w[-1] = 0.5;
    w = h*w
    return np.sum(w*f(x))

def simpz(f,a,b,m):
    # calculates composite simpson's approximation to the integral on a,b
    # using m subintervals
    N = 2*m + 1     #number of points is 2 times number of intervals +1
    x = np.linspace(a, b, N)
    h = (b - a) / (2*m)
    w = np.ones(N)
    for i in range(1,(N-1)):
        if i % 2 == 0:
            w[i] = 2
        elif i % 2 == 1:
            w[i] = 4
        else:
            print("Error")
            return 0
    w = (h/3)*w
    return np.sum(w*f(x))

#prob1(vrb)
prob2(vrb)