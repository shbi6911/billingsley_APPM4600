#By:        Shane Billingsley (some example code provided by instructor)
#Class:     APPM4600 Numerical Analysis and Scientific Computing
#Date:      2-18-2025
#File:      Lab 6 assignment

import numpy as np
from numpy import linalg as lg
import matplotlib.pyplot as plt
import scipy.special as sp
vrb = True

#explore different Quasi-Newton methods

#prelab - explore finite difference schemes for computing approximate derivatives
def prelab(vrb=False):
    #define an array of h-values to use with differences
    h = 0.01*2.**(-np.arange(0,10))
    if vrb:
        print(h)
    #define function to approximate the derivatives of
    f = lambda x: np.cos(x)
    x = np.pi/2

    #define forward difference scheme
    f_prime_forward = (f(x + h) - f(x))/h
    err_forward = np.abs(f_prime_forward - (-1))
    if vrb:
        print(f_prime_forward)
    #define centered difference scheme
    f_prime_centered = (f(x + h) - f(x - h)) / (2*h)
    err_centered = np.abs(f_prime_centered - (-1))
    if vrb:
        print(f_prime_centered)

    if vrb:
        plt.figure()
        plt.plot(h,err_forward)
        plt.plot(h,err_centered)

def exercise3_2(vrb=False):
    #define some functions
    def F(x):
        return np.array(np.array([4*x[0]**2 + x[1]**2 - 4,x[0] + x[1] - np.sin(x[0] - x[1])]))
    def JF(x):
        return np.array([[(8*x[0]),(2*x[1])],[(1 - np.cos(x[0] - x[1])),(1 + np.cos(x[0] - x[1]))]])
    x0 = np.array([1,0]);   tol = 1e-10;    Nmax = 100
    [root1,rn1,_,_] = slacker_newton_nd(F,JF,x0,tol,Nmax,vrb)

# Slacker Newton method in n dimensions implementation
def slacker_newton_nd(f,Jf,x0,tol,nmax,vrb=False):

    # Initialize arrays and function value
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    n=0;
    nf=1; nJ=0; #function and Jacobian evals
    npn=1;

    if (len(x0)<100):
        if (np.linalg.cond(Jf(x0)) > 1e16):
            print("Error: matrix too close to singular");
            print("Newton method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
            r=x0;
            return (r,rn,nf,nJ);

    if vrb:
        print("|--n--|----xn----|---|f(xn)|---|");

    while npn>tol and n<=nmax:
        if (np.mod(n,5) == 0):
            # compute n x n Jacobian matrix
            Jn = Jf(xn);
            nJ+=1;

        if vrb:
            print("|--%d--|%1.7f|%1.15f|" %(n,np.linalg.norm(xn),np.linalg.norm(Fn)));

        # Newton step (we could check whether Jn is close to singular here)
        pn = -np.linalg.solve(Jn,Fn);
        xn = xn + pn;
        npn = np.linalg.norm(pn); #size of Newton step

        n+=1;
        rn = np.vstack((rn,xn));
        Fn = f(xn);
        nf+=1;

    r=xn;

    if vrb:
        if np.linalg.norm(Fn)>tol:
            print("Newton method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Newton method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return (r,rn,nf,nJ);

#prelab(True)
exercise3_2(vrb)
if vrb:
    plt.show()