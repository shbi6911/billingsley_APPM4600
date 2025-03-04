#By:        Shane Billingsley (some example code provided by instructor)
#Class:     APPM4600 Numerical Analysis and Scientific Computing
#Date:      3-4-2025
#File:      Lab 8 assignment

import numpy as np
from numpy import linalg as lg
import matplotlib.pyplot as plt
import scipy.special as sp
vrb = True

import math
from numpy.linalg import inv

def driver(vrb):
    f = lambda x: np.exp(x)

    a = 0
    b = 1
    #create points you want to evaluate at
    Neval = 100
    xeval = np.linspace(a, b, Neval)
    #number of intervals
    Nint = 10
    #evaluate the linear spline
    yeval = eval_lin_spline(xeval, Neval, a, b, f, Nint)
    #evaluate f at the evaluation points
    fex = f(xeval)

    plt.figure()
    plt.plot(xeval, fex,'ro-')
    plt.plot(xeval, yeval,'bs-')
    plt.legend(['Function','Spline'])
    plt.show()
    err = abs(yeval - fex)
    plt.figure()
    plt.plot(xeval, err,'ro-')
    plt.show()



#subroutines
def line_eval(x1,y1,x2,y2,alpha):
    return ((y2-y1)/(x2-x1))*(alpha - x1) + y1

def eval_lin_spline(xeval,Neval,a,b,f,Nint):
    #create the intervals for piecewise approximations
    xint = np.linspace(a, b, Nint + 1)
    #create vector to store the evaluation of the linear splines
    yeval = np.zeros(Neval)
    for j in range(Nint):
        #find indices of xeval in interval (xint(jint),xint(jint+1))
        #let ind denote the indices in the intervals
        atmp = xint[j]
        btmp = xint[j + 1]
        # find indices of values of xeval in the interval
        ind = np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]
        n = len(xloc)
        #temporarily store your info for creating a line in the interval of interest
        fa = f(atmp)
        fb = f(btmp)
        yloc = np.zeros(len(xloc))
        for kk in range(n):
        # use your line evaluator to evaluate the spline at each location
        # Call your line evaluator with points (atmp,fa) and (btmp,fb)
            yloc[kk] =  line_eval(atmp,fa,btmp,fb,xloc[kk])
        # Copy yloc into the final vector
        yeval[ind] = yloc
        return yeval

driver(vrb)