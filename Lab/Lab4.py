#By:        Shane Billingsley (some example code provided by instructor)
#Class:     APPM4600 Numerical Analysis and Scientific Computing
#Date:      2-4-2025
#File:      Lab 4 assignment

import numpy as np
import matplotlib.pyplot as plt
vrb = True

def exercise2(vrb):
    #test functions
    f1 = lambda x: np.sqrt(10/(x+4))
    # fixed point is alpha1 = 1.4987....
    Nmax = 100
    tol = 1e-10
    x0 = 1.5

    [point, ier, iter] = fixedpt(f1,x0,tol,Nmax)
    if vrb:
        for i in iter:
            print(i)
        print(f"fixed point is {point}")
        print(f"Error message is {ier}")
        print(f"Number of iterations: {iter.size}")
    alpha = np.zeros(iter.size-2)   #preallocate
    for i in range(iter.size-2):
        top = np.abs(f1(iter[i+1])-point)/np.abs(f1(iter[i+2])-point)
        bottom = np.abs(f1(iter[i])-point)/np.abs(f1(iter[i+1])-point)
        alpha[i] = np.log10(top)/np.log10(bottom)
    lambda_const = (np.abs(f1(iter[1+1])-point)/np.abs(f1(iter[1+2])-point))/(np.abs(f1(iter[1])-point)/np.abs(f1(iter[1+1])-point))
    if vrb:
        print(alpha)
        print(lambda_const)

# define routines
def fixedpt(f,x0,tol,Nmax):

    ''' x0 = initial guess'''
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''
    iter = []
    count = 0
    while (count <Nmax):
       count = count +1
       x1 = f(x0)
       iter = np.append(iter,x1)
       if (abs(x1-x0) <tol):
          xstar = x1
          ier = 0
          return [xstar,ier,iter]
       x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, iter]



exercise2(vrb)