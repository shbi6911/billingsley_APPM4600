#By:        Shane Billingsley (some example code provided by instructor)
#Class:     APPM4600 Numerical Analysis and Scientific Computing
#Date:      4-25-2025
#File:      Homework 11 assignment

import numpy as np
import math
from scipy import linalg as la
import matplotlib.pyplot as plt
import scipy.special as sp
vrb = True

def prob3(vrb):
    #use power method to find dominant eigenvalue of hilbert matrix
    #report number of iterations
    N = np.arange(4, 21, 4)
    for i in range(len(N)):
        H = la.hilbert(N[i])
        [eigval,_,count] = method_of_might(H)
        if vrb:
            print(f"Eigenvalue is {eigval} with {count} iterations")

    H = la.hilbert(16)
    [eigval,_,count] = method_of_might(H,inv=1)
    if vrb:
        print(f"Smallest Eigenvalue is {eigval} with {count} iterations")




#subroutines
def method_of_might(A, tol=1e-16, Nmax=1000,inv = 0):
    #use the power method to find the dominant eigenpair
    #input inv is a flag for normal power method (=0, default) or inverse power method (=1)
    n = A.shape[0]
    x = np.ones(n)  # Initial vector
    x = x / np.linalg.norm(x)  # Normalize initial vector
    lambda_old = 0;     lambda_new = 1;     count = 0;      #initialize

    while abs(lambda_new - lambda_old) > tol and count < Nmax:
        if inv == 0:
            y = A @ x
        elif inv == 1:
            y = np.linalg.solve(A, x)

        eig_vec = y / np.linalg.norm(y)  # Normalize the result
        lambda_new = eig_vec.T @ A @ eig_vec  # Rayleigh quotient
        count = count +1            #iterate counter
        lambda_old = lambda_new
    return lambda_new, eig_vec, count

prob3(vrb)
