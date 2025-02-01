#By:        Shane Billingsley
#Class:     APPM4600 Numerical Analysis and Scientific Computing
#Date:      1-31-2025
#File:      Homework 2 assignment

import numpy as np
import matplotlib.pyplot as plt
import random as rand
import math as math

vrb = True              #set verbosity flag
def prob2(vrb):
    #Question 2: calculate some things for matrix equation
    A = np.array([[1,1],[1 + 10**-10,1 - 10**-10]])
    condition = np.linalg.cond(A)
    if vrb:
        print(f"Condition number is {condition}")
    delta_b = np.array([3e-5,6e-5])
    b = np.array([1,1])         #same as outupt x when unperturbed
    err_b = np.linalg.norm(delta_b)/np.linalg.norm(b)
    if vrb:
        print(f"Relative error of b is {err_b}")
    delta_x = np.linalg.solve(A,delta_b)
    err_x = np.linalg.norm(delta_x)/np.linalg.norm(b)
    if vrb:
        print(f"Relative error of x is {err_x}")
    ineq = err_b*condition
    if vrb:
        print(f"Relative error of input*Condition # is {ineq}")

def func(x):
    y = math.e**x
    return y-1

def prob3(vrb):
    x = 9.999999995000000e-10
    y = func(x)
    if vrb:
        print(y)
    y2 = (x + (1/math.factorial(2))*(x**2))
    if vrb:
        print(y2)
def prob4a(vrb):
    # Question 4a:    Evaluate a sum of two given arrays
    t = np.linspace(0,np.pi,31)
    y = np.cos(t)
    S = np.sum(np.multiply(t,y))
    #print(t);    print(y);    print(S)
    if vrb:
        print(f"The Sum is {S}.")

# Question 4b:  Evaluate a given expression and plot, first with fixed parameter values
# and then in a for loop with varying values
def prob4b(vrb):
    #set initial values
    theta = np.linspace(0,2*np.pi,1000)
    R = 1.2;   delta_r = 0.1;  f = 15; p = 0;

    #evaluate expression in two steps for readability
    sub_x = 1 + delta_r*(np.sin(f*theta + p))
    x = R*sub_x*np.cos(theta)

    sub_y = 1 + delta_r * (np.sin(f * theta + p))
    y = R * sub_y * np.sin(theta)

    if vrb:
        #plot the parametric curve above
        fig, ax = plt.subplots()
        ax.plot(x,y)
        ax.axis('equal')
        plt.title("Parametric Curve with Fixed Values")

        #repeat above calculation in a for loop with varying parameters
        delta_r = 0.05
        fig,ax = plt.subplots() #initialize figure
        results_list = [None]*10    #preallocate for results
        for i in range(1,11):
            #repeat parametrization with varying values based on loop variable
            p = rand.uniform(0, 2); #random p-value
            sub_x = 1 + delta_r * (np.sin((2+i) * theta + p))
            x = i * sub_x * np.cos(theta)
            sub_y = 1 + delta_r * (np.sin((2+i) * theta + p))
            y = i * sub_y * np.sin(theta)
            ax.plot(x,y)    #plot in same axes
            results_list[i-1] = np.array([x,y])
        ax.axis('equal')
        plt.title ("Parametric Curves with Varying Values")


#run appropriate sections (comment out as needed)
#prob2(vrb)
prob3(vrb)
#prob4a(vrb)
#prob4b(vrb)
if vrb:
    plt.show()