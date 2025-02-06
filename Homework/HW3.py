#By:        Shane Billingsley
#Class:     APPM4600 Numerical Analysis and Scientific Computing
#Date:      2-7-2025
#File:      HW 3 assignment

import numpy as np
import matplotlib.pyplot as plt
vrb = True

#define subroutine for individual questions
def prob1(vrb):
    f = lambda x: 2*x - 1 - np.sin(x)  # set function definition
    a = 0.5;    b = 1;  #set interval of interest
    tol = 1e-9          #set tolerance
    #call bisection to find root r
    [r,ier,r_iter] = bisection(f,a,b,tol)
    if vrb:
        print(f"The root is {r}")
        print(f"The error message is {ier}")
        print(f"The number of iterations is {r_iter.size-1}")

def prob2(vrb):
    f = lambda x: (x-5)**9  #set function definition
    a = 4.82;   b = 5.2;    #set interval of interest
    tol = 1e-4              #set tolerance
    [r_f,ier_f,r_iter_f] = bisection(f,a,b,tol)
    if vrb:
        print(f"The root is {r_f}")
        print(f"The error message is {ier_f}")
        print(f"The number of iterations is {r_iter_f.size-1}")

    #expanded function definition
    g = lambda x: x**9 - 45*x**8 + 900*x**7 - 10500*x**6 + 78750*x**5 - 393750*x**4 + 1312500*x**3 - 2812500*x**2 + 3515625*x - 1953125
    [r_g, ier_g, r_iter_g] = bisection(g, a, b, tol)
    if vrb:
        print(r_iter_g)
        print(f"The root is {r_g}")
        print(f"The error message is {ier_g}")
        print(f"The number of iterations is {r_iter_g.size-1}")

def prob3(vrb):
    f = lambda x: x**3 + x - 4  # set function definition
    a = 1;
    b = 4;  # set interval of interest
    tol = 0.5e-3  # set tolerance
    [r, ier, r_iter] = bisection(f, a, b, tol)
    if vrb:
        print(f"The root is {r}")
        print(f"The error message is {ier}")
        print(f"The number of iterations is {r_iter.size-1}")

def prob5(vrb):
    f = lambda x: x - 4*np.sin(2*x) - 3     #set function definition
    x = np.linspace(-2,8,1000)
    y = f(x)
    if vrb:
        plt.plot(x,y)
        plt.axhline(y=0,color='r')
        plt.title("Plot of $x - 4sin(2x)-3$")
    g = lambda x: -np.sin(2*x) + (5/4)*x - (3/4)    #set fixed pt iteration function
    guess = np.array([-0.90,-0.55,1.72,3.14,4.55])   #set initial guesses
    #set tolerance and max iterations
    tol = 1e-11;    Nmax = 100;
    for i in range(guess.size):
        [root,ier,iter] = fixedpt(g,guess[i],tol,Nmax)
        if vrb:
            print(f"Root for guess {guess[i]} is {root}")
            print(f"Number of iterations is {iter.size-1}")

# define routines
def bisection(f, a, b, tol, vrb=False):
    #    Inputs:
    #     f,a,b       - function and endpoints of initial interval
    #      tol  - bisection stops when interval length < tol

    #    Returns:
    #      astar - approximation of root
    #      ier   - error message
    #            - ier = 1 => Failed
    #            - ier = 0 == success
    #      iter  - all iterated values in order

    #     first verify there is a root we can find in the interval

    fa = f(a)
    fb = f(b);
    iter = []
    if (fa * fb > 0):
        ier = 1
        astar = a
        return [astar, ier,iter]

    #   verify end points are not a root
    if (fa == 0):
        astar = a
        iter = np.append(iter,a)
        ier = 0
        return [astar, ier,iter]

    if (fb == 0):
        astar = b
        iter = np.append(iter, b)
        ier = 0
        return [astar, ier,iter]

    count = 0
    d = 0.5 * (a + b)
    iter = np.append(iter, d)
    while (abs(d - a) > tol):
        fd = f(d)
        if (fd == 0):
            astar = d
            ier = 0
            return [astar, ier, iter]
        if (fa * fd < 0):
            b = d
        else:
            a = d
            fa = fd
        d = 0.5 * (a + b)
        iter = np.append(iter, d)
        count = count + 1
        if vrb:
            print('abs(d-a) = ', abs(d-a))

    astar = d
    ier = 0
    return [astar, ier, iter]

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

#execute subroutines
#prob1(vrb)
#prob2(vrb)
#prob3(vrb)
prob5(vrb)
if vrb:
    plt.show()