#By:        Shane Billingsley (some example code provided by instructor)
#Class:     APPM4600 Numerical Analysis and Scientific Computing
#Date:      1-28-2025
#File:      Lab 3 assignment

import numpy as np


def prob1():
    f = lambda x: (x**2)*(x-1)      #set function definition
    a = 0.5                          #set interval of interest
    b = 2

    tol = 1e-7                      #set tolerance

    [astar, ier] = bisection(f, a, b, tol)  #call bisection method function
    print('f(x) = x^2*(x-1) on [0.5,2]')    #print output
    print('the approximate root is', astar)
    print('the error message reads:', ier)
    print('f(astar) =', f(astar))

    f = lambda x: (x ** 2) * (x - 1)  # set function definition
    a = -1  # set interval of interest
    b = 0.5

    tol = 1e-7  # set tolerance

    [astar, ier] = bisection(f, a, b, tol)  # call bisection method function
    print('f(x) = x^2*(x-1) on [-1,0.5]')  # print output
    print('the approximate root is', astar)
    print('the error message reads:', ier)
    print('f(astar) =', f(astar))

    f = lambda x: (x ** 2) * (x - 1)  # set function definition
    a = -1  # set interval of interest
    b = 2

    tol = 1e-7  # set tolerance

    [astar, ier] = bisection(f, a, b, tol)  # call bisection method function
    print('f(x) = x^2*(x-1) on [-1,2]')  # print output
    print('the approximate root is', astar)
    print('the error message reads:', ier)
    print('f(astar) =', f(astar))

def prob2():
    tol = 1e-5  # set tolerance

    f = lambda x: (x-1)*(x-3)*(x-5)  # set function definition
    a = 0  # set interval of interest
    b = 2.4

    [astar, ier] = bisection(f, a, b, tol)  # call bisection method function
    print('f(x) = (x-1)*(x-3)*(x-5) on [0,2.4]')  # print output
    print('the approximate root is', astar)
    print('the error message reads:', ier)
    print('f(astar) =', f(astar))

    f = lambda x: ((x - 1)**2)*(x-3)  # set function definition
    a = 0  # set interval of interest
    b = 2

    [astar, ier] = bisection(f, a, b, tol)  # call bisection method function
    print('f(x) = ((x - 1)**2)*(x-3) on [0,2]')  # print output
    print('the approximate root is', astar)
    print('the error message reads:', ier)
    print('f(astar) =', f(astar))

    f = lambda x: np.sin(x)  # set function definition
    a = 0  # set interval of interest
    b = 0.1

    [astar, ier] = bisection(f, a, b, tol)  # call bisection method function
    print('f(x) = sin(x) on [0,0.1]')  # print output
    print('the approximate root is', astar)
    print('the error message reads:', ier)
    print('f(astar) =', f(astar))

    f = lambda x: np.sin(x)  # set function definition
    a = 0.5  # set interval of interest
    b = (3*np.pi)/4

    [astar, ier] = bisection(f, a, b, tol)  # call bisection method function
    print('f(x) = sin(x) on [0.5,(3*pi)/4]')  # print output
    print('the approximate root is', astar)
    print('the error message reads:', ier)
    print('f(astar) =', f(astar))

def prob3():
    Nmax = 100  #set max iterations and tolerance
    tol = 1e-10

    f = lambda x: x*(1 + ((7-x**5)/x**2))**3  # set function definition

    x0 = 1.0    #set starting point
    [xstar, ier] = fixedpt(f, x0, tol, Nmax)
    print('f(x) = x*(1 + ((7-x**5)/x**2))**3')  # print output
    print('the approximate fixed point is:', xstar)
    print('f1(xstar):', f1(xstar))
    print('Error message reads:', ier)

    f = lambda x: x - ((x**5 - 7)/x**2)  # set function definition

    x0 = 1.0  # set starting point
    [xstar, ier] = fixedpt(f, x0, tol, Nmax)
    print('f(x) = x - ((x**5 - 7)/x**2)')  # print output
    print('the approximate fixed point is:', xstar)
    print('f1(xstar):', f1(xstar))
    print('Error message reads:', ier)

    f = lambda x: x - ((x ** 5 - 7) / (5*x ** 4))  # set function definition

    x0 = 1.0  # set starting point
    [xstar, ier] = fixedpt(f, x0, tol, Nmax)
    print('f(x) = x - ((x ** 5 - 7) / (5*x ** 4))')  # print output
    print('the approximate fixed point is:', xstar)
    print('f1(xstar):', f1(xstar))
    print('Error message reads:', ier)

    f = lambda x: x - ((x ** 5 - 7) / 12)  # set function definition

    x0 = 1.0  # set starting point
    [xstar, ier] = fixedpt(f, x0, tol, Nmax)
    print('f(x) = x - ((x ** 5 - 7) / 12)')  # print output
    print('the approximate fixed point is:', xstar)
    print('f1(xstar):', f1(xstar))
    print('Error message reads:', ier)

# define routines
def bisection(f, a, b, tol):
    #    Inputs:
    #     f,a,b       - function and endpoints of initial interval
    #      tol  - bisection stops when interval length < tol

    #    Returns:
    #      astar - approximation of root
    #      ier   - error message
    #            - ier = 1 => Failed
    #            - ier = 0 == success

    #     first verify there is a root we can find in the interval

    fa = f(a)
    fb = f(b);
    if (fa * fb > 0):
        ier = 1
        astar = a
        return [astar, ier]

    #   verify end points are not a root
    if (fa == 0):
        astar = a
        ier = 0
        return [astar, ier]

    if (fb == 0):
        astar = b
        ier = 0
        return [astar, ier]

    count = 0
    d = 0.5 * (a + b)
    while (abs(d - a) > tol):
        fd = f(d)
        if (fd == 0):
            astar = d
            ier = 0
            return [astar, ier]
        if (fa * fd < 0):
            b = d
        else:
            a = d
            fa = fd
        d = 0.5 * (a + b)
        count = count + 1
    #      print('abs(d-a) = ', abs(d-a))

    astar = d
    ier = 0
    return [astar, ier]

def fixedpt(f,x0,tol,Nmax):

    ''' x0 = initial guess'''
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    count = 0
    while (count <Nmax):
       count = count +1
       try:
           x1 = f(x0)
       except OverflowError as e:
           print("OverflowError:", e)
           return [xstar,ier]
       x1 = f(x0)
       if (abs(x1-x0) <tol):
          xstar = x1
          ier = 0
          return [xstar,ier]
       x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier]

#prob1()
#prob2()
prob3()