#By:        Shane Billingsley (some example code provided by instructor)
#Class:     APPM4600 Numerical Analysis and Scientific Computing
#Date:      2-11-2025
#File:      Lab 5 assignment

import numpy as np
import matplotlib.pyplot as plt
vrb = True

#construct a hybrid bisection/Newton method rootfinding function

#set intial variables and functions of interest
def exercise(vrb):
    Nmax = 100;
    tol = 1e-10;
    f = lambda x: np.expm1(x**2 +(7*x) -30)
    df = lambda x: (2*x + 7)*np.exp(x**2 +(7*x) - 30)
    a = 2;      b = 4.5
    #call rootfinders and compare
    [bi_root,_] = bisection(f,a,b,tol)
    [newt_root,_,_] = newton_method(f,df,b,tol,Nmax)
    [hyb_root] = hybridRoot(f,df,a,b,tol,Nmax)
    if vrb:
        print(f"Root found by bisection is {bi_root}")
        print(f"Root found by Newton's method is {newt_root}")
        print(f"Root found by hybrid method is {hyb_root}")

# define routines
def hybridRoot(f,df,a,b,tol,Nmax,vrb=False):
#a rootfinding algorithm combining bisection and Newton's method
    bi_tol = 1e-1
    [bi_result,ier] = bisection(f,a,b,bi_tol)
    if ier == 1:
        astar = a
        print("No root in interval")
        return [astar, ier]
    [root,_,_] = newton_method(f, df, bi_result, tol, Nmax)
    return [root]

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

def newton_method(f, df, x0, tol, nmax, vrb=False):
    #newton method to find root of f starting at guess x0

    #Initialize iterates and iterate list
    xn=x0;
    rn=np.array([x0]);
    # function evaluations
    fn=f(xn); dfn=df(xn);
    nfun=2; #evaluation counter nfun
    dtol=1e-10; #tolerance for derivative (being near 0)

    if abs(dfn)<dtol:
        #If derivative is too small, Newton will fail. Error message is
        #displayed and code terminates.
        if vrb:
            print(f'\nderivative at initial guess is near 0, try different x0');
    else:
        n=0;
        if vrb:
            print("\n|--n--|----xn----|---|f(xn)|---|---|f'(xn)|---|");

        #Iteration runs until f(xn) is small enough or nmax iterations are computed.

        while n<=nmax:
            if vrb:
                print("|--%d--|%1.8f|%1.8f|%1.8f|" %(n,xn,np.abs(fn),np.abs(dfn)));

            pn = - fn/dfn; #Newton step
            if np.abs(pn)<tol or np.abs(fn)<2e-15:
                break;

            #Update guess adding Newton step
            xn = xn + pn;

            # Update info and loop
            n+=1;
            rn=np.append(rn,xn);
            dfn=df(xn);
            fn=f(xn);
            nfun+=2;

        r=xn;

        if n>=nmax:
            print("Newton method failed to converge, niter=%d, nfun=%d, f(r)=%1.1e\n'" %(n,nfun,np.abs(fn)));
        else:
            print("Newton method converged succesfully, niter=%d, nfun=%d, f(r)=%1.1e" %(n,nfun,np.abs(fn)));

    return (r,rn,nfun)

#run main code
exercise(vrb)