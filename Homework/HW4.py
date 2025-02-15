#By:        Shane Billingsley
#Class:     APPM4600 Numerical Analysis and Scientific Computing
#Date:      2-14-2025
#File:      HW 4 assignment

import numpy as np
import matplotlib.pyplot as plt
from scipy import special as sp
vrb = True

#define subroutine for individual questions
def prob1(vrb=False):
#evaluate a temperature function using specified givens, and find roots using various methods

#define some given values and functions
    Ts = -15;   Ti = 20;    alpha = 0.138e-6;    t=518400
    tol = 1e-13;     a = 0;     b = 3;       Nmax = 500
    def temp(x):
        return (sp.erf(x/(2*np.sqrt(alpha*t)))*(Ti - Ts)) + Ts
    def d_temp(x):
        return ((2*(Ti - Ts))/np.sqrt(np.pi))*np.exp(-(x/(2*np.sqrt(alpha*t)))**2)
    x_bar = np.linspace(a,b,100)
    if vrb:
        plt.plot(x_bar,temp(x_bar))
        plt.title("Temperature vs. Depth @ t=60 days")
        plt.xlabel("Depth (m)")
        plt.ylabel("Temperature $^{\\circ}C$")
    [bi_root,ier,bi_iter] = bisection(temp,a,b,tol)
    if vrb:
        print(f"Bisection finds the root {bi_root}")
    [noot_root,noot_iter,_] = newton_method(temp,d_temp,0.01,tol,Nmax)
    if vrb:
        print(f"Newton's method finds the root {noot_root}")
    [noot_root2,_,_] = newton_method(temp,d_temp,b,tol,Nmax,True)

def prob4(vrb=False):
    f = lambda x: (np.exp(x)-(3*x**2))**3
    df = lambda x: 3*((np.exp(x)-(3*x**2))**2)*(np.exp(x)-(6*x))
    ddf = lambda x: 6*(np.exp(x)-(3*x**2))*((np.exp(x)-(6*x))**2) + 3*(np.exp(x)-(3*x**2))*(np.exp(x)-6)
    x0 = 5;  tol = 1e-15;    Nmax = 500;    m = 3
    [noot_root1, noot_iter1, _] = newton_method(f, df,x0, tol, Nmax)
    err1 = np.abs(noot_iter1[0:-1] - noot_root1)
    if vrb:
        print(f"Root for unmodified Newton's Method is {noot_root1}")
    g = lambda x: f(x)/df(x)
    dg = lambda x: (df(x)**2 - (ddf(x)*f(x)))/(df(x)**2)
    [noot_root2, noot_iter2, _] = newton_method(g, dg, x0, tol, Nmax)
    err2 = np.abs(noot_iter2[0:-1] - noot_root2)
    if vrb:
        print(f"Root for f(x)/f'(x) Newton's Method is {noot_root2}")
    h = lambda x: m*f(x)
    dh = df
    [noot_root3, noot_iter3, _] = newton_method(h, dh, x0, tol, Nmax)
    err3 = np.abs(noot_iter3[0:-1] - noot_root3)
    if vrb:
        print(f"Root for m*f(x) Newton's Method is {noot_root3}")
    if vrb:
        plt.semilogy(err1)
        plt.semilogy(err2)
        plt.semilogy(err3)
        plt.ylabel("$|x_n - r|$")
        plt.xlabel("Iterations")
        plt.title("Comparison of Modified Newton's Methods")
        plt.legend(["Unmodified","f(x)/f'(x)","mf(x)/f'(x)"])

def prob5(vrb=False):
    f = lambda x: x**6 - x - 1
    df = lambda x: 6*x**5 - 1
    x0 = 2;     x1 = 1;     tol = 1e-15;     Nmax = 500;
    [noot_root, noot_iter, _] = newton_method(f, df, x0, tol, Nmax)
    [sec_root,sec_iter,_] = secant_method(f,x0,x1,tol,Nmax)
    if vrb:
        print("Newton iterations")
        print(np.abs(noot_iter-noot_root))
        print("Secant iterations")
        print(np.abs(sec_iter-sec_root))
        plt.figure()
        plt.loglog(np.abs(noot_iter[1:-1]-noot_root),np.abs(noot_iter[0:-2] - noot_root))
        plt.axis('square')
        plt.ylabel("$|x_{k+1} - r|$")
        plt.xlabel("$|x_k - r|$")
        plt.title("Newton's method")
        plt.figure()
        plt.loglog(np.abs(sec_iter[1:-1]-sec_root),np.abs(sec_iter[0:-2] - sec_root))
        plt.axis('square')
        plt.ylabel("$|x_{k+1} - r|$")
        plt.xlabel("$|x_k - r|$")
        plt.title("Secant method")
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

def newton_method(f,df,x0,tol,nmax,vrb=False):
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
            print('\n derivative at initial guess is near 0, try different x0 \n');
            return (xn,rn,nfun)
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

def secant_method(f,x0,x1,tol,nmax,vrb=False):
    #secant (quasi-newton) method to find root of f starting with guesses x0 and x1

    #Initialize iterates and iterate list
    xnm=x0; xn=x1;
    rn=np.array([x1]);
    # function evaluations
    fn=f(xn); fnm=f(xnm);
    msec = (fn-fnm)/(xn-xnm);
    nfun=2; #evaluation counter nfun
    dtol=1e-10; #tolerance for derivative (being near 0)

    if np.abs(msec)<dtol:
        #If slope of secant is too small, secant will fail. Error message is
        #displayed and code terminates.
        if vrb:
            print('\n slope of secant at initial guess is near 0, try different x0,x1 \n');
    else:
        n=0;
        if vrb:
            print("\n|--n--|----xn----|---|f(xn)|---|---|msec|---|");

        #Iteration runs until f(xn) is small enough or nmax iterations are computed.

        while n<=nmax:
            if vrb:
                print("|--%d--|%1.8f|%1.8f|%1.8f|" %(n,xn,np.abs(fn),np.abs(msec)));

            pn = - fn/msec; #Secant step
            if np.abs(pn)<tol or np.abs(fn)<2e-15:
                break;

            #Update guess adding Newton step, update xn-1
            xnm = xn; #xn-1 is now xn
            xn = xn + pn; #xn is now xn+pn

            # Update info and loop
            n+=1;
            rn=np.append(rn,xn);
            fnm = fn; #Note we can re-use this function evaluation
            fn=f(xn); #So, only one extra evaluation is needed per iteration
            msec = (fn-fnm)/(xn-xnm); # New slope of secant line
            nfun+=1;

        r=xn;

        if n>=nmax:
            print("Secant method failed to converge, niter=%d, nfun=%d, f(r)=%1.1e\n'" %(n,nfun,np.abs(fn)));
        else:
            print("Secant method converged succesfully, niter=%d, nfun=%d, f(r)=%1.1e" %(n,nfun,np.abs(fn)));

    return (r,rn,nfun)

#prob1(vrb)
#prob4(vrb)
prob5(vrb)
if vrb:
    plt.show()