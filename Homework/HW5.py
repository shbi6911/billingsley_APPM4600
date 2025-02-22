#By:        Shane Billingsley
#Class:     APPM4600 Numerical Analysis and Scientific Computing
#Date:      2-21-2025
#File:      HW 5 assignment

import numpy as np
from numpy import linalg as lg
import matplotlib.pyplot as plt
import scipy.special as sp
vrb = True

#test Newton's method for nonlinear systems using given equations
def prob1(vrb=False):
    def F(x):           #define given two-dimensional function
        return np.array([3*x[0]**2 - x[1]**2,3*x[0]*x[1]**2 - x[0]**3 -1])
    J = np.array([[(1/6),(1/18)],[0,(1/6)]])    #define given Jacobian inverse
    rn = np.zeros(502)                          #preallocate iteration storage
    x0_vec = np.array([1,1]);  rn[0] = lg.norm(x0_vec)   #initial guess
    Nmax = 500; tol = 1e-10;    N=1;    #iteration variables
    x_vec = x0_vec - J @ F(x0_vec);    rn[N] = lg.norm(x_vec)#first iteration
    while (lg.norm(F(x_vec)) > tol and N<=Nmax):
        x_vec = x_vec - J @ F(x_vec)    #find new iterate
        N += 1  # increment counter
        rn[N] = lg.norm(x_vec)      #store norm of current iterate
    rn = np.delete(rn,(rn == 0))
    rn_err = np.abs(rn - lg.norm(x_vec))
    rn_log_err = np.log10(rn_err)
    rn_log_err = np.delete(rn_log_err,np.isinf(rn_log_err))
    slope = rn_log_err[1:-1] / rn_log_err[0:-2]
    if vrb:
        if lg.norm(F(x_vec)) < tol:
            print(f"Converged successfully to root [{x_vec}")
        if N > Nmax:
            print(f"Unsuccessful convergence: too many iterations")
        plt.figure(1)
        plt.plot(rn_log_err)
        plt.title("Convergence of Given Iteration Scheme")
        plt.ylabel("$log_{10}(|x_n - r|)$")
        plt.xlabel("Iterations")
        print(np.mean(slope))
    a = np.array([[6, -2], [0, 6]])
    b = lg.inv(a)
    if vrb:
        print(f"Inverse of J_0 is {b}")
    #define Jacobian function for given function
    def JF(x):
        y = x[1];   x = x[0];
        return np.array([[6*x,-2*y],[3*y**2 - 3*x**2,6*x*y]])
    #use Newton's method to iterate
    [noot_root,noot_rn,_,_] = newton_method_nd(F,JF,x0_vec,tol,Nmax,vrb)
    #find error and plot convergence
    noot_rn_err = np.abs(lg.norm((noot_rn - noot_root),axis=1))
    noot_rn_log_err = np.log10(noot_rn_err)
    #noot_rn_log_err = np.delete(noot_rn_log_err, np.isinf(noot_rn_log_err))
    if vrb:
        print(noot_rn_log_err)
        plt.figure(2)
        plt.plot(noot_rn_log_err)
        plt.title("Convergence of Newton's Method")
        plt.ylabel("$log_{10}(|x_n - r|)$")
        plt.xlabel("Iterations")
        print(noot_root)

def prob3(vrb=False):
#iterate to a point on an ellipse, in a hurry so no vectorizing
    def f(x,y,z):
        return x**2 + 4*y**2 + 4*z**2 -16
    fx = lambda x: 2*x
    fy = lambda y: 8*y
    fz = lambda z: 8*z
    def d(x, y, z):
        return (f(x,y,z))/((fx(x)**2 + fy(y)**2 + fz(z)**2))
    x0 = 1;    y0 = 1;  z0 = 1;     tol = 1e-10;    Nmax = 100;
    count = 0;  norms = np.zeros(101); norms[count] = lg.norm(np.array([x0,y0,z0]));
    x = x0 - d(x0,y0,z0)*fx(x0);    y = y0 - d(x0,y0,z0)*fy(y0);    z = z0 - d(x0,y0,z0)*fz(z0);
    norms[count] = lg.norm(np.array([x,y,z]));  count += 1
    while (f(x,y,z) > tol and count <= Nmax):
        x = x - d(x, y, z) * fx(x);
        y = y - d(x, y, z) * fy(y);
        z = z - d(x, y, z) * fz(z);
        norms[count] = lg.norm(np.array([x,y,z]))
        count += 1
    norms = np.delete(norms, (norms == 0))
    err = norms - lg.norm(np.array([x,y,z]))
    log10_err = np.log10(err)
    if vrb:
        print(norms)
        print(f"The root is x={x},y={y},z={z}, with {count} iterations")
        plt.plot(log10_err)
        plt.xlabel("Iterations")
        plt.ylabel("$log_{10}||x_n - r||$")

# Newton method in n dimensions implementation
def newton_method_nd(f,Jf,x0,tol,nmax,vrb=False):

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

#prob1(vrb)
prob3(vrb)
if vrb:
    plt.show()