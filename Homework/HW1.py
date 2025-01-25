#By:        Shane Billingsley
#Class:     APPM4600 Numerical Analysis and Scientific Computing
#Date:      1-24-2025
#File:      Homework 1 assignment

import numpy as np
import matplotlib.pyplot as plt
#Question 1: Evaluate and plot a ninth order polynomial in condensed form (x-2)^9,
#and in expanded form using coefficients

def prob1():
    #coefficients of ninth order polynomial
    coeff = np.array([1,-18,144,-672,2016,-4032,5376,-4608,2304,-512])
    #exponents of ninth order polynomial
    exponents = np.array([9,8,7,6,5,4,3,2,1,0])

    x = np.arange(1.920,2.080,0.001)    #values of independent variable x
    y1 = np.zeros(np.shape(x)[0])       #preallocate array for function results
    # evaluate polynomial in expanded form
    for i in x:
        vec = np.zeros(10)  #preallocate
        vec.fill(i)         #current value of x
        # calculate f(x) in expanded form using coefficients
        y1[np.where(x == i)] = np.sum(np.multiply(np.power(vec,exponents),coeff))
    # evaluate polynomial in condensed form
    y2 = (x - 2)**9

    plt.plot(x,y1,label='Expanded')
    plt.plot(x,y2,label='Condensed')
    plt.legend()
    plt.title("Ninth Order Polynomial in Expanded and Condensed Form")

def prob5b():
    #evaluate a given function and then a modified version to avoid cancellation
    #plot differences between the two for two values of x and a log range of delta
    x1 = np.pi; x2 = 10^6   #values of x
    delta = np.power(10.0,np.arange(-16,1,))    #log range of delta
    result1_orig = np.cos(x1 + delta) - np.cos(x1)  #original function
    result1_mod = -2*np.sin(x1 + (delta/2))*np.sin(delta/2)     #modified version
    result1_diff = np.abs(result1_orig - result1_mod)   #difference

    plt.figure()
    plt.semilogx(delta, result1_diff, label="$x = \\pi$")
    plt.legend()
    plt.title("Difference Between Original and Modified Function")
    plt.xlabel(r"$\delta$")
    plt.ylabel("Difference")

    #repeat for second x-value
    result2_orig = np.cos(x2 + delta) - np.cos(x2)  #original function
    result2_mod = -2 * np.sin(x2 + (delta / 2)) * np.sin(delta / 2) #modified version
    result2_diff = np.abs(result2_orig - result2_mod)   #difference

    plt.figure()
    plt.semilogx(delta, result2_diff, label="$x = 10^6$")
    plt.legend()
    plt.title(r"Difference Between Original and Modified Function")
    plt.xlabel(r"$\delta$")
    plt.ylabel("Difference")

def prob5c():
    #repeat calculations for 5b but now with an additional Taylor approximation.  Compare
    x1 = np.pi; x2 = 10 ^ 6
    delta = np.power(10.0, np.arange(-16, 1, ))     #repeated calcs
    result1_orig = np.cos(x1 + delta) - np.cos(x1)
    result1_mod = -2 * np.sin(x1 + (delta / 2)) * np.sin(delta / 2)
    #Taylor approximation using ksi = x + delta/2
    result1_tay = -(np.sin(x1)*delta) - (np.cos(x1 + (delta / 2))*((delta**2) / 2))

    plt.figure()
    plt.semilogx(delta, np.abs(result1_tay - result1_orig), label="$x = \\pi$")
    plt.legend()
    plt.title("Difference Between Original Function and Taylor Approx")
    plt.xlabel(r"$\delta$")
    plt.ylabel("Difference")

    plt.figure()
    plt.semilogx(delta, np.abs(result1_tay - result1_mod), label="$x = \\pi$")
    plt.legend()
    plt.title("Difference Between Modified Function and Taylor Approx")
    plt.xlabel(r"$\delta$")
    plt.ylabel("Difference")

    #repeat for second x-value
    result2_orig = np.cos(x2 + delta) - np.cos(x2)
    result2_mod = -2 * np.sin(x2 + (delta / 2)) * np.sin(delta / 2)
    result2_tay = -(np.sin(x2) * delta) - (np.cos(x2 + (delta / 2)) * ((delta ** 2) / 2))

    plt.figure()
    plt.semilogx(delta, np.abs(result2_tay - result2_orig), label="$x = 10^6$")
    plt.legend()
    plt.title("Difference Between Original Function and Taylor Approx")
    plt.xlabel(r"$\delta$")
    plt.ylabel("Difference")

    plt.figure()
    plt.semilogx(delta, np.abs(result2_tay - result2_mod), label="$x = 10^6$")
    plt.legend()
    plt.title("Difference Between Modified Function and Taylor Approx")
    plt.xlabel(r"$\delta$")
    plt.ylabel("Difference")

#run all problems, comment out to run by sections
prob1()
prob5b()
prob5c()
plt.show()