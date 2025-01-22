#By:        Shane Billingsley
#Class:     APPM4600 Numerical Analysis and Scientific Computing
#Date:      1-14-2025
#File:      Lab 1 assignment

import numpy as np
import numpy.linalg as la
import math

#run given code to define an take dot product of two vectors
def driver():
    n = 100
    x = np.linspace(0,np.pi,n)
# this is a function handle. You can use it to define
# functions instead of using a subroutine like you
# have to in a true low level language.
    f = lambda x: x**2 + 4*x + 2*np.exp(x)
    g = lambda x: 6*x**3 + 2*np.sin(x)
    y = f(x)
    w = g(x)
# evaluate the dot product of y and w
    dp = dotProduct(y,w,n)
# print the output
    print('the dot product is : ', dp)
    return

def dotProduct(x,y,n):
# Computes the dot product of the n x 1 vectors x and y
    dp = 0.
    for j in range(n):
        dp = dp + x[j]*y[j]
    return dp

driver()

#modify given code to mulitply two orthogonal vectors
def driver():
    n = 3
    #x = np.linspace(0,np.pi,n)
# this is a function handle. You can use it to define
# functions instead of using a subroutine like you
# have to in a true low level language.
    #f = lambda x: x**2 + 4*x + 2*np.exp(x)
    #g = lambda x: 6*x**3 + 2*np.sin(x)
    y = np.array([0,1,0])
    w = np.array([1,0,0])
# evaluate the dot product of y and w
    dp = dotProduct(y,w,n)
# print the output
    print('the dot product is : ', dp)
    return

def dotProduct(x,y,n):
# Computes the dot product of the n x 1 vectors x and y
    dp = 0.
    for j in range(n):
        dp = dp + x[j]*y[j]
    return dp

driver()

#modify provided code to multiply two matrices
def matrixMult(x,y):
    output = np.zeros([x.shape[0],y.shape[1]])
    if x.shape[1] != y.shape[0]:
        print("Incompatible Matrices!")
        return(output)
    for r in range(x.shape[0]):
        for c in range(y.shape[1]):
            output[r,c] = dotProduct(x[r,:],y[:,c],x.shape[1])
    return(output)

#test modified code
x = np.array([[2,2],[2,2]])
y = np.array([[3,3],[3,3]])
#result should be [[12,12],[12,12]]
print(matrixMult(x,y))

x = np.array([[1,2],[3,4],[5,6]])
y = np.array([[9,8,7,6],[5,4,3,2]])
#result should be [[19,16,13,10],[47,40,33,26],[75,64,53,42]]
print(matrixMult(x,y))

#test code execution times versus numpy built-in functions
xvec = np.zeros(10)
xvec.fill(5)
yvec = np.zeros(10)
yvec.fill(2)

import time
start_time1 = time.time()
dp = dotProduct(xvec,yvec,xvec.size)
print('the dot product is : ', dp)
end_time1 = time.time()
execution_time1 = end_time1 - start_time1
print("Execution time:", execution_time1)

start_time2 = time.time()
dp = la.multi_dot([xvec,yvec])
print('the dot product is : ', dp)
end_time2 = time.time()
execution_time2 = end_time2 - start_time2
print("Execution time:", execution_time2)

xmat = np.array([[1,2],[3,4],[5,6]])
ymat = np.array([[9,8,7,6],[5,4,3,2]])

start_time3 = time.time()
mp = matrixMult(xmat,ymat)
print('the matrix product is : ', mp)
end_time3 = time.time()
execution_time3 = end_time3 - start_time3
print("Execution time:", execution_time3)

start_time4 = time.time()
mp = np.matmul(xmat,ymat)
print('the matrix product is : ', mp)
end_time4 = time.time()
execution_time4 = end_time4 - start_time4
print("Execution time:", execution_time4)


