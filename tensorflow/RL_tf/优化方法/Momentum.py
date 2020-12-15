import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x[0]*x[0]+50*x[1]*x[1]


def g(x):
    return np.array([2*x[0], 100*x[1]])


def momentum(x_start, step, g, discount=0.7):
    x = np.array(x_start, dtype='float64')
    pre_grad = np.zeros_like(x)
    for i in range(50):
        grad = g(x)
        pre_grad = pre_grad*discount+grad*step
        x -= pre_grad
        print("[ Epoch{0} ] grad = {1}, x = {2} ".format(i, grad, x))
        if abs(sum(grad)) < 1e-6:
            break
    return x


def nesterov(x_start, step, g, discount=0.7):
    x = np.array(x_start, dtype='float64')
    pre_grad = np.zeros_like(x)
    for i in range(50):
        x_future = x-step*discount*pre_grad
        grad = g(x_future)
        pre_grad = pre_grad*0.7+grad
        x -= pre_grad*step
        print("[Epoch{0}] grad={1}, x={2}".format(i, grad, x))
        if abs(sum(grad)) < 1e-6:
            break
    return x


def plot_contour(X, Y, Z, x_array):
    #fig = plt.figure(figsize=(15, 7))
    plt.contour(X, Y, Z, 10)
    plt.scatter(0, 0, marker='*', s=50, c='r', label='Optimal')

    x_array=np.array(x_array)
    for j in range(len(x_array)-1):
        plt.plot(x_array[j:j+2, 0], x_array[j:j+2, 1])
        plt.scatter(x_array[j,0],x_array[j,1],c='k')
        plt.scatter(x_array[j+1,0],x_array[j+1,1],c='k')
        plt.pause(0.8)

        plt.legend(loc='best')


xi = np.linspace(-200, 200, 1000)
yi = np.linspace(-100, 100, 1000)
X, Y = np.meshgrid(xi, yi)
Z = X*X+50*Y*Y
x_arr=nesterov([150,75],0.014,g)
plot_contour(X,Y,Z,x_arr)
plt.show()