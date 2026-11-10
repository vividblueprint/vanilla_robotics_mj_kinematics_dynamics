from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np

def func(x):
    return x**2 - x - 2

x = np.arange(-6,6,0.01)   # start, stop, step
y = func(x)

plt.figure(1)
plt.plot(x,y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Plot of func")
plt.grid()
# plt.show(block=False)
# plt.pause(20)
# plt.close()

x0 = -40 #guess
root = fsolve(func, x0)
plt.scatter(root, func(root), c="r")
print(root)
plt.show()
