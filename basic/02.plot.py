import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,2*np.pi,100)
y = np.sin(x)

fig, axis =plt.subplots()

axis.set_title('Plot of sin(x)')
axis.set_xlabel('x')
axis.set_ylabel('y')

plt.plot(x, y)

plt.show(block=False)
plt.pause(4)
plt.close()