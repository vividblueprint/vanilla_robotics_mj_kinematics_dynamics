from email.iterators import body_line_iterator

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x = np.linspace(-10, 10, 200)
y = x**2

fig, ax = plt.subplots()
ax.set_title("y=x^2")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)

ax.set_xlim(-10, 10)
ax.set_ylim(0, 110)

line, = ax.plot([], [], linewidth=2)

def init():
    line.set_data([], [])
    return line,

def update(frame):
    line.set_data(x[:frame], y[:frame])
    return line,

ani = FuncAnimation(
    fig,
    update,
    frames=len(x),
    init_func=init,
    interval=20,
    blit=True
)

plt.show()