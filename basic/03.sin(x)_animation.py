from email.iterators import body_line_iterator

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x = np.linspace(0, 2*np.pi, 200)
y = np.sin(x)

fig, ax = plt.subplots()
ax.set_title("y=sin(x)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(False)

ax.set_xlim(0, 2*np.pi)
ax.set_ylim(-1.2, 1.2)

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