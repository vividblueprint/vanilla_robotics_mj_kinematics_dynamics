import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Generate the Sun's trajectory
x = np.linspace(0, 2 * np.pi, 200)
y = np.sin(x)

# Create figure and axes
fig, ax = plt.subplots()
ax.set_title("Sunrise")
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.set_aspect("equal")
ax.grid(False)

# Ground
rect = plt.Rectangle(
    (0, 0), 2, 1,
    color="brown",
    zorder=2
)
ax.add_patch(rect)

# Horizon
line = plt.Line2D(
    [0, 2], [1, 1],
    linewidth=3,
    color="black",
    zorder=3
)
ax.add_line(line)

# Sun
circle = plt.Circle(
    (1, 1),
    radius=0.25,
    color="yellow",
    zorder=1
)
ax.add_patch(circle)

def init():
    circle.set_center((1, 1))
    return circle,

def update(frame):
    # Move the Sun vertically
    sun_y = 1.5 + .75 * np.sin(x[frame]-1)
    circle.set_center((1, sun_y))
    return circle,

ani = FuncAnimation(
    fig,
    update,
    frames=len(x),
    init_func=init,
    interval=10,
    blit=False
)

plt.show()