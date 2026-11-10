import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def rotation(phi, theta, psi):

    cx, sx = np.cos(phi), np.sin(phi)
    cy, sy = np.cos(theta), np.sin(theta)
    cz, sz = np.cos(psi), np.sin(psi)

    Rx = np.array([
        [1,  0,   0],
        [0, cx, -sx],
        [0, sx,  cx],
    ])

    Ry = np.array([
        [ cy, 0, sy],
        [  0, 1,  0],
        [-sy, 0, cy],
    ])

    Rz = np.array([
        [cz, -sz, 0],
        [sz,  cz, 0],
        [ 0,   0, 1],
    ])

    return Rx @ Ry @ Rz

def create_box(lx=0.5, ly=0.25, lz=0.1):
    """Return the 8 vertices and 12 triangular faces."""

    vertices = np.array([
        [-lx, -ly, -lz],  # 0
        [ lx, -ly, -lz],  # 1
        [ lx,  ly, -lz],  # 2
        [-lx,  ly, -lz],  # 3
        [-lx, -ly,  lz],  # 4
        [ lx, -ly,  lz],  # 5
        [ lx,  ly,  lz],  # 6
        [-lx,  ly,  lz],  # 7
    ])

    # faces = np.array([         # Triangular
    #     [0, 2, 1], [0, 3, 2],  # Bottom
    #     [4, 5, 6], [6, 7, 4],  # Top
    #     [0, 1, 5], [0, 5, 4],  # Side
    #     [1, 2, 6], [1, 6, 5],  # Side
    #     [2, 3, 7], [6, 2, 7],  # Side
    #     [0, 4, 7], [7, 3, 0],  # Side
    # ])

    faces = np.array([   # Rectengular
        [0, 1, 2, 3, 0], # Bottom
        [4, 5, 6, 7, 4], # Top
        [0, 1, 5, 4, 0], # Side
        [0, 3, 7, 4, 0], # Side
        [3, 2, 6, 7, 3], # Side
        [1, 2, 6, 5, 1], # Side
    ])

    return vertices, faces

def draw_frame(ax, R, length=0.8, linewidth=2):
    """Draw rotated X, Y, Z axes."""

    origin = np.zeros(3)
    colors = ["red", "green", "blue"]
    labels = ["X", "Y", "Z"]

    for i, (color, label) in enumerate(zip(colors, labels)): # Original frame
        direction = np.eye(3)[:, i] * length * 2

        ax.quiver(
            *origin,
            *direction,
            color=color,
            arrow_length_ratio=0.12,
            linewidth=linewidth//2,
        )

        # Label each axis near its arrow tip.
        ax.text(
            *direction,
            label,
            color=color,
            fontsize=10,
        )

    for i, (color, label) in enumerate(zip(colors, labels)): # Rotated frame
        direction = R[:, i] * length

        ax.quiver(
            *origin,
            *direction,
            color=color,
            arrow_length_ratio=0.12,
            linewidth=linewidth,
        )

        # Label each axis near its arrow tip.
        ax.text(
            *direction,
            label,
            color=color,
            fontsize=10,
        )

def animate(ax, phi, theta, psi):
    """Draw a cuboid and its rotated coordinate frame."""

    vertices, faces = create_box()

    # Rotation matrix
    R = rotation(phi, theta, psi)

    # Rotate every vertex: v_rotated = R @ v
    rotated_vertices = vertices @ R.T

    # Draw the rotated cuboid
    box = Poly3DCollection(
        rotated_vertices[faces],
        facecolors="skyblue",
        edgecolors="black",
        alpha=0.65,
    )

    ax.add_collection3d(box)

    # Draw the rotated coordinate frame
    draw_frame(ax, R)

    # Convert radians to degrees for the title
    angles = np.rad2deg([phi, theta, psi])

    ax.set_title(f"(φ,θ,ψ) = ({angles[0]:.0f},{angles[1]:.0f},{angles[2]:.0f})°")

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(False)
    ax.axis(False)

    ax.set_box_aspect((1, 1, 1))

def main():
    # Four orientations to compare
    angles = [
        (0, 0, 0),
        (np.pi / 2, 0, 0),
        (np.pi / 2, np.pi / 2, 0),
        (np.pi / 2, np.pi / 2, np.pi / 2),
    ]

    fig = plt.figure(figsize=(11, 9))

    for i, (phi, theta, psi) in enumerate(angles, start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        animate(ax, phi, theta, psi)

    fig.suptitle(
        "Euler Angles: 3D Rotation",
        fontsize=16,
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()