import numpy as np

import utility as ram
from forward_kinematics import forward_kinematics


def jac_torques(q):
    robot, _ = forward_kinematics(q)
    n = len(robot.body)

    origins = []
    com_positions = []
    rotations = [np.eye(3)]
    joint_axes = []

    for i in range(1, n + 1):
        body = robot.body[i]
        H = body.H_global

        origins.append(H[:3, 3])

        com_positions.append((H @ np.append(body.ipos, 1))[:3])

        rotations.append(H[:3, :3])
        joint_axes.append(body.joint_axis)

    jacobians = []

    for i in range(n):
        Jv = np.zeros((3, n))
        Jw = np.zeros((3, n))

        for j in range(i + 1):
            # Joint axis in world frame
            axis = rotations[j] @ joint_axes[j]

            # Linear Jacobian (original equation)
            Jv[:, j] = (ram.vec2skew(axis) @ (com_positions[i] - origins[j]))
            Jw[:, j] = axis

        J = np.vstack([Jv, Jw])
        jacobians.append(J)

    return robot, *jacobians