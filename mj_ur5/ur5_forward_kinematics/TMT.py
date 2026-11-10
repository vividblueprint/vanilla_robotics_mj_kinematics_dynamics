import numpy as np
from scipy.linalg import block_diag

import utility as ram
from jac_torques import jac_torques


def TMT(q):
    # ==========================================
    # Compute geometric Jacobians
    # ==========================================
    robot, *jacobians = jac_torques(q)

    mass_matrices = []

    # ==========================================
    # Compute link spatial inertia matrices
    # ==========================================
    for i in range(1, len(robot.body) + 1):
        body = robot.body[i]

        # Translational inertia
        mass = body.mass * np.eye(3)

        # Rotational inertia in body frame
        R_inertia = ram.quat2rotation(body.iquat)

        body.I_body = (
            R_inertia
            @ np.diag(body.inertia)
            @ R_inertia.T
        )

        # Transform inertia into world frame
        R_world = body.H_global[:3, :3]

        inertia = (
            R_world
            @ body.I_body
            @ R_world.T
        )

        # 6x6 spatial inertia matrix at COM
        M_local = block_diag(mass, inertia)

        mass_matrices.append(M_local)

    # ==========================================
    # Assemble block-diagonal inertia matrix
    # ==========================================
    M_ = block_diag(*mass_matrices)

    # ==========================================
    # Stack all COM Jacobians
    # ==========================================
    J = np.vstack(jacobians)

    # ==========================================
    # Joint-space mass matrix
    # ==========================================
    M_TMT = J.T @ M_ @ J

    return np.asarray(M_TMT)