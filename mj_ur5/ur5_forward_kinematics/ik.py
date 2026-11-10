import numpy as np
import utility as rm
from forward_kinematics import forward_kinematics

def inverse_kinematics(q, X_ref):
    quat_ref = rm.euler2quat(np.array(X_ref[3:])) # phi_ref, theta_ref, psi_ref
    R_ref = rm.euler2rotation(np.array(X_ref[3:]))

    robot, fk = forward_kinematics(q)

    x, y, z = fk.end_eff_pos
    phi, theta, psi = rm.rotation2euler(fk.end_eff_rot)
    quat = rm.rotation2quat(fk.end_eff_rot)

    RRt =  fk.end_eff_rot @ R_ref.T # RR^T = eye(3)

    dquat = np.zeros(4)
    for i in range(4):
        dquat[i] = quat[i] - quat_ref[i]

    # return x - X_ref[0], y - X_ref[1], z - X_ref[2], phi - X_ref[3], theta - X_ref[4], psi - X_ref[5]
    # return x - X_ref[0], y - X_ref[1], z - X_ref[2], dquat[1], dquat[2], dquat[3]
    return x - X_ref[0], y - X_ref[1], z - X_ref[2], RRt[0,0]-1, RRt[1,1]-1, RRt[2,2]-1


# q = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])
# X_ref = np.array([0.5, 0.2, 0.5, 3.14, 0, 0])

# print(inverse_kinematics(q, X_ref))