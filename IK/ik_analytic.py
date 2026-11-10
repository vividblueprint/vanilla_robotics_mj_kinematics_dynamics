import numpy as np
from robot_3R import robot

def inverse_kinematics_analytic(target, elbow_up=True):

    x, y, theta = target

    l1 = robot.body[2].pos[0]
    l2 = robot.body[3].pos[0]
    l3 = robot.params.end_eff_pos_local[0]

    xw = x - l3 * np.cos(theta)
    yw = y - l3 * np.sin(theta)

    c2 = (xw**2 + yw**2 - l1**2 - l2**2) / (2 * l1 * l2) # theta2
    c2 = np.clip(c2, -1.0, 1.0) # Numerical protection

    if elbow_up:
        s2 = -np.sqrt(1.0 - c2**2)
    else:
        s2 = np.sqrt(1.0 - c2**2)

    theta1 = (np.arctan2(yw, xw) - np.arctan2(l2 * s2, l1 + l2 * c2))
    theta2 = np.arctan2(s2, c2)
    theta3 = theta - theta1 - theta2

    return np.array([theta1, theta2, theta3])