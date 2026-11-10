from robot_3R import robot
import numpy as np
import utility as rm
from types import SimpleNamespace


def forward_kinematics(q):
    H_base = np.identity(4)

    transforms = {}

    for i, body in robot.body.items():
        R_local = rm.rotation(q[i-1], np.argmax(np.abs(body.joint_axis)))
        # R_local = rm.quat2rotation(body.quat) @ R
        p_local = body.pos.reshape(-1, 1) # [1x3] -> [3x1]

        H_body = np.block([
            [R_local, p_local], 
            [np.zeros((1, 3)), np.ones((1, 1))]
        ])

        H_base = H_base @ H_body
        transforms[i] = H_base

    theta = np.arccos(np.clip(transforms[3][0, 0], -1.0, 1.0))
    w = robot.params.w
    h = robot.params.h
    l3 = robot.params.end_eff_pos_local[0]
    e = (H_base @ np.append(robot.params.end_eff_pos_local, 1))[:3]

    E_left1 = np.matmul(H_base, np.array([l3, -w, 0, 1]))
    E_right1 = np.matmul(H_base, np.array([l3, w, 0, 1]))
    E_left2 = np.matmul(H_base, np.array([l3+h, -w, 0, 1]))
    E_right2 = np.matmul(H_base, np.array([l3+h, w, 0, 1]))

    e_left1 = E_left1[0:2]
    e_right1 = E_right1[0:2]
    e_left2 = E_left2[0:2]
    e_right2 = E_right2[0:2]

    fk = SimpleNamespace(
        o = transforms[1][:3, 3],
        p = transforms[2][:3, 3],
        q = transforms[3][:3, 3],
        e=np.array(e),
        e_left1=np.array(e_left1),
        e_right1 = np.array(e_right1),
        e_left2=np.array(e_left2),
        e_right2 = np.array(e_right2),
        theta = theta
    )

    return fk
