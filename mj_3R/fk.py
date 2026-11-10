import numpy as np
from types import SimpleNamespace

def forward_kinematics(q, params):
    c1 = np.cos(q[0])
    s1 = np.sin(q[0])
    c2 = np.cos(q[1])
    s2 = np.sin(q[1])
    c3 = np.cos(q[2])
    s3 = np.sin(q[2])

    # compute H01, H02, H03
    H01 = np.array([
        [c1, -s1, 0, 0],
        [s1, c1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    H12 = np.array([
        [c2, -s2, 0, params[0]],
        [s2, c2, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    H23 = np.array([
        [c3, -s3, 0, params[1]],
        [s3, c3, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    H02 = H01 @ H12
    H03 = H01 @ H12 @ H23

    O = np.array([0, 0])
    P = H01@H12@np.array([0, 0, 0, 1])
    Q = H03@np.array([0, 0, 0, 1])

    E3 = np.array([params[2], 0, 0, 1])
    E = H03@E3

    return SimpleNamespace(
        o=O, p=P[:2], q=Q[:2], e=E[:3],
        H01=H01, H12 = H12, H23 = H23,
        H02=H02, H03=H03
        )