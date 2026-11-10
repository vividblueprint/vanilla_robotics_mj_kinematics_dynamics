import numpy as np
from fk import forward_kinematics
import utility as rm
from types import SimpleNamespace

def differential_kinematics(q, u, params):
    fk = forward_kinematics(q, params)

    H01 = fk.H01
    H12 = fk.H12
    H23 = fk.H23

    v00 = w00 = np.array([0, 0, 0])

    k = np.array([0, 0, 1])

    R01 = H01[:3,:3]
    R12 = H12[:3,:3]
    R23 = H23[:3,:3]

    o01 = H01[:3,3]
    o12 = H12[:3,3]
    o23 = H23[:3,3]

    R02 = R01 @ R12
    R03 = R02 @ R23

    w11 = R01.T @ w00 + u[0]*k
    w22 = R02.T @ w11 + u[1]*k
    w33 = R03.T @ w22 + u[2]*k

    w1 = R01 @ w11
    w2 = R02 @ w22
    w3 = R03 @ w33

    v11 = R01.T @ (v00 + rm.vec2skew(w00) @ o01)
    v22 = R12.T @ (v11 + rm.vec2skew(w11) @ o12)
    v33 = R23.T @ (v22 + rm.vec2skew(w22) @ o23)

    v1 = R01 @ v11
    v2 = R02 @ v22
    v3 = R03 @ v33

    pc1 = np.array([0.5 * params[0], 0, 0])
    pc2 = np.array([0.5 * params[1], 0, 0])
    pc3 = np.array([0.5 * params[2], 0, 0])
    
    vc1 = R01 @ (v11 + rm.vec2skew(w11) @ pc1)
    vc2 = R02 @ (v22 + rm.vec2skew(w22) @ pc2)
    vc3 = R03 @ (v33 + rm.vec2skew(w33) @ pc3)

    e3 = np.array([params[2],0,0])

    ve0 = R03@(v33 + rm.vec2skew(w33)@e3)
    we0 = R03@w33

    dfk = SimpleNamespace(
        ve=ve0, we=we0,
        vC1=vc1,vC2=vc2,vC3=vc3,
        v1=v1,v2=v2,v3=v3,
        w1=w1,w2=w2,w3=w3
    )

    return dfk