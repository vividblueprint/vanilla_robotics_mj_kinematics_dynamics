import numpy as np
import utility as rm
from fk import forward_kinematics

def jac_end_effector(q, params):
    fk = forward_kinematics(q, params)

    R00 = np.eye(3)
    R01 = fk.H01[:3, :3]
    R02 = fk.H02[:3, :3]
    R03 = fk.H03[:3, :3]

    e0 = fk.e
    
    k = np.array([0, 0, 1])
    
    o00 = fk.H01[:3, 3]
    o01 = fk.H02[:3, 3]
    o02 = fk.H03[:3, 3]

    Jv_E = np.column_stack([rm.vec2skew(R00@k)@(e0 - o00), rm.vec2skew(R01@k)@(e0 - o01), rm.vec2skew(R02@k)@(e0 - o02)])
    Jw_E = np.column_stack([R00@k, R01@k, R02@k])

    J_E = np.vstack([Jv_E, Jw_E])

    return J_E