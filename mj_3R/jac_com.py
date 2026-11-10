import numpy as np
import utility as rm
from fk import forward_kinematics

def jac_com(q, params):
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

    g1 = fk.H01@np.array([0.5 * params[0], 0, 0, 1])
    g2 = fk.H02@np.array([0.5 * params[1], 0, 0, 1])
    g3 = fk.H03@np.array([0.5 * params[2], 0, 0, 1])

    g1 = g1[:3]
    g2 = g2[:3]
    g3 = g3[:3]

    Jv_G1 = np.column_stack([rm.vec2skew(R00@k)@(g1 - o00), np.zeros((3, 1)), np.zeros((3, 1))])
    Jv_G2 = np.column_stack([rm.vec2skew(R00@k)@(g2 - o00), rm.vec2skew(R01@k)@(g2 - o01), np.zeros((3, 1))])
    Jv_G3 = np.column_stack([rm.vec2skew(R00@k)@(g3 - o00), rm.vec2skew(R01@k)@(g3 - o01), rm.vec2skew(R02@k)@(g3 - o02)])
    
    Jw_G1 = np.column_stack([R00@k, np.zeros((3,1)), np.zeros((3,1))])
    Jw_G2 = np.column_stack([R00@k, R01@k, np.zeros((3,1))])
    Jw_G3 = np.column_stack([R00@k, R01@k, R02@k])

    J_G1 = np.vstack([Jv_G1, Jw_G1])
    J_G2 = np.vstack([Jv_G2, Jw_G2])
    J_G3 = np.vstack([Jv_G3, Jw_G3])

    return J_G1, J_G2, J_G3