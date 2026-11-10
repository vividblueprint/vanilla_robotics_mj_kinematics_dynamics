from fk import forward_kinematics

def inverse_kinematics(q, params):
    fk = forward_kinematics(q)
    return fk.e[0] - params[0], fk.e[1] - params[1], fk.theta - params[2] # params = X_ref