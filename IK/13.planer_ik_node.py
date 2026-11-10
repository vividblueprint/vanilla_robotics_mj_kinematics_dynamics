import numpy as np
from fk import forward_kinematics
from ik import inverse_kinematics
from scipy.optimize import fsolve
from animate import animate

x_ref = 0.5; y_ref = 0; theta_ref = np.pi/2;

parmas = [x_ref, y_ref, theta_ref]
q0 = np.array([0.5,0.0,0])

q = fsolve(inverse_kinematics, q0, args=parmas)
sol = forward_kinematics(q)

animate(sol, np.array(parmas))

