import numpy as np
from scipy.optimize import fsolve
from fk import forward_kinematics
from animate import animate

from ik_analytic import inverse_kinematics_analytic
from ik import inverse_kinematics


x_ref = 0.5
y_ref = 0
theta_ref = np.pi/2

params = [x_ref, y_ref, theta_ref]
q0 = np.array([0.5, 0.5, 0.5])

# q=fsolve(inverse_kinematics, q0, args=params)
q =inverse_kinematics_analytic(params, elbow_up=True)

sol = forward_kinematics(q)

animate(sol, np.array(params))


