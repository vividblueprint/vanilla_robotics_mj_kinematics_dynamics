import numpy as np
from fk import forward_kinematics
from animate import animate

l = np.array([1, 1, 0.25])
q = np.array([np.pi/2, -np.pi/2, np.pi/2])

sol = forward_kinematics(q, l)

# print(sol.o)
# print(sol.p)
# print(sol.q)
# print(sol.e)

animate(sol)