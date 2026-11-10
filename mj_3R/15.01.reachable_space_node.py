import numpy as np
from fk import forward_kinematics
import matplotlib.pyplot as plt
from animate import animate

l = np.array([1, 1, 0.25])
q_range = np.array([
    [0,         np.pi/2],
    [-np.pi/2,  np.pi/2],
    [-np.pi/2,  np.pi/2]
])

n = 40

end_eff_x = []
end_eff_y = []

phi =   np.linspace(q_range[0, 0], q_range[0, 1], n)
theta = np.linspace(q_range[1, 0], q_range[1, 1], n)
psi =   np.linspace(q_range[2, 0], q_range[2, 1], n)

for i in range(0, len(phi)):
    for j in range(0, len(theta)):
        for k in range(0, len(psi)):
            sol = forward_kinematics(np.array([phi[i], theta[j], psi[k]]), l)
            end_eff_x.append(sol.e[0])
            end_eff_y.append(sol.e[1])

plt.figure(figsize=(8, 6))
plt.scatter(end_eff_x, end_eff_y, marker="o", color="blue", label="reachable space")

plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Reachable space')
plt.legend()
plt.grid(True)
plt.show()

# print(sol.o)
# print(sol.p)
# print(sol.q)
# print(sol.e)
