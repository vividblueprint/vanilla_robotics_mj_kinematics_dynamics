import sympy as sy
from rotation_sym import rotation

phi, theta, psi = sy.symbols('phi theta psi', real=True)

Rx = rotation(phi, 1)
Ry = rotation(theta, 2)
Rz = rotation(psi, 3)

R = Rx * Ry * Rz
print("Combined Rotation Matrix (R):")
sy.pprint(R)

# latex_r = sy.latex(R)
# sy.pprint(latex_r)









