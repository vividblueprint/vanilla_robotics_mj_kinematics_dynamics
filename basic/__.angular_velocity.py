import sympy as sy

phi, theta, psi = sy.symbols("phi theta psi", real = True)
phidot, thetadot, psidot = sy.symbols('phidot thetadot psidot', real=True)

i = sy.Matrix([1, 0, 0])
j = sy.Matrix([0, 1, 0])
k = sy.Matrix([0, 0, 1])

R_x = sy.Matrix(
    [
        [1, 0, 0],
        [0, sy.cos(phi), -sy.sin(phi)],
        [0, sy.sin(phi), sy.cos(phi)]
    ]
)

R_y = sy.Matrix(
    [
        [sy.cos(theta), 0, sy.sin(theta)],
        [0, 1, 0],
        [-sy.sin(theta), 0, sy.cos(theta)]
    ]
)

R_z = sy.Matrix(
    [
        [sy.cos(psi), -sy.sin(psi), 0],
        [sy.sin(psi), sy.cos(psi), 0],
        [0, 0, 1]
    ]
)

# Angular velocity in the world frame
omega = (phidot * i) + R_x * (thetadot * j) + R_x * R_y * (psidot * k) # 1-2-3

A = omega.jacobian([phidot, thetadot, psidot])

det_A = sy.simplify(sy.det(A))
inv_A = sy.simplify(A.inv())

sy.pprint(A)
sy.pprint(det_A)
sy.pprint(inv_A)

# Angular velocity in the body frame
Omega = (psidot * k) + R_z.T * (thetadot * j) + R_z.T * R_y.T * (phidot * i) # 1-2-3

B = Omega.jacobian([phidot, thetadot, psidot])

det_B = sy.simplify(sy.det(B))
inv_B = sy.simplify(B.inv())

sy.pprint(B)
sy.pprint(det_B)
sy.pprint(inv_B)
