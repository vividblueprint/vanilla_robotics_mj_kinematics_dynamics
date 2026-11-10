import sympy as sp
from utility import vec2skew

l = sp.symbols("l1, l2, l3", real=True)
m = sp.symbols("m1, m2, m3", real=True)
g = sp.symbols("g", real=True)
theta = sp.symbols("theta1, theta2, theta3", real=True)

n = sp.Matrix([0, 0, 1])

# Positions
O01 = sp.Matrix([0, 0, 0])
O12 = sp.Matrix([l[0], 0, 0])
O23 = sp.Matrix([l[1], 0, 0])

# com Positions
G1 = sp.Matrix([l[0] / 2, 0, 0, 1])
G2 = sp.Matrix([l[1] / 2, 0, 0, 1])
G3 = sp.Matrix([l[2] / 2, 0, 0, 1])

# Gravitational forces
F_g1 = sp.Matrix([0, -m[0] * g, 0])
F_g2 = sp.Matrix([0, -m[1] * g, 0])
F_g3 = sp.Matrix([0, -m[2] * g, 0])

# Rotation Matrices
R00 = sp.eye(3)
R01 = sp.Matrix([
    [sp.cos(theta[0]), -sp.sin(theta[0]), 0],
    [sp.sin(theta[0]), sp.cos(theta[0]), 0],
    [0, 0, 1]
])

R12 = sp.Matrix([
    [sp.cos(theta[1]), -sp.sin(theta[1]), 0],
    [sp.sin(theta[1]), sp.cos(theta[1]), 0],
    [0, 0, 1]
])

R23 = sp.Matrix([
    [sp.cos(theta[2]), -sp.sin(theta[2]), 0],
    [sp.sin(theta[2]), sp.cos(theta[2]), 0],
    [0, 0, 1]
])

# ////////////////| Homogeneous Matrices|////////////////
# H01 = sp.Matrix.vstack(
#     sp.Matrix.hstack(R01, O01),
#     sp.Matrix([[0, 0, 0, 1]])
# )

# H12 = sp.Matrix.vstack(
#     sp.Matrix.hstack(R12, O12),
#     sp.Matrix([[0, 0, 0, 1]])
# )

# H23 = sp.Matrix.vstack(
#     sp.Matrix.hstack(R23, O23),
#     sp.Matrix([[0, 0, 0, 1]])
# )

# # # Transformation matrices
# H02 = H01 @ H12
# H03 = H02 @ H23

# R02 = H02[:3, :3]
# R03 = H03[:3, :3]

# o01 = H01[:3, 3]
# o02 = H02[:3, 3]
# o03 = H03[:3, 3]

# G01  = H01@sp.Matrix(G1)
# G02  = H02@sp.Matrix(G2)
# G03  = H03@sp.Matrix(G3)

# g01 = sp.Matrix(G01[:3])
# g02 = sp.Matrix(G02[:3])
# g03 = sp.Matrix(G03[:3])

# sp.pprint(o01)
# sp.pprint(o02)
# sp.pprint(o03)
# ////////////////| forward positions |////////////////
R02 = R01@R12
R03 = R02@R23

o02 = O01 + R01@O12
o03 = o02 + R02@O23

g01 = O01 + R01@sp.Matrix(G1[:3])
g02 = o02 + R02@sp.Matrix(G2[:3])
g03 = o03 + R03@sp.Matrix(G3[:3])
# //////////////////////////////////////////////

# sp.pprint(R02)
# sp.pprint(R03)

# sp.pprint(sp.simplify(g01))
# sp.pprint(sp.simplify(g02))
# sp.pprint(sp.simplify(g03))

# sp.pprint(sp.simplify(O01))
# sp.pprint(sp.simplify(o02))
# sp.pprint(sp.simplify(o03))

Jv_g1 = sp.Matrix.hstack(
    vec2skew(R00@n)@(g01 - O01),
    sp.zeros(3, 1),
    sp.zeros(3, 1)
)

Jv_g2 = sp.Matrix.hstack(
    vec2skew(R00@n)@(g02 - O01),
    vec2skew(R01@n)@(g02 - o02),
    sp.zeros(3, 1)
)

Jv_g3 = sp.Matrix.hstack(
    vec2skew(R00@n)@(g03 - O01),
    vec2skew(R01@n)@(g03 - o02),
    vec2skew(R02@n)@(g03 - o03)
)

# Torques
tau1 = Jv_g1.T @ F_g1
tau2 = Jv_g2.T @ F_g2
tau3 = Jv_g3.T @ F_g3

tau = tau1 + tau2 + tau3

sp.pprint(sp.simplify(tau1))
sp.pprint(sp.simplify(tau2))
sp.pprint(sp.simplify(tau3))
sp.pprint(sp.simplify(tau))