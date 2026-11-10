import sympy as sp
from utility import vec2skew

l = sp.symbols("l1, l2, l3", real=True)
theta = sp.symbols("theta1, theta2, theta3", real=True)

n = sp.Matrix([0, 0, 1])

# Positions
O01 = sp.Matrix([0, 0, 0])
O12 = sp.Matrix([l[0], 0, 0])
O23 = sp.Matrix([l[1], 0, 0])

E3  = sp.Matrix([l[2], 0, 0, 1])

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

# E0 = H03@E3
# e0 = sp.Matrix(E0[:3])

# sp.pprint(o01)
# sp.pprint(o02)
# sp.pprint(o03)
# ////////////////| forward positions |////////////////
R02 = R01@R12
R03 = R02@R23

o02 = O01 + R01@O12
o03 = o02 + R02@O23
e0  = o03 + R03@sp.Matrix(E3[:3])
# //////////////////////////////////////////////

# sp.pprint(R02)
# sp.pprint(R03)

# sp.pprint(O01)
# sp.pprint(O02)
# sp.pprint(O03)

Jv_E = sp.Matrix.hstack(
    vec2skew(R00@n)@(e0 - O01),
    vec2skew(R01@n)@(e0 - o02),
    vec2skew(R02@n)@(e0 - o03)
)

Jw_E = sp.Matrix.hstack(
    R00@n,
    R01@n,
    R02@n
)

J_E = sp.Matrix.vstack(Jv_E, Jw_E)

Jv_o02 = sp.Matrix.hstack(
    vec2skew(R00@n)@(o02 - O01),
    sp.zeros(3, 1),
    sp.zeros(3, 1)
)

Jw_o02 = sp.Matrix.hstack(
    R00@n,
    sp.zeros(3, 1),
    sp.zeros(3, 1)
)

J_o02 = sp.Matrix.vstack(Jv_o02, Jw_o02)

sp.pprint(sp.simplify(J_E))
sp.pprint(sp.simplify(J_o02))