import sympy as sp

theta  = sp.symbols("theta1 theta2 theta3", real=True)
omega  = sp.symbols("omega1 omega2 omega3", real=True)
alpha  = sp.symbols("alpha1 alpha2 alpha3", real=True)
m = sp.symbols("m1 m2 m3", real=True)
I = sp.symbols("Iz1 Iz2 Iz3", real=True)
l = sp.symbols("l1 l2 l3", real=True)
g = sp.symbols("g", real=True)

q = sp.Matrix([theta[0], theta[1], theta[2]])
qdot = sp.Matrix([omega[0], omega[1], omega[2]])
qddot = sp.Matrix([alpha[0], alpha[1], alpha[2]])

c1 = sp.cos(theta[0])
s1 = sp.sin(theta[0])
c2 = sp.cos(theta[1])
s2 = sp.sin(theta[1])
c3 = sp.cos(theta[2])
s3 = sp.sin(theta[2])

O01 = sp.Matrix([0, 0, 0])
O12 = sp.Matrix([l[0], 0, 0])
O23 = sp.Matrix([l[1], 0, 0])

R01 = sp.Matrix([
    [c1, -s1, 0],
    [s1, c1,  0],
    [0, 0, 1]
])

R12 = sp.Matrix([
    [c2, -s2, 0],
    [s2, c2,  0],
    [0, 0, 1]
])

R23 = sp.Matrix([
    [c3, -s3, 0],
    [s3, c3,  0],
    [0, 0, 1]
])

H01 = sp.Matrix.vstack(sp.Matrix.hstack(R01, O01), sp.Matrix([[0, 0, 0, 1]]))
H12 = sp.Matrix.vstack(sp.Matrix.hstack(R12, O12), sp.Matrix([[0, 0, 0, 1]]))
H23 = sp.Matrix.vstack(sp.Matrix.hstack(R23, O23), sp.Matrix([[0, 0, 0, 1]]))

H02 = H01@H12
H03 = H02@H23

G1 = sp.Matrix(H01*sp.Matrix([0.5*l[0], 0, 0, 1]))
G2 = sp.Matrix(H02*sp.Matrix([0.5*l[1], 0, 0, 1]))
G3 = sp.Matrix(H03*sp.Matrix([0.5*l[2], 0, 0, 1]))

V_G1 = G1.jacobian(q)*qdot
V_G2 = G2.jacobian(q)*qdot
V_G3 = G3.jacobian(q)*qdot

T = 0.5*(m[0]*V_G1.dot(V_G1) + \
         m[1]*V_G2.dot(V_G2) + \
         m[2]*V_G3.dot(V_G3) + \
         I[0]*omega[0]*omega[0] + \
         I[1]*(omega[0]+omega[1])*(omega[0]+omega[1]) + \
         I[2]*(omega[0]+omega[1]+omega[2])*(omega[0]+omega[1]+omega[2])
)

V = m[0]*g*G1[1] +  m[1]*g*G2[1] + m[2]*g*G3[1]

L = T - V

dLdqdot = []
ddt_dLdqdot = []
dLdq = []
n = len(qddot)
EOM = []

for i in range(n):
    dLdqdot.append(sp.diff(L, qdot[i]))
    temp = 0
    for j in range(n):
        temp += sp.diff(dLdqdot[i], q[j])*qdot[j] + sp.diff(dLdqdot[i], qdot[j])*qddot[j]
    ddt_dLdqdot.append(temp)
    dLdq.append(sp.diff(L, q[i]))
    EOM.append(ddt_dLdqdot[i] - dLdq[i])

EOM = sp.Matrix(EOM)

print('KE = ', sp.simplify(T))
print('PE = ', sp.simplify(V))
print('TE = KE+PE \n')

M = EOM.jacobian(qddot)
N = EOM.subs({qddot[0]:0, qddot[1]:0, qddot[2]:0}) # N = C + G 
G = N.subs({qdot[0]:0, qdot[1]:0, qdot[2]:0})
C = N - G

print('M11 = ', sp.simplify(M[0,0]))
print('M12 = ', sp.simplify(M[0,1]))
print('M13 = ', sp.simplify(M[0,2]))
print('M21 = ', sp.simplify(M[1,0]))
print('M22 = ', sp.simplify(M[1,1]))
print('M23 = ', sp.simplify(M[1,2]))
print('M31 = ', sp.simplify(M[2,0]))
print('M32 = ', sp.simplify(M[2,1]))
print('M33 = ', sp.simplify(M[2,2]))

print()

print('C1 = ', sp.simplify(C[0]))
print('C2 = ', sp.simplify(C[1]))
print('C3 = ', sp.simplify(C[2]))

print()

print('G1 = ', sp.simplify(G[0]))
print('G2 = ', sp.simplify(G[1]))
print('G3 = ', sp.simplify(G[2]))