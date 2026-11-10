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

i = sp.Matrix([1, 0, 0])
j = sp.Matrix([0, 1, 0])
k = sp.Matrix([0, 0, 1])

c1 = sp.cos(theta[0])
s1 = sp.sin(theta[0])
c2 = sp.cos(theta[1])
s2 = sp.sin(theta[1])
c3 = sp.cos(theta[2])
s3 = sp.sin(theta[2])

O01 = sp.Matrix([0, 0, 0])
O12 = sp.Matrix([l[0], 0, 0])
O23 = sp.Matrix([l[1], 0, 0])
O34 = sp.Matrix([l[2], 0, 0])

grav = sp.Matrix([0, -g, 0])

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

R34 = sp.Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

H01 = sp.Matrix.vstack(sp.Matrix.hstack(R01, O01), sp.Matrix([[0, 0, 0, 1]]))
H12 = sp.Matrix.vstack(sp.Matrix.hstack(R12, O12), sp.Matrix([[0, 0, 0, 1]]))
H23 = sp.Matrix.vstack(sp.Matrix.hstack(R23, O23), sp.Matrix([[0, 0, 0, 1]]))
H34 = sp.Matrix.vstack(sp.Matrix.hstack(R34, O34), sp.Matrix([[0, 0, 0, 1]]))

H02 = H01@H12
H03 = H02@H23
H04 = H03@H34

R02 = H02[:3, :3]
R03 = H03[:3, :3]
R04 = H04[:3, :3]

R10 = R01.T
R21 = R12.T
R32 = R23.T

#Velocity
v00 = w00 = sp.Matrix([0, 0, 0])

w11 = R10*w00 + qdot[0]*k
w22 = R21*w11 + qdot[1]*k
w33 = R32*w22 + qdot[2]*k

v11 = R10*(v00 + w00.cross(O01))
v22 = R21*(v11 + w11.cross(O12))
v33 = R32*(v22 + w22.cross(O23))

w01 = R01*w11;
w02 = R02*w22;
w03 = R03*w33;




#Energy
C1_1 = sp.Matrix([0.5*l[0],0,0])
C2_2 = sp.Matrix([0.5*l[1],0,0])
C3_3 = sp.Matrix([0.5*l[2],0,0])


C1_0 = H01*C1_1.col_join(sp.Matrix([1]))
C2_0 = H02*C2_2.col_join(sp.Matrix([1]))
C3_0 = H03*C3_3.col_join(sp.Matrix([1]))

C1_0 =  C1_0[:-1,:]
C2_0 =  C2_0[:-1,:]
C3_0 =  C3_0[:-1,:]

vC1_0 = v11 + w11.cross(C1_1)
vC2_0 = v22 + w22.cross(C2_2)
vC3_0 = v33 + w33.cross(C3_3)

Iner1 = sp.diag(0, 0, I[0])
Iner2 = sp.diag(0, 0, I[1])
Iner3 = sp.diag(0, 0, I[2])

KE = 0.5*m[0]*vC1_0.dot(vC1_0)+ \
     0.5*m[1]*vC2_0.dot(vC2_0)+ \
     0.5*m[2]*vC3_0.dot(vC3_0)+ \
     0.5*w01.dot(Iner1*w01) + \
     0.5*w02.dot(Iner2*w02) + \
     0.5*w03.dot(Iner3*w03)

PE = m[0]*g*C1_0[1]+m[1]*g*C2_0[1]+m[2]*g*C3_0[1]



#acceleration stuff
vdot00 = wdot00 = sp.Matrix([0, 0, 0])

wdot11 = R10*wdot00 + (R10*w00).cross(qdot[0]*k) + qddot[0]*k
wdot22 = R21*wdot11 + (R21*w11).cross(qdot[1]*k) + qddot[1]*k
wdot33 = R32*wdot22 + (R32*w22).cross(qdot[2]*k) + qddot[2]*k

vdot11 = R10*(vdot00 + wdot00.cross(O01) + w00.cross(w00.cross(O01)))
vdot22 = R21*(vdot11 + wdot11.cross(O12) + w11.cross(w11.cross(O12)))
vdot33 = R32*(vdot22 + wdot22.cross(O23) + w22.cross(w22.cross(O23)))

vdotC11 = vdot11 + wdot11.cross(C1_1) + w11.cross(w11.cross(C1_1))
vdotC22 = vdot22 + wdot22.cross(C2_2) + w22.cross(w22.cross(C2_2))
vdotC33 = vdot33 + wdot33.cross(C3_3) + w33.cross(w33.cross(C3_3))

# BACKWARD RECURSION
F1 = m[0]*vdotC11
F2 = m[1]*vdotC22
F3 = m[2]*vdotC33

N1 = Iner1*wdot11 + w11.cross(Iner1*w11)
N2 = Iner2*wdot22 + w22.cross(Iner2*w22)
N3 = Iner3*wdot33 + w33.cross(Iner3*w33)

f4 = sp.Matrix([0, 0, 0])
n4 = sp.Matrix([0, 0, 0])

#n3 is the same as tau3
f3 = R34*f4 + F3 - R32*R21*R10*m[2]*grav
f2 = R23*f3 + F2 - R21*R10*m[1]*grav
f1 = R12*f2 + F1 - R10*m[0]*grav

n3 = N3+R34*n4+C3_3.cross(f3)+(O34-C3_3).cross(R34*f4)
n2 = N2+R23*n3+C2_2.cross(f2)+(O23-C2_2).cross(R23*f3)
n1 = N1+R12*n2+C1_1.cross(f1)+(O12-C1_1).cross(R12*f2)



# torques
tau = sp.Matrix([n1[2],n2[2],n3[2]])

print('KE = ', sp.simplify(KE))
print('PE = ', sp.simplify(PE))
print('TE = KE + PE')
print(' ')

M = tau.jacobian(qddot)                             # M(theta)*thetaddot + C(theta,thetadot) + G(theta,g) = 0
N = tau.subs({qddot[0]:0, qddot[1]:0, qddot[2]:0})  # N = C + G 
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