from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.integrate import odeint

class parameters:
    def __init__(self):
        self.m1 = 1
        self.m2 = 1
        self.m3 = 0.2;
        self.Iz1 = 0.1
        self.Iz2 = 0.1
        self.Iz3 = 0.02;
        self.l1 = 1
        self.l2 = 1
        self.l3 = 0.25
        self.g = 9.81
        self.pause = 0.001
        self.fps =100

def cos(angle):
    return np.cos(angle)

def sin(angle):
    return np.sin(angle);

def energy(t,z,parms):

    m1 = parms.m1
    m2 = parms.m2
    m3 = parms.m3
    Iz1 = parms.Iz1
    Iz2 = parms.Iz2
    Iz3 = parms.Iz3
    l1 = parms.l1
    l2 = parms.l2
    l3 = parms.l3
    g= parms.g

    KE_all = []
    PE_all = []
    TE_all = []

    [m,n] = np.shape(z)

    for i in range(0,m):
        theta1 = z[i,0];
        omega1 = z[i,1];
        theta2 = z[i,2];
        omega2 = z[i,3];
        theta3 = z[i,4];
        omega3 = z[i,5];

        KE =  0.5*Iz1*omega1**2 + 0.5*Iz2*(omega1 + omega2)**2 + 0.5*Iz3*(omega1 + omega2 + omega3)**2 + 0.125*l1**2*m1*omega1**2 + 0.5*m2*(l1**2*omega1**2*sin(theta2)**2 + (l1*omega1*cos(theta2) + 0.5*l2*(omega1 + omega2))**2) + 0.5*m3*((l1*omega1*sin(theta2 + theta3) + l2*omega1*sin(theta3) + l2*omega2*sin(theta3))**2 + (1.0*l1*omega1*cos(theta2 + theta3) + 1.0*l2*omega1*cos(theta3) + 1.0*l2*omega2*cos(theta3) + 0.5*l3*omega1 + 0.5*l3*omega2 + 0.5*l3*omega3)**2)
        PE =  g*(0.5*l1*m1*sin(theta1) + m2*(l1*sin(theta1) + 0.5*l2*sin(theta1 + theta2)) + m3*(l1*sin(theta1) + l2*sin(theta1 + theta2) + 0.5*l3*sin(theta1 + theta2 + theta3)))
        TE = KE + PE

        KE_all = np.append(KE_all,KE)
        PE_all = np.append(PE_all,PE)
        TE_all = np.append(TE_all,TE)

    return KE_all, PE_all, TE_all

def rhs(z,t,m1,m2,m3,Iz1,Iz2,Iz3,l1,l2,l3,g):

    theta1 = z[0];
    omega1 = z[1];
    theta2 = z[2];
    omega2 = z[3];
    theta3 = z[4];
    omega3 = z[5];

    M11 =  1.0*Iz1 + 1.0*Iz2 + 1.0*Iz3 + 0.25*l1**2*m1 + 1.0*l1**2*m2 + 1.0*l1**2*m3 + 1.0*l1*l2*m2*cos(theta2) + 2.0*l1*l2*m3*cos(theta2) + 1.0*l1*l3*m3*cos(theta2 + theta3) + 0.25*l2**2*m2 + 1.0*l2**2*m3 + 1.0*l2*l3*m3*cos(theta3) + 0.25*l3**2*m3
    M12 =  1.0*Iz2 + 1.0*Iz3 + 0.5*l1*l2*m2*cos(theta2) + 1.0*l1*l2*m3*cos(theta2) + 0.5*l1*l3*m3*cos(theta2 + theta3) + 0.25*l2**2*m2 + 1.0*l2**2*m3 + 1.0*l2*l3*m3*cos(theta3) + 0.25*l3**2*m3
    M13 =  Iz3 + 0.5*l1*l3*m3*cos(theta2 + theta3) + 0.5*l2*l3*m3*cos(theta3) + 0.25*l3**2*m3
    M21 =  1.0*Iz2 + 1.0*Iz3 + 0.5*l1*l2*m2*cos(theta2) + 1.0*l1*l2*m3*cos(theta2) + 0.5*l1*l3*m3*cos(theta2 + theta3) + 0.25*l2**2*m2 + 1.0*l2**2*m3 + 1.0*l2*l3*m3*cos(theta3) + 0.25*l3**2*m3
    M22 =  1.0*Iz2 + 1.0*Iz3 + 0.25*l2**2*m2 + 1.0*l2**2*m3 + 1.0*l2*l3*m3*cos(theta3) + 0.25*l3**2*m3
    M23 =  Iz3 + 0.5*l2*l3*m3*cos(theta3) + 0.25*l3**2*m3
    M31 =  Iz3 + 0.5*l3*m3*(1.0*l1*cos(theta2 + theta3) + 1.0*l2*cos(theta3) + 0.5*l3)
    M32 =  Iz3 + 0.5*l3*m3*(l2*cos(theta3) + 0.5*l3)
    M33 =  Iz3 + 0.25*l3**2*m3

    C1 =  -1.0*l1*l2*m2*omega1*omega2*sin(theta2) - 0.5*l1*l2*m2*omega2**2*sin(theta2) - 2.0*l1*l2*m3*omega1*omega2*sin(theta2) - 1.0*l1*l2*m3*omega2**2*sin(theta2) - 1.0*l1*l3*m3*omega1*omega2*sin(theta2 + theta3) - 1.0*l1*l3*m3*omega1*omega3*sin(theta2 + theta3) - 0.5*l1*l3*m3*omega2**2*sin(theta2 + theta3) - 1.0*l1*l3*m3*omega2*omega3*sin(theta2 + theta3) - 0.5*l1*l3*m3*omega3**2*sin(theta2 + theta3) - 1.0*l2*l3*m3*omega1*omega3*sin(theta3) - 1.0*l2*l3*m3*omega2*omega3*sin(theta3) - 0.5*l2*l3*m3*omega3**2*sin(theta3)
    C2 =  0.5*l1*l2*m2*omega1**2*sin(theta2) + 1.0*l1*l2*m3*omega1**2*sin(theta2) + 0.5*l1*l3*m3*omega1**2*sin(theta2 + theta3) - 1.0*l2*l3*m3*omega1*omega3*sin(theta3) - 1.0*l2*l3*m3*omega2*omega3*sin(theta3) - 0.5*l2*l3*m3*omega3**2*sin(theta3)
    C3 =  0.5*l3*m3*(l1*omega1**2*sin(theta2)*cos(theta3) + (l1*omega1**2*cos(theta2) + l2*(omega1 + omega2)**2)*sin(theta3))

    G1 =  g*(0.5*l1*m1*cos(theta1) + 1.0*l1*m2*cos(theta1) + 1.0*l1*m3*cos(theta1) + 0.5*l2*m2*cos(theta1 + theta2) + 1.0*l2*m3*cos(theta1 + theta2) + 0.5*l3*m3*cos(theta1 + theta2 + theta3))
    G2 =  g*(0.5*l2*m2*cos(theta1 + theta2) + 1.0*l2*m3*cos(theta1 + theta2) + 0.5*l3*m3*cos(theta1 + theta2 + theta3))
    G3 =  0.5*g*l3*m3*cos(theta1 + theta2 + theta3)

    A = np.array([[M11, M12, M13], [M21,M22, M23], [M31,M32, M33]]);
    b = -np.array([C1+G1,C2+G2,C3+G3])
    invA = np.linalg.inv(A)
    thetaddot = invA.dot(b) #-inv(M)*(C+G)
    alpha1 = thetaddot[0]
    alpha2 = thetaddot[1]
    alpha3 = thetaddot[2]

    dzdt = np.array([omega1, alpha1, omega2, alpha2,omega3, alpha3]);
    return dzdt

def animate(t,z,parms):
    t_interp = np.arange(t[0],t[len(t)-1],1/parms.fps)
    z_interp = np.zeros((len(t_interp), np.shape(z)[1]))

    f = interpolate.interp1d(t, z, axis=0)
    z_interp = f(t_interp)

    l1 = parms.l1
    l2 = parms.l2
    l3 = parms.l3
    ll = 1.1*(l1+l2+l3);

    for i in range(len(t_interp)):
        theta1 = z_interp[i,0];
        theta2 = z_interp[i,2];
        theta3 = z_interp[i,4];

        O = np.array([0, 0])
        P = np.array([l1*cos(theta1), l1*sin(theta1)])
        Q = P + np.array([l2*cos(theta1+theta2),l2*sin(theta1+theta2)])
        E = Q + np.array([l3*cos(theta1+theta2+theta3),l3*sin(theta1+theta2+theta3)])
        G1 = np.array([0.5*l1*cos(theta1), 0.5*l1*sin(theta1)])
        G2 = P + np.array([0.5*l2*cos(theta1+theta2),0.5*l2*sin(theta1+theta2)])
        G3 = Q + np.array([0.5*l3*cos(theta1+theta2+theta3),0.5*l3*sin(theta1+theta2+theta3)])

        pend1, = plt.plot([O[0], P[0]],[O[1], P[1]],linewidth=5, color='red')
        pend2, = plt.plot([P[0], Q[0]],[P[1], Q[1]],linewidth=5, color='blue')
        pend3, = plt.plot([Q[0], E[0]],[Q[1], E[1]],linewidth=5, color='green')
        com1, = plt.plot(G1[0],G1[1],color='black',marker='o',markersize=8)
        com2, = plt.plot(G2[0],G2[1],color='black',marker='o',markersize=8)
        com3, = plt.plot(G3[0],G3[1],color='black',marker='o',markersize=8)

        plt.xlim(-ll,ll)
        plt.ylim(-ll,ll)
        plt.gca().set_aspect('equal')


        plt.pause(parms.pause)
        if (i < len(t_interp)-1):
            pend1.remove()
            pend2.remove()
            pend3.remove()
            com1.remove()
            com2.remove()
            com3.remove()

    #plt.show()
    plt.show(block=False)
    plt.pause(2)
    plt.close()

parms = parameters()

t = np.linspace(0, 10, 1001)
z0 = np.array([0, 0, 0, 0, 0, 0])
all_parms = (parms.m1,parms.m2,parms.m3,parms.Iz1,parms.Iz2,parms.Iz3,parms.l1,parms.l2, parms.l3,parms.g)
z = odeint(rhs, z0, t, args=all_parms)

KE,PE,TE = energy(t,z,parms)

animate(t,z,parms)

plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(t,z[:,0],color='red',label='theta1');
plt.plot(t,z[:,2],color='blue',label='theta2');
plt.plot(t,z[:,4],color='green',label='theta3');
plt.ylabel("angle")
plt.legend(loc="upper left")
plt.subplot(2, 1, 2)
plt.plot(t,z[:,1],color='red',label='theta1');
plt.plot(t,z[:,3],color='blue',label='theta2');
plt.plot(t,z[:,5],color='green',label='theta3');
plt.xlabel("t")
plt.ylabel("angular rate")
plt.legend(loc="lower left")
#plt.show()
plt.show(block=False)
plt.pause(2)
plt.close()

plt.figure(2)
plt.plot(t,KE,color='red',label='KE');
plt.plot(t,PE,color='blue',label='PE');
plt.plot(t,TE,color='black',label='TE');
plt.legend(loc="upper left")
plt.show(block=False)
plt.pause(2)
plt.close()
