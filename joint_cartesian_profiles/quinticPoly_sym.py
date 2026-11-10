import sympy as sy

t0, tf  = sy.symbols('t0 tf', real=True) # defining the variables
q0, qf  = sy.symbols('q0 qf', real=True) # defining the variables

A = sy.Matrix([ [1, t0, t0**2, t0**3, t0**4, t0**5],
                [1, tf, tf**2, tf**3, tf**4, tf**5],
                [0,  1, 2*t0,  3*t0**2, 4*t0**3, 5*t0**4 ],
                [0,  1, 2*tf,  3*tf**2, 4*tf**3, 5*tf**4],
                [0,  0, 2,  6*t0, 12*t0**2, 20*t0**3 ],
                [0,  0, 2,  6*tf, 12*tf**2, 20*tf**3],
                 ])
b = sy.Matrix([ [q0],
                [qf],
                [0],
                [0],
                [0],
                [0]
                 ])


Ainv = A.inv()
x = Ainv*b
#print(x)

for i in range(len(x)):
    print(f'a{i} = {x[i]}')
