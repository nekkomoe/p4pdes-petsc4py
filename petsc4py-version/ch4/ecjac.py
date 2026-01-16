import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np

# Newton's method for a two-variable system.
# Implements analytical Jacobian and a struct to hold a parameter.
# e.g. mpirun -np 1 python ecjac.py

user = {'b': 2.0}
comm = PETSc.COMM_SELF

def formFunction(snes, X, F, user):
    b = user['b']
    x = X.array_r
    with F as f:
        f[0] = (1.0 / b) * np.exp(b * x[0]) - x[1]
        f[1] = x[0]**2 + x[1]**2 - 1.0

def formJacobian(snes, X, J, P, user):
    b = user['b']
    x = X.array_r

    P.setValues([0, 1], [0, 1], np.array([
        [np.exp(b * x[0]), -1.0],
        [2.0 * x[0], 2.0 * x[1]],
    ]))
    
    P.assemble()
    if J != P:
        J.assemble()

# Create vectors
x = PETSc.Vec().create(comm=comm)
x.setSizes(2)
x.setFromOptions()
r = x.duplicate()

# Create Jacobian matrix
J = PETSc.Mat().create(comm=comm)
J.setSizes(2)
J.setFromOptions()
J.setUp()

# Create SNES solver
snes = PETSc.SNES().create(comm=comm)
snes.setFunction(formFunction, r, args=(user,))
snes.setJacobian(formJacobian, J, J, args=(user,))
snes.setFromOptions()

# Initial iterate
x.set(1.0)

# Solve
snes.solve(None, x)
x.view()

for obj in [x, r, J, snes]:
    obj.destroy()