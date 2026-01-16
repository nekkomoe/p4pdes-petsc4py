import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np

# Newton's method for a two-variable system.
# No analytical Jacobian.  Run with -snes_fd or -snes_mf.
# e.g. mpirun -np 1 python expcircle.py -snes_fd -mat_view
#      mpirun -np 1 python expcircle.py -snes_mf -mat_view

comm = PETSc.COMM_SELF

def formFunction(snes, x, F):
    b = 2.0
    ax = x.getArray(readonly=True)
    with F as aF:
        # MPI is not needed since this is a single process example
        aF[0] = (1.0 / b) * np.exp(b * ax[0]) - ax[1]
        aF[1] = ax[0] * ax[0] + ax[1] * ax[1] - 1.0

x = PETSc.Vec().create(comm=comm)
x.setSizes(2)
x.setFromOptions()
x.set(1.0)  # initial iterate
r = x.duplicate()

snes = PETSc.SNES().create(comm=comm)
snes.setFunction(formFunction, r)
snes.setFromOptions()
snes.solve(None, x)

x.view()

for obj in [x, r, snes]:
    obj.destroy()
