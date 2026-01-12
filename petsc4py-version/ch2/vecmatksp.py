import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

# Solve a 4x4 linear system using KSP.
# e.g. mpirun -np 4 python vecmatksp.py

comm = PETSc.COMM_WORLD
rank = comm.getRank()
size = comm.getSize()

b = PETSc.Vec().create()
b.setSizes(4)

A = PETSc.Mat().create()
A.setSizes((4, 4))
A.setFromOptions()
A.setUp()

j = [0, 1, 2, 3]                        # j = column index
ab = [7.0, 1.0, 1.0, 3.0]               # vector entries
b.setValues(j, ab, PETSc.InsertMode.INSERT)
b.assemble()

aA = [[ 1.0,  2.0,  3.0,  0.0],
      [ 2.0,  1.0, -2.0, -3.0],
      [-1.0,  1.0,  1.0,  0.0],
      [ 0.0,  1.0,  1.0, -1.0]]         # matrix entries
for i in range(*A.getOwnershipRange()):
    A.setValues([i], j, aA[i], PETSc.InsertMode.INSERT)
A.assemble()

ksp = PETSc.KSP().create()
ksp.setOperators(A, A)
ksp.setFromOptions()
x = b.duplicate()
ksp.solve(b, x)
x.view(PETSc.Viewer.STDOUT)

for obj in [A, b, x, ksp]:
    obj.destroy()
