import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

# Assemble a Mat sparsely.
# e.g. mpirun -np 4 python sparsemat.py

comm = PETSc.COMM_WORLD
rank = comm.getRank()
size = comm.getSize()

A = PETSc.Mat().create()
A.setSizes((4, 4))
A.setFromOptions()
A.setUp()

i1 = [0, 1, 2]
j1 = [0, 1, 2]
aA1 = [1.0, 2.0, 3.0,
       2.0, 1.0, -2.0,
       -1.0, 1.0, 1.0]
A.setValues(i1, j1, aA1, PETSc.InsertMode.INSERT)

i2 = [3]
j2 = [1, 2, 3]
aA2 = [1.0, 1.0, -1.0]
A.setValues(i2, j2, aA2, PETSc.InsertMode.INSERT)

i3 = [1]
j3 = [3]
aA3 = [-3.0]
A.setValues(i3, j3, aA3, PETSc.InsertMode.INSERT)

A.assemble()

for obj in [A]:
    obj.destroy()