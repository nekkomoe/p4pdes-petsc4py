import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import math

# Solve a tridiagonal system of arbitrary size.
# Option prefix = tri_.
# e.g. mpirun -np 4 python tri.py -tri_m 16 -a_mat_view

comm = PETSc.COMM_WORLD
rank = comm.getRank()
size = comm.getSize()

opts = PETSc.Options("tri_") # options for tri
m = opts.getInt("m", 4)      # dimension of linear system

x = PETSc.Vec().create()
x.setSizes(m)
x.setFromOptions()
b = x.duplicate()
xexact = x.duplicate()

A = PETSc.Mat().create()
A.setSizes((m, m))
A.setPreallocationNNZ(3) # 3 for diag-elements, and 3 for offdiag-elements
A.setOptionsPrefix("a_")
A.setFromOptions()
A.setUp()

Istart, Iend = A.getOwnershipRange()

for i in range(Istart, Iend):
    if i == 0:
        cols = [0, 1]
        vals = [3.0, -1.0]
    elif i == m - 1:
        cols = [m - 2, m - 1]
        vals = [-1.0, 3.0]
    else:
        cols = [i - 1, i, i + 1]
        vals = [-1.0, 3.0, -1.0]
    A.setValues([i], cols, vals, addv=PETSc.InsertMode.INSERT)
    xval = math.exp(math.cos(float(i)))
    xexact.setValues([i], [xval], addv=PETSc.InsertMode.INSERT)

A.assemble()
xexact.assemble()

A.mult(xexact, b)

ksp = PETSc.KSP().create()
ksp.setOperators(A, A)
ksp.setFromOptions()
ksp.solve(b, x)

x.axpy(-1.0, xexact) # x = x + (-1.0) * xexact
errnorm = x.norm(PETSc.NormType.NORM_2)

PETSc.Sys.Print(f"error for m = {m:d} system is |x-xexact|_2 = {errnorm:.1e}")

for obj in [x, b, xexact, A, ksp]:
    obj.destroy()