import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

# Load a matrix  A  and right-hand-side  b  from binary files (PETSc format).
# Then solve the system  A x = b  using KSPSolve().
# Example.  First save a system from tri.py:
#   mpirun -np 4 python tri.py -ksp_view_mat binary:A.dat -ksp_view_rhs binary:b.dat
# then load it and solve:
#   mpirun -np 4 python loadsolve.py -fA A.dat -fb b.dat
# To time the solution read the third printed number:
#   mpirun -np 4 python loadsolve.py -fA A.dat -fb b.dat -log_view | grep KSPSolve
# (This is a simpler code than src/ksp/ksp/examples/tutorials/ex10.c.)
# 
# small system example w/o RHS:
#   mpirun -np 4 python tri.py -ksp_view_mat binary:A.dat
#   mpirun -np 4 python loadsolve.py -fA A.dat -ksp_view_mat -ksp_view_rhs -log_view | grep KSPSolve
# 
# small system example with RHS:
#   mpirun -np 4 python tri.py -ksp_view_mat binary:A.dat -ksp_view_rhs binary:b.dat
#   mpirun -np 4 python loadsolve.py -fA A.dat -fb b.dat -log_view | grep KSPSolve
# 
# large tridiagonal system (m=10^7) example:
#   mpirun -np 4 python tri.py -tri_m 10000000 -ksp_view_mat binary:A.dat -ksp_view_rhs binary:b.dat
#   mpirun -np 4 python loadsolve.py -fA A.dat -fb b.dat -log_view | grep KSPSolve

comm = PETSc.COMM_WORLD
rank = comm.getRank()
size = comm.getSize()

opts = PETSc.Options()
nameA = opts.getString('fA', '')         # input file containing matrix A
nameb = opts.getString('fb', '')         # input file containing vector b
verbose = opts.getBool('verbose', False) # say what is going on
flg = bool(nameb)

if len(nameA) == 0:
    raise PETSc.Error("no input matrix provided ... ending  (usage: loadsolve -fA A.dat)")
    # SETERRQ(PETSC_COMM_WORLD,1,"no input matrix provided ... ending  (usage: loadsolve -fA A.dat)\n");

if verbose:
    PETSc.Sys.Print(f"reading matrix from {nameA} ...")
A = PETSc.Mat().create()
A.setFromOptions()
fileA = PETSc.Viewer().createBinary(nameA, 'r')
A.load(fileA)
fileA.destroy()
m, n = A.getSize()
if verbose:
    PETSc.Sys.Print(f"matrix has size {m} x {n} ...")


if m != n:
    raise PETSc.Error("only works for square matrices")
    # SETERRQ(PETSC_COMM_WORLD,2,"only works for square matrices\n");

b = PETSc.Vec().create()
b.setFromOptions()
if flg:
    fileb = PETSc.Viewer().createBinary(nameb, 'r')
    if verbose:
        PETSc.Sys.Print(f"reading vector from {nameb} ...")
    b.load(fileb)
    fileb.destroy()
    mb = b.getSize()
    if mb != m:
        raise PETSc.Error("size of matrix and vector do not match")
        # SETERRQ(PETSC_COMM_WORLD,3,"size of matrix and vector do not match\n");
else:
    if verbose:
        PETSc.Sys.Print(f"right-hand-side vector b not provided ... using zero vector of length {m}")
    b.setSizes(m)
    b.set(0.0)

ksp = PETSc.KSP().create()
ksp.setOperators(A, A)
ksp.setFromOptions()

x = b.duplicate()
x.set(0.0)
ksp.solve(b, x)

for obj in [A, b, x, ksp]:
    obj.destroy()