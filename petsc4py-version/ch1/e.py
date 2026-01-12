import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI

comm = PETSc.COMM_WORLD
rank = comm.getRank()
size = comm.getSize()

# compute  1/n!  where n = (rank of process) + 1
localval = 1.0
for i in range(2, rank + 1):
    localval /= i

# sum the contributions over all processes
globalsum = comm.tompi4py().allreduce(localval, op=MPI.SUM)

# output estimate of e and report on work from each process
PETSc.Sys.Print(f"e is about {globalsum:17.15f}")
PETSc.Sys.syncPrint(f"rank {rank} did {rank - 1 if rank > 0 else 0} flops", comm=comm)
PETSc.Sys.syncFlush(comm=comm)
