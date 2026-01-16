import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

# A structured-grid Poisson solver using DMDA+KSP.
# e.g. mpirun -np 4 python poisson.py -da_grid_x 100 -da_grid_y 200

comm = PETSc.COMM_WORLD
rank = comm.getRank()
size = comm.getSize()

# change default 9x9 size using -da_grid_x M -da_grid_y N
da = PETSc.DMDA().create(
    dim = 2,
    dof = 1,
    sizes = (9, 9),
    boundary_type = (PETSc.DMDA.BoundaryType.NONE, PETSc.DMDA.BoundaryType.NONE),
    stencil_type = PETSc.DMDA.StencilType.STAR,
    stencil_width = 1,
    setup = False
)
da.setFromOptions()
da.setUp()

# create linear system matrix A
A = da.createMat()
A.setFromOptions()

# create RHS b, approx solution u, exact solution uexact
b = da.createGlobalVec()
u = b.duplicate()
uexact = b.duplicate()

def formExact(da, uexact):
    (mx, my) = da.getSizes()
    (xs, ys), (xm, ym) = da.getCorners()
    hx = 1.0 / (mx - 1)
    hy = 1.0 / (my - 1)
    with da.getVecArray(uexact) as auexact:
        for i in range(xs, xs + xm):
            for j in range(ys, ys + ym):
                x = i * hx
                y = j * hy
                auexact[i, j] = x * x * (1.0 - x * x) * y * y * (y * y - 1.0)

def formRHS(da, b):
    (mx, my) = da.getSizes()
    (xs, ys), (xm, ym) = da.getCorners()
    hx = 1.0 / (mx - 1)
    hy = 1.0 / (my - 1)
    with da.getVecArray(b) as ab:
        for i in range(xs, xs + xm):
            for j in range(ys, ys + ym):
                x = i * hx
                y = j * hy
                if i == 0 or i == mx - 1 or j == 0 or j == my - 1:
                    ab[i, j] = 0.0  # on boundary: 1*u = 0
                else:
                    f = 2.0 * ( (1.0 - 6.0*x*x) * y*y * (1.0 - y*y)
                        + (1.0 - 6.0*y*y) * x*x * (1.0 - x*x) )
                    ab[i, j] = hx * hy * f

def formMat(da, A):
    (mx, my) = da.getSizes()
    (xs, ys), (xm, ym) = da.getCorners()
    hx = 1.0 / (mx - 1)
    hy = 1.0 / (my - 1)
    t0 = time.time()
    for i in range(xs, xs + xm):
        for j in range(ys, ys + ym):
            cols, vals = [], []

            # row of A corresponding to (x_i,y_j)
            row = PETSc.Mat.Stencil()
            row.i, row.j = i, j
            
            col = PETSc.Mat.Stencil()
            col.i, col.j = i, j
            cols.append(col)
            
            if i == 0 or i == mx - 1 or j == 0 or j == my - 1:
                # on boundary: trivial equation
                vals.append(1.0)
            else:
                # interior: build a row
                vals.append(2.0 * (hy / hx + hx / hy))
                if i - 1 > 0:
                    col = PETSc.Mat.Stencil()
                    col.i, col.j = i - 1, j
                    cols.append(col)
                    vals.append(-hy / hx)
                if i + 1 < mx - 1:
                    col = PETSc.Mat.Stencil()
                    col.i, col.j = i + 1, j
                    cols.append(col)
                    vals.append(-hy / hx)
                if j - 1 > 0:
                    col = PETSc.Mat.Stencil()
                    col.i, col.j = i, j - 1
                    cols.append(col)
                    vals.append(-hx / hy)
                if j + 1 < my - 1:
                    col = PETSc.Mat.Stencil()
                    col.i, col.j = i, j + 1
                    cols.append(col)
                    vals.append(-hx / hy)
            
            for col, val in zip(cols, vals):
                # setValuesStencil is not implemented now
                A.setValueStencil(row, col, val, PETSc.InsertMode.INSERT_VALUES)
    A.assemble()

# fill vectors and assemble linear system
formExact(da, uexact)
formRHS(da, b)
formMat(da, A)

# create and solve the linear system
ksp = PETSc.KSP().create()
ksp.setOperators(A, A)
ksp.setFromOptions()
ksp.solve(b, u)

# report on grid and numerical error
u.axpy(-1.0, uexact)  # u = u + (-1.0) * uexact
errnorm = u.norm(PETSc.NormType.INFINITY)
(mx, my) = da.getSizes()
PETSc.Sys.Print(f"on {mx} x {my} grid:  error |u-uexact|_inf = {errnorm}")

for obj in [u, uexact, b, A, ksp, da]:
    obj.destroy()