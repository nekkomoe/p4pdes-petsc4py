import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np
from dataclasses import dataclass, field

# 1D reaction-diffusion problem with DMDA and SNES.
# Option prefix -rct_.
# e.g. mpirun -np 4 python reaction.py -rct_noRinJ True
#      mpirun -np 4 python reaction.py -rct_noRinJ False

@dataclass
class AppCtx:
    rho: float = 10.0
    M: float = field(init=False)
    alpha: float = field(init=False)
    beta: float = field(init=False)
    noRinJ: bool = False

    def __post_init__(self):
        self.M = (self.rho / 12.0)**2
        self.alpha = self.M
        self.beta = 16.0 * self.M

user = AppCtx()

def f_source(x):
    return 0.0

def initialAndExact(da, user):
    u = da.createGlobalVector()
    uexact = da.createGlobalVector()
    
    (mx,) = da.sizes
    (xs, xe), = da.ranges
    h = 1.0 / (mx - 1)
    
    au = da.getVecArray(u)
    auex = da.getVecArray(uexact)
    
    for i in range(xs, xe):
        x = h * i
        au[i] = user.alpha * (1.0 - x) + user.beta * x
        auex[i] = user.M * (x + 1.0)**4
        
    return u, uexact

def formFunction(snes, X, F, user):
    da = snes.getDM()
    localX = da.createLocalVector()
    da.globalToLocal(X, localX)
    
    u = da.getVecArray(localX)
    ff = da.getVecArray(F)
    
    (mx,) = da.sizes
    (xs, xe), = da.ranges
    h = 1.0 / (mx - 1)
    
    for i in range(xs, xe):
        if i == 0:
            ff[i] = u[i] - user.alpha
        elif i == mx - 1:
            ff[i] = u[i] - user.beta
        else:
            if i == 1:
                ff[i] = -u[i+1] + 2.0*u[i] - user.alpha
            elif i == mx - 2:
                ff[i] = -user.beta + 2.0*u[i] - u[i-1]
            else:
                ff[i] = -u[i+1] + 2.0*u[i] - u[i-1]
            
            R = -user.rho * np.sqrt(u[i])
            ff[i] -= h*h * (R + f_source(i * h))

def formJacobian(snes, X, J, P, user):
    da = snes.getDM()
    localX = da.createLocalVector()
    da.globalToLocal(X, localX)
    u = da.getVecArray(localX)
    
    (mx,) = da.sizes
    (xs, xe), = da.ranges
    h = 1.0 / (mx - 1)
    
    for i in range(xs, xe):
        if i == 0 or i == mx - 1:
            P.setValues([i], [i], [1.0])
        else:
            cols = []
            vals = []
            
            # Diagonal
            val_diag = 2.0
            if not user.noRinJ:
                dRdu = -(user.rho / 2.0) / np.sqrt(u[i])
                val_diag -= h*h * dRdu
            
            cols.append(i)
            vals.append(val_diag)
            
            # Left neighbor
            cols.append(i-1)
            val_left = -1.0 if i > 1 else 0.0
            vals.append(val_left)
            
            # Right neighbor
            cols.append(i+1)
            val_right = -1.0 if i < mx - 2 else 0.0
            vals.append(val_right)
            
            P.setValues([i], cols, vals)

    P.assemble()
    if J != P: J.assemble()

# Process options
opts = PETSc.Options("rct_")
user.noRinJ = opts.getBool("noRinJ", user.noRinJ)

# Create DMDA
da = PETSc.DMDA().create(
    dim=1,
    sizes=(9,),
    boundary_type=PETSc.DM.BoundaryType.NONE,
    stencil_width=1,
    stencil_type=PETSc.DMDA.StencilType.STAR,
    setup=False,
)
da.setFromOptions()
da.setUp()

# Create global vectors and set initial/exact values
u, uexact = initialAndExact(da, user)

# Create SNES
snes = PETSc.SNES().create()
snes.setDM(da)
snes.setFunction(formFunction, args=(user,))
snes.setJacobian(formJacobian, args=(user,))
snes.setFromOptions()
snes.setUp()

# Solve
snes.solve(None, u)

# Calculate error
u.axpy(-1.0, uexact) # u <- u + (-1.0) uexact
errnorm = u.norm(PETSc.NormType.INFINITY)

(mx,) = da.sizes
PETSc.Sys.Print(f"on {mx} point grid:  |u-u_exact|_inf = {errnorm:g}")

for obj in [u, uexact, snes, da]:
    obj.destroy()