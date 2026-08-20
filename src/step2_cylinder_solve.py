"""
Step 2 — 3D Continuous Model: Solve the EIT continuous problem on a cylinder.

PDE:
    div(gamma * grad(u)) = 0    in Omega (cylinder)

Neumann boundary condition:
    gamma * du/dn = g           on dOmega (cylinder surface)

With:
    gamma = 1  (constant conductivity)
    g(x,y,z) = x + y + 3z      (as proposed by the advisor)

Variational form (what FEniCSx solves):
    integral_Omega  gamma * grad(u) . grad(v) dx
    =
    integral_dOmega  g * v ds

Mesh: P1 geometry (tetrahedra).
Function space: P2 (quadratic Lagrange).
"""

import gmsh
import dolfinx
import dolfinx.fem.petsc
import ufl
import basix
import pyvista
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.io import gmsh as gmshio
import os

os.makedirs("outputs", exist_ok=True)

# ── 1. Generate cylinder mesh ─────────────────────────────────────────────────
print("Generating mesh...")

gmsh.finalize()
gmsh.initialize()

cylinder = gmsh.model.occ.addCylinder(0, 0, -1, 0, 0, 2, 1)
gmsh.model.occ.synchronize()

volumes  = gmsh.model.getEntities(dim=3)
surfaces = gmsh.model.getEntities(dim=2)
gmsh.model.addPhysicalGroup(3, [v[1] for v in volumes],  tag=1)
gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces], tag=2)

gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.05)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.02)
gmsh.option.setNumber("Mesh.Algorithm",      6)
gmsh.option.setNumber("Mesh.Algorithm3D",    1)
gmsh.option.setNumber("Mesh.Optimize",       1)
gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

gmsh.model.mesh.generate(3)
gmsh.model.mesh.optimize("Netgen")
gmsh.model.mesh.optimize("Relocate3D")

mesh_comm = MPI.COMM_WORLD
mesh_data = gmshio.model_to_mesh(gmsh.model, mesh_comm, 0, gdim=3)
mesh = mesh_data.mesh
gmsh.finalize()

print(f"  Cells:    {mesh.topology.index_map(3).size_global}")
print(f"  Vertices: {mesh.topology.index_map(0).size_global}")

# ── 2. Function space P2 ──────────────────────────────────────────────────────
Ve = basix.ufl.element('Lagrange', 'tetrahedron', degree=2, shape=())
V  = dolfinx.fem.functionspace(mesh, Ve)
print(f"  DOFs:     {V.dofmap.index_map.size_global}")

# ── 3. Boundary measure ds ────────────────────────────────────────────────────
ds = ufl.Measure("ds", domain=mesh)

# ── 4. Define g(x,y,z) = x + y + 3z ─────────────────────────────────────────
x = ufl.SpatialCoordinate(mesh)
g = x[0] + x[1] + 3*x[2]

# ── 5. Variational formulation ────────────────────────────────────────────────
eps = 1e-10
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + eps * ufl.inner(u, v) * ufl.dx
L = g * v * ds

# ── 6. Assemble and solve ─────────────────────────────────────────────────────
print("Assembling system...")
a_form = dolfinx.fem.form(a)
L_form = dolfinx.fem.form(L)

A = dolfinx.fem.petsc.assemble_matrix(a_form)
A.assemble()

b = dolfinx.fem.petsc.assemble_vector(L_form)
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

print("Solving...")
solver = PETSc.KSP().create(mesh_comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.CG)
solver.getPC().setType(PETSc.PC.Type.HYPRE)
solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
solver.setFromOptions()

u_h = dolfinx.fem.Function(V)
solver.solve(b, u_h.x.petsc_vec)
u_h.x.scatter_forward()

# Normalize: remove constant offset
u_h.x.array[:] -= u_h.x.array.mean()
u_h.x.scatter_forward()

print(f"  Solver converged in {solver.getIterationNumber()} iterations")
print(f"  u min: {u_h.x.array.min():.4f}  max: {u_h.x.array.max():.4f}")

# ── 7. Plot solution ──────────────────────────────────────────────────────────
print("Plotting...")
topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
grid["u"] = u_h.x.array.real

clim = [u_h.x.array.min(), u_h.x.array.max()]
surface = grid.extract_surface(algorithm="dataset_surface")

sargs = dict(
    title="u",
    title_font_size=16,
    label_font_size=13,
    color="white",
    position_x=0.05,
    position_y=0.05,
    width=0.38,
    height=0.04,
)

plotter = pyvista.Plotter(off_screen=True, shape=(1, 2), window_size=(1400, 700))

# ── Left: full cylinder ───────────────────────────────────────────────────────
plotter.subplot(0, 0)
plotter.add_text("Solution u on cylinder (P2)", font_size=11, color="white")
plotter.add_mesh(
    surface,
    scalars="u",
    cmap="turbo",
    clim=clim,
    show_edges=False,
    lighting=True,
    smooth_shading=True,
    scalar_bar_args=sargs,
)
plotter.add_axes(color="white")
plotter.view_isometric()

# ── Right: 3 cross-sections shown at their actual z heights (isometric) ──────
plotter.subplot(0, 1)
plotter.add_text("Cross-sections: z = -0.5, 0, +0.5", font_size=11, color="white")

for z_val in [-0.5, 0.0, 0.5]:
    # Clip: keep only the thin slice at z = z_val (thickness ~0.01)
    slab = grid.clip(normal="z",  origin=(0, 0, z_val + 0.01))
    slab = slab.clip(normal="-z", origin=(0, 0, z_val - 0.01))
    plotter.add_mesh(
        slab,
        scalars="u",
        cmap="turbo",
        clim=clim,
        show_edges=False,
        lighting=True,
        smooth_shading=True,
        show_scalar_bar=False,
    )

plotter.add_axes(color="white")
plotter.view_isometric()

plotter.set_background("#1e1e2e")
plotter.screenshot("outputs/step2_cylinder_solution.png")
plotter.close()

print("Saved: outputs/step2_cylinder_solution.png")