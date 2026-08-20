import gmsh
import dolfinx
import dolfinx.fem.petsc
import ufl
import basix
import pyvista
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.io import gmsh as gmshio
import os

os.makedirs("outputs", exist_ok=True)

BG     = "#1e1e2e"

# Mesh
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
mesh      = mesh_data.mesh
gmsh.finalize()

n_cells = mesh.topology.index_map(3).size_global
n_verts = mesh.topology.index_map(0).size_global
print(f"  Cells: {n_cells}  Vertices: {n_verts}")

# Function space P2
Ve     = basix.ufl.element('Lagrange', 'tetrahedron', degree=2, shape=())
V      = dolfinx.fem.functionspace(mesh, Ve)
n_dofs = V.dofmap.index_map.size_global
print(f"  DOFs: {n_dofs}")

# Time parameter
T       = 1.0
dt      = 0.1
n_steps = int(T / dt)

# Forms (backward Euler)
ds       = ufl.Measure("ds", domain=mesh)
x        = ufl.SpatialCoordinate(mesh)
g        = x[0] + x[1] + 3*x[2]
u_n      = dolfinx.fem.Function(V)
dt_const = dolfinx.fem.Constant(mesh, PETSc.ScalarType(dt))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = (u / dt_const) * v * ufl.dx + ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = (u_n / dt_const) * v * ufl.dx + g * v * ds

# Assemble matrix (once)
a_form = dolfinx.fem.form(a)
L_form = dolfinx.fem.form(L)

A = dolfinx.fem.petsc.assemble_matrix(a_form)
A.assemble()

# Solver (CG + HYPRE)
solver = PETSc.KSP().create(mesh_comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.CG)
solver.getPC().setType(PETSc.PC.Type.HYPRE)
solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
solver.setFromOptions()

# RHS vector created ONCE and reused every step
b   = A.createVecRight()
u_h = dolfinx.fem.Function(V)

# Time loop
snapshot_times = [0.1, 0.3, 0.5, 1.0]
snapshots = {}

print(f"Running {n_steps} time steps...")
t = 0.0
for step in range(n_steps):
    t += dt

    with b.localForm() as loc_b:
        loc_b.set(0)
    dolfinx.fem.petsc.assemble_vector(b, L_form)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    solver.solve(b, u_h.x.petsc_vec)
    u_h.x.scatter_forward()
    u_n.x.array[:] = u_h.x.array.copy()

    print(f"  t={t:.1f}  u=[{u_h.x.array.min():.3f}, {u_h.x.array.max():.3f}]")

    for st in snapshot_times:
        if abs(t - st) < 1e-10:
            snapshots[st] = u_h.x.array.copy()

# PyVista grid
topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

all_vals = np.concatenate(list(snapshots.values()))
clim = [float(all_vals.min()), float(all_vals.max())]

sargs = dict(
    title="u", title_font_size=18, label_font_size=14,
    color="white", position_x=0.03, position_y=0.03,
    width=0.38, height=0.04,
)

# Render 4 snapshots
tmp_paths = []
for i, st in enumerate(snapshot_times):
    grid["u"] = snapshots[st]
    surface   = grid.extract_surface(algorithm="dataset_surface")
    path      = f"outputs/_tmp_{i}.png"

    p = pyvista.Plotter(off_screen=True, window_size=(950, 950))
    p.add_mesh(surface, scalars="u", cmap="turbo", clim=clim,
                show_edges=False, lighting=True, smooth_shading=True,
                show_scalar_bar=(i == 0), scalar_bar_args=sargs)
    p.set_background(BG)
    p.view_isometric()
    p.camera.zoom(0.95)
    p.screenshot(path)
    p.close()
    tmp_paths.append(path)

# Compose
imgs = [np.array(Image.open(p)) for p in tmp_paths]

fig = plt.figure(figsize=(22, 7), facecolor=BG)
gs  = gridspec.GridSpec(
    1, 4, figure=fig,
    hspace=0.01, wspace=0.03,
    left=0.02, right=0.98,
    top=0.93, bottom=0.01,
)

for i, (img, st) in enumerate(zip(imgs, snapshot_times)):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(img)
    ax.axis("off")
    ax.set_facecolor(BG)
    ax.set_title(f"t = {st}", color="white", fontsize=15, pad=8)

plt.savefig("outputs/step3_heat_equation.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()

for p in tmp_paths:
    os.remove(p)

A.destroy()
b.destroy()
solver.destroy()

print("Saved: outputs/step3_heat_equation.png")