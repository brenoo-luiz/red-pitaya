"""
Step 2 — 3D Continuous Model: Solve the EIT continuous problem on a cylinder.
"""

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
PANEL  = "#2a2a4a"
BORDER = "#4a4a6a"

# ── 1. Mesh ───────────────────────────────────────────────────────────────────
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

# ── 2. Function space P2 ──────────────────────────────────────────────────────
Ve    = basix.ufl.element('Lagrange', 'tetrahedron', degree=2, shape=())
V     = dolfinx.fem.functionspace(mesh, Ve)
n_dofs = V.dofmap.index_map.size_global
print(f"  DOFs: {n_dofs}")

# ── 3. Forms ──────────────────────────────────────────────────────────────────
ds  = ufl.Measure("ds", domain=mesh)
x   = ufl.SpatialCoordinate(mesh)
g   = x[0] + x[1] + 3*x[2]
eps = 1e-10
u   = ufl.TrialFunction(V)
v   = ufl.TestFunction(V)
a   = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + eps * ufl.inner(u, v) * ufl.dx
L   = g * v * ds

# ── 4. Solve ──────────────────────────────────────────────────────────────────
print("Solving...")
a_form = dolfinx.fem.form(a)
L_form = dolfinx.fem.form(L)

A = dolfinx.fem.petsc.assemble_matrix(a_form)
A.assemble()
b = dolfinx.fem.petsc.assemble_vector(L_form)
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

solver = PETSc.KSP().create(mesh_comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.CG)
solver.getPC().setType(PETSc.PC.Type.HYPRE)
solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
solver.setFromOptions()

u_h = dolfinx.fem.Function(V)
solver.solve(b, u_h.x.petsc_vec)
u_h.x.scatter_forward()
u_h.x.array[:] -= u_h.x.array.mean()
u_h.x.scatter_forward()

u_min = float(u_h.x.array.min())
u_max = float(u_h.x.array.max())
print(f"  Converged in {solver.getIterationNumber()} iters | u: [{u_min:.3f}, {u_max:.3f}]")

# ── 5. PyVista grid ───────────────────────────────────────────────────────────
topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
grid    = pyvista.UnstructuredGrid(topology, cell_types, geometry)
grid["u"] = u_h.x.array.real
clim    = [u_min, u_max]
surface = grid.extract_surface(algorithm="dataset_surface")

sargs = dict(
    title="u", title_font_size=20, label_font_size=16,
    color="white", position_x=0.03, position_y=0.03,
    width=0.40, height=0.05,
)

# ── 6. Render cylinder + barra de cores PyVista ───────────────────────────────
print("Rendering cylinder...")
p1 = pyvista.Plotter(off_screen=True, window_size=(900, 900))
p1.add_mesh(surface, scalars="u", cmap="turbo", clim=clim,
            show_edges=False, lighting=True, smooth_shading=True,
            show_scalar_bar=True, scalar_bar_args=sargs)
p1.set_background(BG)
p1.view_isometric()
p1.screenshot("outputs/_tmp_cyl.png")
p1.close()

# ── 7. Render cross-sections (sem barra — já tem no cilindro) ─────────────────
print("Rendering cross-sections...")
p2 = pyvista.Plotter(off_screen=True, window_size=(900, 900))
for z_val in [-0.5, 0.0, 0.5]:
    slab = grid.clip(normal="z",  origin=(0, 0, z_val + 0.01))
    slab = slab.clip(normal="-z", origin=(0, 0, z_val - 0.01))
    p2.add_mesh(slab, scalars="u", cmap="turbo", clim=clim,
                show_edges=False, lighting=True, smooth_shading=True,
                show_scalar_bar=False)
p2.set_background(BG)
p2.view_isometric()
p2.screenshot("outputs/_tmp_sec.png")
p2.close()

# ── 8. Compose: formula panel + duas imagens ──────────────────────────────────
print("Composing...")
img_cyl = np.array(Image.open("outputs/_tmp_cyl.png"))
img_sec = np.array(Image.open("outputs/_tmp_sec.png"))

fig = plt.figure(figsize=(20, 14), facecolor=BG)
gs  = gridspec.GridSpec(
    2, 2,
    figure=fig,
    height_ratios=[0.22, 1],
    hspace=0.04,
    wspace=0.03,
    left=0.02, right=0.98,
    top=0.98, bottom=0.01,
)

# ── Formula panel ─────────────────────────────────────────────────────────────
ax_f = fig.add_subplot(gs[0, :])
ax_f.set_facecolor(PANEL)
ax_f.set_xlim(0, 1)
ax_f.set_ylim(0, 1)
ax_f.axis("off")
for spine in ax_f.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor(BORDER)
    spine.set_linewidth(1.5)

formulas = [
    (0.04,  r"$\nabla \cdot (\gamma\, \nabla u) = 0$",
             r"PDE in $\Omega$"),
    (0.27,  r"$\gamma\, \dfrac{\partial u}{\partial n} = g$",
             r"Neumann BC on $\partial\Omega$"),
    (0.53,  r"$\gamma = 1\,,\quad g\,(x,y,z) = x + y + 3z$",
             "Parameters"),
    (0.77,  r"$\int_\Omega \nabla u\cdot\nabla v\,dx = \int_{\partial\Omega} g\,v\,ds$",
             "Variational form"),
]

for xpos, formula, label in formulas:
    ax_f.text(xpos, 0.65, formula, ha="left", va="center",
              fontsize=16, color="#cce0ff",
              transform=ax_f.transAxes, fontfamily="serif")
    ax_f.text(xpos, 0.18, label, ha="left", va="center",
              fontsize=13, color="#8888aa",
              transform=ax_f.transAxes)

for xd in [0.24, 0.50, 0.74]:
    ax_f.axvline(xd, ymin=0.08, ymax=0.92, color=BORDER, linewidth=1.0)

# ── Imagens ───────────────────────────────────────────────────────────────────
ax_cyl = fig.add_subplot(gs[1, 0])
ax_cyl.imshow(img_cyl)
ax_cyl.axis("off")
ax_cyl.set_facecolor(BG)

ax_sec = fig.add_subplot(gs[1, 1])
ax_sec.imshow(img_sec)
ax_sec.axis("off")
ax_sec.set_facecolor(BG)

plt.savefig("outputs/step2_cylinder_solution.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()

os.remove("outputs/_tmp_cyl.png")
os.remove("outputs/_tmp_sec.png")

print("Saved: outputs/step2_cylinder_solution.png")