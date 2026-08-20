import gmsh
import dolfinx
import pyvista
import numpy as np
from mpi4py import MPI
from dolfinx.io import gmsh as gmshio
import os

os.makedirs("outputs", exist_ok=True)

# Generate cylinder mesh with Gmsh
gmsh.finalize()
gmsh.initialize()

cylinder = gmsh.model.occ.addCylinder(0, 0, -1,  # base center
                                        0, 0,  2,  # axis: height = 2
                                        1)          # radius = 1

gmsh.model.occ.synchronize()

volumes  = gmsh.model.getEntities(dim=3)
surfaces = gmsh.model.getEntities(dim=2)
gmsh.model.addPhysicalGroup(3, [v[1] for v in volumes],  tag=1)
gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces], tag=2)

# Mesh size
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.05)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.02)

# Algorithms
gmsh.option.setNumber("Mesh.Algorithm",   6)  # Frontal-Delaunay (2D)
gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay (3D)

# Quality optimization
gmsh.option.setNumber("Mesh.Optimize",       1)
gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

# Generate mesh
gmsh.model.mesh.generate(3)
gmsh.model.mesh.optimize("Netgen")
gmsh.model.mesh.optimize("Relocate3D")

# Convert to DOLFINx
mesh_comm = MPI.COMM_WORLD
mesh_data = gmshio.model_to_mesh(gmsh.model, mesh_comm, 0, gdim=3)
mesh = mesh_data.mesh

gmsh.finalize()

print(f"Mesh OK:")
print(f"  Cells (tetrahedra): {mesh.topology.index_map(3).size_global}")
print(f"  Vertices:           {mesh.topology.index_map(0).size_global}")

# Plot and save
topology, cell_types, geometry = dolfinx.plot.vtk_mesh(mesh, mesh.topology.dim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# Extract ONLY the exterior surface — no interior visible
surface = grid.extract_surface(algorithm="dataset_surface")

plotter = pyvista.Plotter(off_screen=True)

plotter.add_mesh(
    surface,
    show_edges=True,
    color="white",
    opacity=1.0,
    edge_color="navy",
    line_width=0.8,
    lighting=True,
    smooth_shading=True,
)

plotter.set_background("#3a3a5c")

plotter.add_axes()
plotter.camera_position = [
    (3.5, -3.5, 2.5),
    (0.0,  0.0, 0.0),
    (0.0,  0.0, 1.0),
]
plotter.screenshot("outputs/step1_cylinder_mesh.png")
plotter.close()

print("Saved: outputs/step1_cylinder_mesh.png")