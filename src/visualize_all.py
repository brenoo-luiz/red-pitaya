"""
Visualizações EXATAS que estavam comentadas no código original
(generate_synthetic_data.py e reconstruct_image_direct.py), salvas direto
como PNG em outputs/, sem abrir janelas interativas.

Também inclui plot_tent_function e plot_indicator_function de eitx.py,
que existem como funções reais na biblioteca mas não eram chamadas
em nenhum lugar do pipeline.
"""
import os
import json
import numpy as np
import scipy
import matplotlib.pyplot as plt
import dolfinx
import pyvista
import eitx

pyvista.OFF_SCREEN = True
os.makedirs("outputs", exist_ok=True)

# =====================================================================
# PARTE 1 — blocos comentados em generate_synthetic_data.py
# =====================================================================

with open("settings/data_gen_settings.json") as f:
    settings = json.loads(f.read())

L = settings['L']
n_g = len(settings["currents"])
I_all = eitx.current_method(L, n_g, method=2)

radius = 1
per_cober = 0.3543
rotate = 0
z = np.ones(L) * 1e-3

ele_pos = eitx.Electrodes(L, per_cober, rotate, anticlockwise=False)
mesh_object = eitx.MeshClass(ele_pos, 0.4, 0.6)

dir_problem = eitx.DirectProblem(mesh_object, z)
V0 = dir_problem.V0

bg = settings['bg']
gamma0 = dolfinx.fem.Function(V0)
gamma0.x.array[:] = bg

list_u, list_U0_m_gen = dir_problem.solve_problem_current(I_all, gamma0)

mesh = mesh_object.mesh

# 1. Malha
p2 = pyvista.Plotter(off_screen=True)
grid2 = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh))
p2.add_mesh(grid2, show_edges=True)
p2.view_xy()
p2.set_background("white")
p2.screenshot("outputs/plot_malha.png")
p2.close()
print("Salvo: outputs/plot_malha.png")

# 7. Potencial elétrico u em 3D (função tenda) — eitx.plot_tent_function
# Função existente em eitx.py, nunca chamada em nenhum script do pipeline
eitx.plot_tent_function(list_u[0], savefile=True, filename="outputs/plot_tent_potencial")
print("Salvo: outputs/plot_tent_potencial.png")

# Gera uma amostra com inclusão pra poder mostrar o plot de 'differ'
gamma = eitx.GammaCircle(V0, 1, bg, 0, 0, 0)
gamma_prov = gamma.x.array
iv = settings['ivhigh']
ValuesCells1 = eitx.GammaCircle(V0, iv - bg, 0.0, 0.2, 0.0, 0.0).x.array
gamma_prov = gamma_prov + ValuesCells1
gamma.x.array[:] = gamma_prov

# 8. Indicadora de condutividade gamma — eitx.plot_indicator_function
# Função existente em eitx.py, nunca chamada em nenhum script do pipeline
eitx.plot_indicator_function(gamma, savefile=True, filename="outputs/plot_indicator_gamma")
print("Salvo: outputs/plot_indicator_gamma.png")

list_u1, list_U1_m = dir_problem.solve_problem_current(I_all, gamma)
differ = np.array(list_U1_m) - np.array(list_U0_m_gen)

ME = []
ME.append([0, 1, 15])
i, j, k = 0, 1, 2
ME.append([i, j, k])
while k < 15:
    i, j, k = i+1, j+1, k+1
    ME.append([i, j, k])
ME.append([0, 14, 15])

for s in range(16):
    differ[s][ME[s]] = 0
for i in range(differ.shape[0]):
    differ[i] = differ[i] - np.sum(differ[i])/13
for s in range(16):
    differ[s][ME[s]] = 0

# 2. Diferença de potencial (com inclusão - sem inclusão)
fig, ax = plt.subplots(figsize=(8,5))
for w, U_vec in enumerate(differ):
    zx = np.linspace(1,1.8,L) + w
    ax.plot(zx,U_vec, linewidth=1.3, marker='.', markersize=5)
plt.savefig("outputs/plot_differ.png", dpi=120)
plt.close(fig)
print("Salvo: outputs/plot_differ.png")

# =====================================================================
# PARTE 2 — blocos comentados em reconstruct_image_direct.py
# =====================================================================

DATAMAT_PATH = 'data/lab_measurements/20241111'

mat = scipy.io.loadmat(f"{DATAMAT_PATH}/Dados/Vref.mat")
Uel = mat.get("signal_peak")

# 3. Sinal bruto
fig = plt.figure()
plt.plot(Uel.T, '-r')
plt.savefig("outputs/plot_Uel_raw.png", dpi=120)
plt.close(fig)
print("Salvo: outputs/plot_Uel_raw.png")

Uel_b = Uel.reshape(16,16)
l, L2 = np.shape(Uel_b)

# 4. Uel_b antes de forçar soma=0
fig, ax = plt.subplots(figsize=(8,5))
for i, U_vec in enumerate(Uel_b):
    x = np.linspace(1,1.8,L2)+i
    ax.plot(x,U_vec, linewidth=1.3, marker='.', markersize=5)
plt.savefig("outputs/plot_Uel_b_antes.png", dpi=120)
plt.close(fig)
print("Salvo: outputs/plot_Uel_b_antes.png")

for i in range(L2):
    Uel_b[i][i] = -np.sum(Uel_b[i]) + Uel_b[i][i]

# 5. Uel_b depois de forçar soma=0
fig, ax = plt.subplots(figsize=(8,5))
for i, U_vec in enumerate(Uel_b):
    x = np.linspace(1,1.8,L2)+i
    ax.plot(x,U_vec, linewidth=1.3, marker='.', markersize=5)
plt.savefig("outputs/plot_Uel_b_depois.png", dpi=120)
plt.close(fig)
print("Salvo: outputs/plot_Uel_b_depois.png")

list_U0_m = np.zeros_like(Uel_b)
for index, potential in enumerate(Uel_b):
    list_U0_m[index] = eitx.ConvertingData(potential, method="KIT4")
list_U0_m = -list_U0_m

# 6. list_U0_m (convertido, formato final de referência)
fig, ax = plt.subplots(figsize=(8,5))
for i, U_vec in enumerate(list_U0_m):
    x = np.linspace(1,1.8,L2)+i
    ax.plot(x,U_vec, linewidth=1.3, marker='.', markersize=5)
plt.savefig("outputs/plot_list_U0_m.png", dpi=120)
plt.close(fig)
print("Salvo: outputs/plot_list_U0_m.png")

print("\nTodas as 8 visualizações salvas em outputs/, sem janelas.")