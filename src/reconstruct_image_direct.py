import dolfinx
import eitx
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf
from tensorflow import keras

DATAMAT_PATH = 'data/lab_measurements/20241111' #"ifsc_data2/20240506"
bg_estimated =   0.007 # 0.04    #estimated background
rotacao = 90            #rotation to plot the images

'Load files'
# Load data of background
mat = scipy.io.loadmat(f"{DATAMAT_PATH}/Dados/Vref.mat")
#  mat = scipy.io.loadmat(f"{DATAMAT_PATH}/datamat/Referencia.mat")
Uel=mat.get("signal_peak")
#print(Uel.shape)
# plt.plot(Uel.T, '-r')
Uel_b = Uel.reshape(16,16)    #Matrix of measuarements
l, L = np.shape(Uel_b)  #Number of experiments, Number of Electrodes

#Plot
#fig, ax = plt.subplots(figsize=(8,5))
#for i, U_vec in enumerate(Uel_b):
#    x=np.linspace(1,1.8,L)+i
#    ax.plot(x,U_vec, linewidth=1.3, marker='.', markersize=5);

'Forces sum = 0 on each experiment'
# print(np.sum(Uel_b[0]))
for i in range(L):
  Uel_b[i][i] = -np.sum(Uel_b[i]) + Uel_b[i][i]
  #Uel_b[i][i]=-Uel_b[i][i] #Em cada primeiro emissor, troque de sinal
  # Uel_b[i] -= np.sum(Uel_b[i])/L #Force a soma ser zero em cada experimento

#Plot
#fig, ax = plt.subplots(figsize=(8,5))
#for i, U_vec in enumerate(Uel_b):
#    x=np.linspace(1,1.8,L)+i
#    ax.plot(x,U_vec, linewidth=1.3, marker='.', markersize=5);

'Convert type of data (differential to absolute)'
list_U0_m=np.zeros_like(Uel_b)
for index, potential in enumerate(Uel_b):
    list_U0_m[index]=eitx.ConvertingData(potential, method="KIT4")
list_U0_m = -list_U0_m #/np.max(list_U0_m)

#Plot
#fig, ax = plt.subplots(figsize=(8,5))
#for i, U_vec in enumerate(list_U0_m):
#    x=np.linspace(1,1.8,L)+i
#    ax.plot(x,U_vec, linewidth=1.3, marker='.', markersize=5)
#plt.show()

'Basic Definitions of FEM mesh'
radius=1       #Circle radius
per_cober=0.3543  #Percentage of area covered by electrodes
rotate= 0      #Electrodes Rotation
z = np.ones(L)*1e-3                         #Impedance of each electrode

'Return object with angular position of each electrode'
ele_pos = eitx.Electrodes(L, per_cober, rotate,anticlockwise=False)

'Mesh'
mesh_object = eitx.MeshClass(ele_pos,0.4,0.6)

'Forward problem'
dir_problem = eitx.DirectProblem(mesh_object,z)
V0 = dir_problem.V0   # Discontinuous Garlekin space function
# V = dir_problem.V     # Continuous Garlekin space function

'Homogeneus mesh'
N = 128               # grid with N*N points (works well with 0 < N < 400)
h = 2*radius/(N-1)    # step-size
x = [radius - i*h for i in range(N)]  # x grid points
y = [-radius + i*h for i in range(N)] # y grid points

# MESH x and y
mesh_x = np.zeros((N,N))                              # x-Data (input of CNN)
mesh_y = np.zeros((N,N))                              # y-Data (input of CNN)
for i in range(N):
  for j in range(N):
    mesh_x[i][j] = x[i]
    mesh_y[i][j] = y[j]

'Measurements to be Excluded (emissors)'
ME = []
ME.append([0, 1, 15])
i, j, k = 0, 1, 2
ME.append([i, j, k])
while k < 15:
  i, j, k = i+1, j+1, k+1
  ME.append([i, j, k])
ME.append([0, 14, 15])
#print(ME)

"Define gamma as constant = Background"
gamma0 = dolfinx.fem.Function(V0) #Define the function with basis DG
iv, bg= 10, 1/bg_estimated
gamma0.x.array[:] = bg

'Measurements'
exper = [name.replace(".mat","") for name in os.listdir(DATAMAT_PATH+'/Dados')]
n_exper = len(exper)

'Prepare data to input the neural network'
T1 = []
for sample in range(n_exper):
  'Get data'
  mat = scipy.io.loadmat(DATAMAT_PATH+'/Dados/' +exper[sample]+".mat")
  # mat = scipy.io.loadmat(exper)
  Uel=mat.get("signal_peak").T
  Uel_f=Uel.reshape(l,L) #Matrix of measuarements

  'Forces sum = 0 on each experiment'
  for i in range(0,L):
    Uel_f[i][i] = -np.sum(Uel_f[i]) + Uel_f[i][i]
  
  'Convert type of data (differential to absolute)'
  list_U1_m=np.zeros_like(Uel_f)
  for index, potential in enumerate(Uel_f):
      list_U1_m[index]=eitx.ConvertingData(potential, method="KIT4")
  list_U1_m = -list_U1_m #/np.max(list_U1_m)
  
  'Difference of potential'
  differ = [list_U1_m[k] - list_U0_m[k] for k in range(len(list_U0_m))]
  for s in range(16):
    differ[s][ME[s]] = 0      #Put zeros on measurements involving the emissors
  for i in range(len(differ)):
    differ[i] = differ[i] - np.sum(differ[i])/13
  for s in range(16):
      differ[s][ME[s]] = 0

  'Solve Forward Problem with gamma=Background and Current=(Difference of Potentials)'
  list_ur_dif, list_U_dif = dir_problem.solve_problem_current(differ, gamma0)

  'Define data in the homogeneus grid to input the Neural Network'
  T = np.zeros((l + 2,N,N))
  for k in range(l):
    T[k] = eitx.genPotentialImg(list_ur_dif[k],mesh_x,mesh_y,bg)

  T[l] = mesh_x
  T[l+1] = mesh_y
  T1.append(np.transpose(T))
input_val = tf.convert_to_tensor(T1)
print(np.array(input_val).shape)
mat.keys()

'Upload Neural Network'
with open('models/unet/config.json') as f:
    model_config = f.read()
model = keras.models.model_from_json(model_config)
model.load_weights('models/unet/model.weights.h5')
model.summary()

'Predict and prepare images to plot'
from scipy.ndimage import rotate

classes = model.predict(input_val)
result = 0.5*np.ones((n_exper,N,N))
for k in range(n_exper):
  result1 = rotate(classes[k],rotacao,reshape = False)
  for i in range(N):
    for j in range(N):
      if x[i]**2 + y[j]**2 > radius**2:
        result1[i][j] = 0.5
  result[k, :, :] = result1[:,:,0]
#result.shape

'Prepare target photo list'
photo_array = []
for test in range(len(exper)):
  img = np.asarray(Image.open(f'{DATAMAT_PATH}/Fotos/' + exper[test] + '.jpg'))
  photo_array.append(img)

'Plot'
fig, ax = plt.subplots(result.shape[0],2,figsize=(10,40))
img_array = []
for k in range(result.shape[0]):
  img_array.append(ax[k][0].imshow(result[k], interpolation='none',vmin=-1.0,vmax=1.0))
  ax[k][0].set_axis_off()
  ax[k][1].imshow(photo_array[k])
  ax[k][1].set_axis_off()
fig.colorbar(img_array[0],ax=ax,orientation='vertical')

'Save images'
plt.savefig('outputs/reconstruction_result_direct.png')