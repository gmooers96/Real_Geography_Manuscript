# Import the required cbrain functions
from imports import *
from data_generator import *
from models import *

#shouldn't need on a local computer?
from utils import limit_mem

#Module to save and load models
import h5py
import netCDF4
import numpy as np
#Keras
from keras.models import Sequential
from keras.layers import *
import time
#added by Griffin
import xarray as xr

t0 = time.time()

output_vector = 65
input_vector = 64
#Attempt to deal with dead kernel
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True   # Allocates as much memory as needed.
#keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
#sess = tf.Session(config=config)
print('Starting')


#If running on the GPU or GPU-shared partition uncomment
limit_mem()


DATADIR = 'Preprocessed_Data/Paper_Annual/'



valid_gen = DataGenerator(
    data_dir=DATADIR, 
    feature_fn='full_physics_essentials_valid_month02_features.nc',
    target_fn='full_physics_essentials_valid_month02_targets.nc',
    batch_size=512,
    norm_fn='full_physics_essentials_train_month01_norm.nc',  # SAME NORMALIZATION FILE!
    fsub='feature_means', 
    fdiv='feature_stds', 
    tmult='target_conv',
    shuffle=False,
)


fsub = valid_gen.feature_norms[0]
fdiv = valid_gen.feature_norms[1]
tsub = valid_gen.target_norms[0]
tdiv = valid_gen.target_norms[1]

def custom_activation(z):
    #y = x**10*K
    #y = K.exp(z)
    y = z**10
    return y

#load the trained model
#with custom activation
#model = keras.models.load_model('Models/Summer_Months_10x.h5', custom_objects = {"custom_activation": custom_activation})

#normal
model = keras.models.load_model('Models/Sigmoid.h5')

#Jordan
#model = tf.keras.models.load_model('Models/ContinentModels/second_round_model.h5')
#nc file
#model = keras.models.load_model('Models/Very_Small_Sherpa.h5')
#dataset = netCDF4.Dataset("Preprocessed_Data/Better_1_Year_Valid/full_physics_essentials_small_valid_month02_features.nc")
#features = np.array(dataset.variables['features'])

path_to_file = 'Preprocessed_Data/Paper_Annual/full_physics_essentials_valid_month02_features.nc'
real_ds = xr.open_dataset(path_to_file)
features = real_ds.features[:, :].values
#features = real_ds.features[:, :]
print(features.shape)

print('files imported')
model_data = np.zeros(shape=(len(features)))
model_data[:] = np.nan

segments = int(len(features)/100000)
steps = segments+1
start = 0
gap = 100000

for i in range(steps):
    if i <= steps-2:
        print(i)
        feature_here=features[start:gap]
        f = feature_here-fsub
        f = f/fdiv
        #f = f.values
        f=f.reshape(-1,1)
        x = np.reshape(f, (100000,64))
        p = model.predict_on_batch(x)
        p = p/tdiv
        #s = np.reshape(p, (65,100000))
        p = p+tsub
        model_data[start:gap] = np.squeeze(p[:,-1])  
        start = start+100000
        gap = gap+ 100000
    else:
        feature_here=features[start:]
        f = feature_here-fsub
        f = f/fdiv
        #f = f.values
        f=f.reshape(-1,1)
        x = np.reshape(f, (len(features[start:]),64))
        p = model.predict_on_batch(x)
        #s = np.reshape(p, (65,100000))
        p = p/tdiv
        p = p+tsub
        model_data[start:] = np.squeeze(p[:,-1]) 

print('made it')



print('creating nc file')

#lev = np.arange(len(model_data[0]))
sample = np.arange(len(model_data))

myda = xr.DataArray(model_data, coords = {'sample': sample}, dims=('sample'))
myda.name = 'Prediction'
myds = myda.to_dataset()
myds.to_netcdf('Models/Annual_Sigmoid.nc')
#myds.to_netcdf('Models/Paper_Full_2018_Primative.nc')


t1 = time.time()
total = t1-t0
total = total/(60.0*60.0)
print(total)
print('done')