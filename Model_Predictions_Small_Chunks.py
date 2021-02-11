# mport the required cbrain functions
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
print('Starting')


#If running on the GPU or GPU-shared partition uncomment
limit_mem()


DATADIR = 'Preprocessed_Data/Summer_2021_From_Annual/'



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


model = keras.models.load_model('Models/Annual_Exponential.h5')

path_to_file = 'Preprocessed_Data/Summer_2021_From_Annual/full_physics_essentials_valid_month02_features.nc'
real_ds = xr.open_dataset(path_to_file)
features = real_ds.features[:, :]
print(features.shape)

print('files imported')
model_data = np.empty(shape=(len(features), output_vector))

segments = int(len(features)/100000)
steps = segments+1
start = 0
gap = 100000

for i in range(steps):
    if i <= steps-2:
        print(i)
        feature_here=features[start:gap].values
        f = feature_here-fsub
        f = f/fdiv
        f=f.reshape(-1,1)
        x = np.reshape(f, (100000,64))
        p = model.predict_on_batch(x)
        p = p/tdiv
        p = p+tsub
        model_data[start:gap,:] = p 
        start = start+100000
        gap = gap+ 100000
    else:
        feature_here=features[start:].values
        f = feature_here-fsub
        f = f/fdiv
        f=f.reshape(-1,1)
        x = np.reshape(f, (len(features[start:].values),64))
        p = model.predict_on_batch(x)
        p = p/tdiv
        p = p+tsub
        model_data[start:,:] = p 

print('made it')



print('creating nc file')

lev = np.arange(len(model_data[0]))
sample = np.arange(len(model_data))

myda = xr.DataArray(model_data, coords = {'sample': sample, 'lev': lev}, dims=('sample', 'lev'))
myda.name = 'Prediction'
myds = myda.to_dataset()
myds.to_netcdf('Models/Annual_Exponential_Summer.nc')


t1 = time.time()
total = t1-t0
total = total/(60.0*60.0)
print(total)
print('done')