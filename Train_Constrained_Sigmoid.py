# Import the required cbrain functions
#from cbrain.imports import *
import os 
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from imports import *
from data_generator import *
from models import *
import time
from utils import limit_mem


t0 = time.time()

# If you are running on the GPU, execute this
# Otherwise tensorflow will use ALL your GPU RAM for no reason
limit_mem()

DATADIR = 'Preprocessed_Data/SPCAM5_12_Months/'

train_gen = DataGenerator(
    data_dir=DATADIR, 
    feature_fn='full_physics_essentials_train_month01_shuffle_features.nc',
    target_fn='full_physics_essentials_train_month01_shuffle_targets.nc',
    batch_size=512,
    norm_fn='full_physics_essentials_train_month01_norm.nc',
    fsub='feature_means', 
    fdiv='feature_stds', 
    tmult='target_conv',
    shuffle=True,
)

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


from keras.models import Sequential
from keras.layers import *


print(train_gen.feature_shape)
print(train_gen.target_shape)


#build the model
#key objects
#standard config
hidden_layers = [128,128,128,128,128]


#custom activation function code
from keras import backend as K

activation = 'LeakyReLU'
loss = 'mse'
lr = 0.0001
reg = None
valid_after = True
#Input layer
#Should be 64
inp = Input(shape=(train_gen.feature_shape,))

#hidden layers
#create an initial hidden layer
x = Dense(hidden_layers[0], kernel_regularizer=reg)(inp)
x = act_layer(activation)(x)
#loop through to create a deep nueral net expect the first hidden layer already constructed
for i in range(len(hidden_layers)-1):
    x = Dense(hidden_layers[i], kernel_regularizer=reg)(x)
    x = act_layer(activation)(x)

#Utilize Functional API?
branchA = Dense(64, kernel_regularizer=reg)(x)
branchA = act_layer(activation)(branchA)

#standard relu case
#https://keras.io/api/layers/activations/
activation = 'sigmoid'
branchB = Dense(1, kernel_regularizer=reg)(x)
branchB = act_layer(activation)(branchB)

x= concatenate([branchA, branchB])
print('New Model')
print('New Model')



# Now compile the model
model = Model(inp, x)
opt = keras.optimizers.RMSprop(lr=lr)
model.compile(optimizer=opt, loss=loss, metrics=metrics)

#model.compile(Adam(lr), loss=loss, metrics=metrics)
print('Compiled')

model.summary()

#added in by griffin for early stop/best model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.h5', save_best_only=True, monitor='val_loss', mode='min')
                              
#end of add in                              
                              

# Finally train the model
print('train_gen_n_batches: ', train_gen.n_batches)
print('valid_gen_n_batches', valid_gen.n_batches)

callbacks = [keras.callbacks.TensorBoard(log_dir='my_log_dir/Sigmoid', histogram_freq=0), earlyStopping, mcp_save]

h = model.fit_generator(
    train_gen.return_generator(False, False),   # This actually returns the generator
    train_gen.n_batches,
    epochs=24, 
    validation_data=valid_gen.return_generator(False, False),
    validation_steps=valid_gen.n_batches, workers =16, max_queue_size =50, callbacks=callbacks
)


#other visual tensorboard information
from keras.utils import plot_model
import pydot
from keras.utils.vis_utils import model_to_dot
keras.utils.vis_utils.pydot = pydot
plot_model(model, to_file='my_log_dir/Sigmoid/model.png')
plot_model(model, show_shapes=True, to_file='my_log_dir/Sigmoid/models.png')



#Save the model
model.save('Models/Annual_Sigmoid.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model
print('It Finished')



import matplotlib.pyplot as plt

t1 = time.time()
total = t1-t0
total = total/(60.0*60.0)

hdict1 = h.history
train_loss_values1 = hdict1['loss']
valid_loss_values1 = hdict1['val_loss']
epochs1 = range(1, len(train_loss_values1) + 1)
plt.plot(epochs1, train_loss_values1, 'bo', label='Train loss')
plt.plot(epochs1, valid_loss_values1, 'b', label='Valid loss')
plt.title('Training and validation loss - SPCAM5; time:'+str(total)[:8]+' hours')
plt.xlabel('Epochs', fontsize = 20)
plt.ylabel('Loss', fontsize = 20)
#plt.yscale('log')
plt.legend()
plt.savefig('Figures/Annual_Sigmoid.png')