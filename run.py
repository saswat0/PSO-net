import h5py

from datagen import *
from func import *

# Read HDF
hdf_data_path = 'voxforge.h5'
data_store = h5py.File(hdf_data_path,'r') 

# Generator
train_generator = data_gen(data_store['X_Train'], data_store['Y_Train'], batch_size = 128)
test_generator = data_gen(data_store['X_Test'], data_store['Y_Test'], batch_size = 1)

# Training without optimisation
acc_without = fitness_without_optimization(train_generator, test_generator)
print('accuracy without optimization : ' + str(acc_without))

# Training with optimisation
pso_calculate(92, train_generator, test_generator)