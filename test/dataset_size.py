import numpy as np
import os
# train = np.load("data/Dendrite_growth/train.npy")
# test_10 = np.load("data/Dendrite_growth/test.npy")
# valid = np.load("data/Dendrite_growth/test.npy")
# test_50 = np.load("data/Dendrite_growth/Dendrite_growth_test_10_50.npy")
# test_90 = np.load("data/Dendrite_growth/Dendrite_growth_test_10_90.npy")

# train = np.load("data/Plane_wave_propagation/Plane_wave_propagation_train_10_10.npy")
# test_10 = np.load("data/Plane_wave_propagation/Plane_wave_propagation_test_10_10.npy")
# test_50 = np.load("data/Plane_wave_propagation/Plane_wave_propagation_test_10_50.npy")
# test_90 = np.load("data/Plane_wave_propagation/Plane_wave_propagation_test_10_90.npy")
# valid = np.load("data/Plane_wave_propagation/Plane_wave_propagation_valid_10_10.npy")


# train = np.load("data/Spinodal_decomposition/Spinodal_decomposition_train_10_10.npy")
# test_10 = np.load("data/Spinodal_decomposition/Spinodal_decomposition_test_10_10.npy")
# test_50 = np.load("data/Spinodal_decomposition/Spinodal_decomposition_test_10_50.npy")
# test_90 = np.load("data/Spinodal_decomposition/Spinodal_decomposition_test_10_90.npy")
# valid = np.load("data/Spinodal_decomposition/Spinodal_decomposition_valid_10_10.npy")


# train = np.load("data/Grain_growth/Grain_growth_train_10_10.npy")
test_10 = np.load("data/Grain_growth/Grain_growth_test_10_10.npy")
test_50 = np.load("data/Grain_growth/Grain_growth_test_10_50.npy")
test_90 = np.load("data/Grain_growth/Grain_growth_test_10_90.npy")
# valid = np.load("data/Grain_growth/Grain_growth_valid_10_10.npy")

data_name = 'Grain_growth'
path = f'new_data/{data_name}/'
os.makedirs(path, exist_ok=True)

# np.save(path+f'{data_name}_train_10_10.npy', train[:18, :, :, :]) # 18 for den 
# np.save(path+f'{data_name}_valid_10_10.npy', valid[:18, :, :, :]) # 18 for plane
np.save(path+f'{data_name}_test_10_10.npy', test_10[:18, :, :, :]) # 18 for spin
np.save(path+f'{data_name}_test_10_50.npy', test_50[:6, :, :, :])
np.save(path+f'{data_name}_test_10_90.npy', test_90[:4, :, :, :])

# print("shape of training set : ", train.shape)
# print("shape of valid set : ", valid.shape)
print("shape of test set(10-10) : ", test_10.shape)
print("shape of test set(10-50) : ", test_50.shape)
print("shape of test set(10-90) : ", test_90.shape)