import pandas as pd
import numpy as np
from skimage.io import imsave


# file paths
data_path = 'autoencoder/data/sgp'
img_save_path = 'autoencoder'


# prepare training set

training_file = pd.read_csv(f'{data_path}/training_file_8_bit.csv')

location_head = training_file.columns[2:4]
location = training_file[location_head].to_numpy()
location = location.T

width = max(location[0])
height = max(location[1])
loc_img = np.ones((height,width))
loc_img *= 255
duplicate_count = 0
for index in range(len(location[0])):
    x = location[0][index] -1
    y = location[1][index] -1
    if loc_img[y][x] == 255:
        loc_img[y][x] = 0
    else:
        duplicate_count += 1

imsave(f'{img_save_path}/analysis.png', loc_img)
print('duplicate_count: ', duplicate_count)
