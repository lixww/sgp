# %%
import pandas as pd
import numpy as np
from skimage.io import imsave
import matplotlib.pyplot as plt


# file paths
folder_path = '~/develop/projects/sgp/'
data_path = folder_path + 'networks/data/sgp'
img_save_path = folder_path + 'autoencoder'


# prepare training set

training_file = pd.read_csv(f'{data_path}/training_file_8_bit.csv')

location_head = training_file.columns[2:4]
location = training_file[location_head].to_numpy()

channel_head = training_file.columns[4:]
channel = training_file[channel_head].to_numpy()

y_true = training_file['class_name'].to_numpy()

data_idx = training_file.index

channel_len = len(channel_head)

# %%
# statistics
y_o = 0
y_b = 0
y_u = 0
for y in y_true:
    if y == 0:
        y_o += 1
    elif y == 1:
        y_b += 1
    elif y == 2:
        y_u += 1

print('overtext: ', y_o)
print('background: ', y_b)
print('undertext: ', y_u)
print('overall:', y_true.shape)


# %%
# plot locations

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


# %%
# plot channels

idx_o = []
idx_b = []
idx_u = []

for i in data_idx:
    if y_true[i] == 0:
        idx_o.append(i)
    elif y_true[i] == 1:
        idx_b.append(i)
    elif y_true[i] == 2:
        idx_u.append(i)


x_axis = range(channel_len)

plt.figure()

plt.subplot(311)
for i in idx_o:
    plt.plot(x_axis, channel[i])
plt.ylim([0, 256])
plt.title('overtext')

plt.subplot(312)
for i in idx_b:
    plt.plot(x_axis, channel[i])
plt.ylim([0, 256])
plt.title('background')

plt.subplot(313)
for i in idx_u:
    plt.plot(x_axis, channel[i])
plt.ylim([0, 256])
plt.title('undertext')

plt.show()



# %%
