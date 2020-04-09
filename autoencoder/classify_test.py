import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
from torch.utils.data import DataLoader
from torch import nn

import models
from utils import FolioDataset



data_class = 'allClass'

# file paths
data_path = 'autoencoder/data/sgp'
model_path = 'autoencoder/model'


# prepare test set

test_file = pd.read_csv(f'{data_path}/training_file_8_bit.csv')

location_head = test_file.columns[2:4]
channel_head = test_file.columns[4:]

y_true = test_file['class_name'].to_numpy()
location = test_file[location_head].to_numpy()
channel = test_file[channel_head].to_numpy()

data_idx = test_file.index

# load data
test_dataset = FolioDataset(location, channel, y_true, 
                            location_head=location_head,
                            channel_head=channel_head)


# load model
autoencoder = models.sdae(
    dimensions=[len(channel_head), 10, 10, 20, 3]
)
model = models.sdae_lr(autoencoder)
model.load_state_dict(torch.load(f'{model_path}/ae_on_{data_class}.pth', map_location='cpu'))
model.eval()

# view output

acc = models.cal_accuracy(test_dataset, model)
print('accuracy: ', acc)

features = models.encode(test_dataset, model)
features = features.numpy()
features_with_label = {0:[], 1:[], 2:[]}
for i in range(len(test_dataset)):
    label = test_dataset.grdtruth[i].item()
    features_with_label[label].append(features[i])

fig = plt.figure()
ax = Axes3D(fig)

colors = ['r', 'g', 'b']
for i in range(3):
    class_f = np.array(features_with_label[i])
    class_f = class_f.T
    ax.scatter(class_f[0], class_f[1], class_f[2], c=colors[i])

plt.show()