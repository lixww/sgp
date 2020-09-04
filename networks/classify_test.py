import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
from torch.utils.data import DataLoader
from torch import nn

import models
from utils import FolioDataset, load_labeled_dataset



data_class = 'allClass'

# file paths
model_path = 'networks/model'


# prepare test set
test_dataset, channel_len, _ = load_labeled_dataset()


# load model
autoencoder = models.sdae(
    dimensions=[channel_len, 10, 10, 20, 3]
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