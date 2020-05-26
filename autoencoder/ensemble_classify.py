import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
from torch.utils.data import DataLoader
from torch import nn

from sklearn.metrics import precision_score

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
full_dataset = FolioDataset(location, channel, y_true)

# load model
channel_len = len(channel_head)

autoencoder = models.sdae(
    dimensions=[channel_len, 10, 10, 20, 3]
)
model_ae = models.sdae_lr(autoencoder)
model_ae.load_state_dict(torch.load(f'{model_path}/ae_on_{data_class}.pth', map_location='cpu'))
model_ae.eval()

model_conv = models.conv1d_net(channel_len, 3)
model_conv.load_state_dict(torch.load(f'{model_path}/conv1d_on_{data_class}.pth', map_location='cpu'))
model_conv.eval()


# ensemble prediction
prob_ae = models.encode(full_dataset, model_ae)
prob_conv = models.encode(full_dataset, model_conv)

prob_ensemb = torch.add(prob_ae, prob_conv)
pred_result = []
for prob in prob_ensemb:
    _, prediction = torch.max(prob.data, 0)
    pred_result.append(prediction.item())

precision = precision_score(y_true, pred_result, average='micro')

print('accuracy: ', precision)