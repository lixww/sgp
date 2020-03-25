import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn

from classes import simple_ae
from utils import FolioDataset



data_class = 'notUclass'

# file paths
data_path = 'autoencoder/data/sgp'
model_path = 'autoencoder/model'


# prepare test set

test_file = pd.read_csv(f'{data_path}/uclass_8_bit.csv')

location_head = test_file.columns[2:4]
channel_head = test_file.columns[4:]

y_true = test_file['class_name'].to_numpy()
location = test_file[location_head].to_numpy()
channel = test_file[channel_head].to_numpy()

data_idx = test_file.index


# load data
full_dataset = FolioDataset(location, channel, y_true)
dataloader = DataLoader(full_dataset, batch_size=50, shuffle=False)

# load model
model = simple_ae(len(channel_head))
model.load_state_dict(torch.load(f'{model_path}/ae_on_{data_class}.pth', map_location='cpu'))

# model setting
criterion = nn.MSELoss()

# view output
loss = []
with torch.no_grad():
    for data in dataloader:
        inp = data[1]
        
        output = model(inp)
        loss.append(criterion(output, inp).numpy())

loss = np.array(loss).flatten()
print('average loss: ', np.mean(loss))