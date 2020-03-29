import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch import nn

from models import simple_ae
from utils import FolioDataset



data_class = 'notUclass'

# file paths
data_path = 'autoencoder/data/sgp'
model_path = 'autoencoder/model'


def prepare_dataset(csv_f):
    # prepare test set

    test_file = pd.read_csv(csv_f)

    location_head = test_file.columns[2:4]
    channel_head = test_file.columns[4:]

    y_true = test_file['class_name'].to_numpy()
    location = test_file[location_head].to_numpy()
    channel = test_file[channel_head].to_numpy()

    data_idx = test_file.index

    # load data
    res_dataset = FolioDataset(location, channel, y_true, 
                                location_head=location_head,
                                channel_head=channel_head)

    return res_dataset


test_dataset = prepare_dataset(f'{data_path}/uclass_8_bit.csv')
dataloader = DataLoader(test_dataset, batch_size=50, shuffle=False)

train_dataset = prepare_dataset(f'{data_path}/{data_class}_8_bit.csv')
train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=False)

# load model
model = simple_ae(len(test_dataset.channel_head))
model.load_state_dict(torch.load(f'{model_path}/ae_on_{data_class}.pth', map_location='cpu'))

# model setting
criterion = nn.MSELoss()

# view output

def get_model_output(model, dataloader):
    # reconstruction loss & distribution in latent space

    loss = []
    latent_v = []

    with torch.no_grad():
        for data in dataloader:
            inp = data[1]
            
            output = model(inp)
            loss.append(criterion(output, inp))

            encoding = model.encoder(inp)
            if len(latent_v) <= 0:
                latent_v = encoding
                continue
            latent_v = torch.cat((latent_v, encoding))

    return loss, latent_v


loss, latent_v = get_model_output(model, dataloader)
train_loss, train_latent_v = get_model_output(model, train_dataloader)

# average loss 
print('average loss: ', np.mean(loss))
print('average loss on train set: ', np.mean(train_loss))

# plot 

latent_v = latent_v.numpy().T
train_latent_v = train_latent_v.numpy().T

plt.scatter(latent_v[0], latent_v[1], c='red', marker='o')
plt.scatter(train_latent_v[0], train_latent_v[1], c='blue', marker='1')

plt.show()