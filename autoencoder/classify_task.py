import pandas as pd

from sklearn.model_selection import ShuffleSplit

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from torchvision import transforms

import models 
from utils import FolioDataset



data_class = 'allClass'

# file paths
data_path = 'autoencoder/data/sgp'
model_path = 'autoencoder/model'



# prepare training set

training_file = pd.read_csv(f'{data_path}/training_file_8_bit.csv')

location_head = training_file.columns[2:4]
channel_head = training_file.columns[4:]

y_true = training_file['class_name'].to_numpy()
location = training_file[location_head].to_numpy()
channel = training_file[channel_head].to_numpy()

data_idx = training_file.index


# load data
full_dataset = FolioDataset(location, channel, y_true)
# split into train & test?
#


# hyperparameter
learning_rate = 1e-2
num_epoch = 50


# simple ae
# models.train_simple_ae(
#     dataset=full_dataset, 
#     inp_dim=len(channel_head),
#     learning_rate=learning_rate,
#     num_epoch=num_epoch,
#     save_path=f'{model_path}/ae_on_{data_class}.pth'
# )



# sdae-lr

# hyperparameter
pretrain_epoch = 30
finetune_epoch = 50


autoencoder = models.sdae(
    dimensions=[len(channel_head), 250, 250, 1000, 3]
)
models.pretrain(
    dataset=full_dataset,
    autoencoder=autoencoder,
    num_epoch=pretrain_epoch
)
models.fine_tune(
    dataset=full_dataset,
    autoencoder=autoencoder,
    num_epoch=finetune_epoch
)