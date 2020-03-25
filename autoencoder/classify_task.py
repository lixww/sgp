import pandas as pd

from sklearn.model_selection import ShuffleSplit

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from torchvision import transforms

from classes import simple_ae
from utils import FolioDataset




data_class = 'notUclass'

# file paths
data_path = 'autoencoder/data/sgp'
model_path = 'autoencoder/model'



# prepare training set

training_file = pd.read_csv(f'{data_path}/notUclass_8_bit.csv')

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
dataloader = DataLoader(full_dataset, batch_size=50, shuffle=True)



# hyperparameter
learning_rate = 1e-2
num_epoch = 50

# model setting
model = simple_ae(len(channel_head))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# shuffle dataset
# for train, test in ShuffleSplit(n_splits=2, test_size=0.1).split(data_idx):
#     print(train.shape)

for epoch in range(num_epoch):
    for data in dataloader:
        inp = data[1]
        
        output = model(inp)
        loss = criterion(output, inp)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # log
    print('epoch [{}/{}], loss:{:.4f}' 
            .format(epoch + 1, num_epoch, loss.data.item()))


torch.save(model.state_dict(), f'{model_path}/ae_on_{data_class}.pth')
