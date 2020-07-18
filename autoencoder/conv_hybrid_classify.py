import pandas as pd

from sklearn.metrics import precision_score

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, random_split

import models 
from utils import plot_roc
from utils import load_patch_dataset, split_dataset



data_class = 'allClass'

# file paths
model_path = 'autoencoder/model'
data_path = 'autoencoder/data/sgp/folio_8_bit_extended_3x3.csv'


# prepare training set
full_dataset, channel_len = load_patch_dataset(data_path=data_path)

# split into train & develop_set
train_dataset, dev_dataset = split_dataset(full_dataset)


# hyperparameter
learning_rate = 1e-2
num_epoch = 500
batch_size = 16


model = models.conv_hybrid(channel_len, 3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
model.train()

dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epoch):
    for data in dataloader:
        inp = data[1]
        target = data[2]
        output = model(inp)

        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # log
    acc = models.cal_accuracy(train_dataset, model)
    print('epoch [{}/{}], loss:{:.4f}, accuracy:{:.4f}' 
        .format(epoch + 1, num_epoch, loss.data.item(), acc))

def print_acc(model, dataset, print_note=''):
    acc = models.cal_accuracy(dataset, model)
    print(print_note, 'accuracy: ', acc)

print_acc(model, train_dataset, print_note='train')
print_acc(model, dev_dataset, print_note='validat')

# save model
torch.save(model.state_dict(), f'{model_path}/conv_hybrid_on_{data_class}.pth')