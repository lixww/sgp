import pandas as pd

from sklearn.metrics import precision_score

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, random_split

import models 
from utils import FolioDataset, plot_roc
from utils import load_labeled_dataset, split_dataset



data_class = 'allClass'

# file paths
model_path = 'autoencoder/model'


# prepare training set

full_dataset, channel_len, _ = load_labeled_dataset()

# split into train & develop_set
train_dataset, dev_dataset = split_dataset(full_dataset)


# hyperparameter
learning_rate = 1e-2
num_epoch = 50


model = models.conv1d_net(channel_len, 3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
model.train()

dataloader = DataLoader(train_dataset, batch_size=1000, shuffle=True)

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
    print('epoch [{}/{}], loss:{:.4f}' 
        .format(epoch + 1, num_epoch, loss.data.item()))

def print_acc(model, dataset, print_note=''):
    acc = models.cal_accuracy(dataset, model)
    print(print_note, 'accuracy: ', acc)

print_acc(model, train_dataset, print_note='train')
print_acc(model, dev_dataset, print_note='validat')
# plot roc
fpr_train, tpr_train, rocauc_train = models.get_roc(train_dataset, model)
fpr_dev, tpr_dev, rocauc_dev = models.get_roc(dev_dataset, model)
plot_roc([fpr_train, fpr_dev], [tpr_train, tpr_dev], [rocauc_train, rocauc_dev])

# save model
torch.save(model.state_dict(), f'{model_path}/conv1d_on_{data_class}.pth')