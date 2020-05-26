import pandas as pd

from sklearn.metrics import precision_score

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, random_split

import models 
from utils import FolioDataset, plot_roc



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

channel_len = len(channel_head)


# load data
full_dataset = FolioDataset(location, channel, y_true)
# split into train & develop_set
train_size = int(0.9 * len(full_dataset))
dev_size = len(full_dataset) - train_size
train_dataset, dev_dataset = random_split(full_dataset, [train_size, dev_size])


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