import pandas as pd

from sklearn.metrics import precision_score

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, random_split

import models 
from utils import plot_roc
from utils import load_patch_dataset, split_dataset
from utils import initialize_log_dataframe

import time

from pathlib import Path


# train conv_hybrid model on folio_8_bit_extended_3x3.csv

data_class = 'allClass'

# file paths
model_path = 'networks/model'
data_path = 'networks/data/sgp/folio_8_bit_extended_3x3.csv'
log_path = 'networks/training_log'
# mkdir if not exists
Path(f'{model_path}').mkdir(parents=True, exist_ok=True)
Path(f'{log_path}').mkdir(parents=True, exist_ok=True)

is_consider_residual = True


# prepare training set
full_dataset, channel_len = load_patch_dataset(data_path=data_path)

# split into train & develop_set
train_dataset, dev_dataset = split_dataset(full_dataset)


# hyperparameter
learning_rate = 1e-2
num_epoch = 500
batch_size = 256


model = models.conv_hybrid(channel_len, 3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
model.train()

dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# log loss & acc in dataframe
log_df = initialize_log_dataframe()
# log time
start_time = time.time()

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
    acc = models.cal_accuracy(dev_dataset, model)
    print('epoch [{}/{}], loss:{:.4f}, accuracy:{:.4f}' 
        .format(epoch + 1, num_epoch, loss.data.item(), acc))
    log_df.loc[epoch] = [epoch + 1, loss.data.item(), acc]

# time log
print("--- %s seconds ---" % (time.time() - start_time))
# save log df
if is_consider_residual:
    log_df.to_pickle(f'{log_path}/conv_hybrid_loss_acc_log.pkl')
else:
    log_df.to_pickle(f'{log_path}/conv_hybrid_no_res_loss_acc_log.pkl')

def print_acc(model, dataset, print_note=''):
    acc = models.cal_accuracy(dataset, model)
    print(print_note, 'accuracy: ', acc)

print_acc(model, train_dataset, print_note='train')
print_acc(model, dev_dataset, print_note='validat')

# save model
if is_consider_residual:
    torch.save(model.state_dict(), f'{model_path}/conv_hybrid_on_{data_class}.pth')
else:
    torch.save(model.state_dict(), f'{model_path}/conv_hybrid_no_res_on_{data_class}.pth')