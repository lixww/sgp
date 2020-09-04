import pandas as pd

from sklearn.metrics import precision_score

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, random_split

import models 
from utils import FolioDataset, plot_roc
from utils import load_labeled_dataset, split_dataset
from utils import initialize_log_dataframe, plot_loss_acc_one_model

import time


data_class = 'allClass'

# file paths
model_path = 'networks/model'
log_path = 'networks/training_log/conv1d'


# prepare training set
full_dataset, channel_len, _ = load_labeled_dataset()

# hyperparameter
learning_rate = 1e-2
num_epoch = 60
cv_round = 1


model = models.conv1d_net(channel_len, 3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
model.train()

# log loss
log_df = initialize_log_dataframe()
# log time
start_time = time.time()

for cv in range(cv_round):
    print("--- cv round: %s ---" % cv)
    # split into train & develop_set
    train_dataset, dev_dataset = split_dataset(full_dataset)
    dataloader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
    for epoch in range(num_epoch):
        for data in dataloader:
            inp = data[1]
            target = data[2]
            output = model(inp)
            _, conv_pred = torch.max(output.data, 1)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # log
        # acc = precision_score(target, conv_pred, average='micro')
        acc = models.cal_accuracy(dev_dataset, model)
        print('epoch [{}/{}], loss:{:.4f}, accuracy:{:.4f}' \
            .format(epoch + 1, num_epoch, loss.data.item(), acc))
        log_df.loc[epoch + cv*num_epoch] = [epoch+1, loss.data.item(), acc]

# time log
print("--- %s seconds ---" % (time.time() - start_time))

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
# save log df
log_df.to_pickle(f'{log_path}/conv1d_loss_acc_log_sgd.pkl')

# plot loss & acc
plot_loss_acc_one_model(log_df)