import pandas as pd

from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import precision_score

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, random_split

from torchvision import transforms

import models 
from utils import FolioDataset, cal_accuracy_given_pred



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
# split into train & develop_set
train_size = int(0.9 * len(full_dataset))
dev_size = len(full_dataset) - train_size
train_dataset, dev_dataset = random_split(full_dataset, [train_size, dev_size])


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
pretrain_epoch = 40
finetune_epoch = 60


autoencoder = models.sdae(
    dimensions=[len(channel_head), 10, 10, 20, 3]
)
print('Pretraining..')
models.pretrain(
    dataset=train_dataset,
    autoencoder=autoencoder,
    num_epoch=pretrain_epoch
)
print('Fine Tuning..')
model = models.fine_tune(
    dataset=train_dataset,
    autoencoder=autoencoder,
    num_epoch=finetune_epoch,
    validation=dev_dataset,
    train_encoder_more=True
)
print('SVM')
classifier = svm.SVC(decision_function_shape='ovo', gamma='auto')
model.eval()
features = models.encode(train_dataset, model)
grdtruth = []
with torch.no_grad():
    dataloader = DataLoader(train_dataset, batch_size=1000, shuffle=False)
    for data in dataloader:
        truth = data[2]
        if len(grdtruth) <= 0:
            grdtruth = truth
            continue
        grdtruth = torch.cat((grdtruth, truth))
    grdtruth = grdtruth.numpy()
prediction = classifier.fit(features, grdtruth)
precision_clf = classifier.score(features, grdtruth)
print('accuracy: ', precision_clf)

def print_acc(model, dataset, print_note=''):
    acc = models.cal_accuracy(dataset, model)

    print(print_note, 'accuracy: ', acc)

print_acc(model, train_dataset, print_note='train')
print_acc(model, dev_dataset, print_note='validat')

# save model
torch.save(model.state_dict(), f'{model_path}/ae_on_{data_class}.pth')