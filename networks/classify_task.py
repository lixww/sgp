import pandas as pd

import time

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
from utils import FolioDataset, cal_accuracy_given_pred, plot_roc
from utils import load_labeled_dataset, split_dataset
from utils import load_images_data


# train sae on training_file_8_bit.csv

data_class = 'allClass'
folio_ids = ['024r_029v', '102v_107r', '214v_221r']
data_id = folio_ids[1]

# file paths
model_path = 'networks/model'
pre_train_data_path = f'networks/data/sgp/{data_id}/cropped_roi/*'


# prepare training set
full_dataset, channel_len, _ = load_labeled_dataset()

# split into train & develop_set
train_dataset, dev_dataset = split_dataset(full_dataset)

# training set for pre-train
pre_train_dataset, sample_img = load_images_data(pre_train_data_path, rescale_ratio=0.25)


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

# log time
start_time = time.time()

autoencoder = models.sdae(
    dimensions=[channel_len, 10, 10, 20, 3]
)
print('Pretraining..')
models.pretrain(
    dataset=pre_train_dataset,
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
# time log
print("--- %s seconds ---" % (time.time() - start_time))

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

print('Network')
print_acc(model, train_dataset, print_note='train')
print_acc(model, dev_dataset, print_note='validat')
# plot roc
fpr_train, tpr_train, rocauc_train = models.get_roc(train_dataset, model)
fpr_dev, tpr_dev, rocauc_dev = models.get_roc(dev_dataset, model)
plot_roc([fpr_train, fpr_dev], [tpr_train, tpr_dev], [rocauc_train, rocauc_dev])

# save model
torch.save(model.state_dict(), f'{model_path}/ae_on_{data_class}.pth')