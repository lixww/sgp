import pandas as pd

import torch
from torch.utils.data import Dataset, random_split

from sklearn.preprocessing import normalize
from sklearn.metrics import precision_score

import matplotlib.pyplot as plt




class FolioDataset(Dataset):
    def __init__(self, location, channel, grdtruth, 
                 location_head=None, channel_head=None):
        # normalize
        self.location = torch.FloatTensor(location)
        self.channel = torch.FloatTensor(normalize(channel, axis=0, norm='max'))
        self.grdtruth = torch.LongTensor(grdtruth)

        self.location_head = location_head
        self.channel_head = channel_head

    def __getitem__(self, index):
        location = self.location[index]
        channel = self.channel[index]
        grdtruth = self.grdtruth[index]
        return location, channel, grdtruth

    def __len__(self):
        return len(self.grdtruth)


def load_labeled_dataset(folder_path='autoencoder/data/sgp'):
    data_path = f'{folder_path}/training_file_8_bit.csv'
    training_file = pd.read_csv(data_path)

    location_head = training_file.columns[2:4]
    channel_head = training_file.columns[4:]

    y_true = training_file['class_name'].to_numpy()
    location = training_file[location_head].to_numpy()
    channel = training_file[channel_head].to_numpy()

    data_idx = training_file.index

    channel_len = len(channel_head)

    # load data
    full_dataset = FolioDataset(location, channel, y_true,
                                location_head=location_head,
                                channel_head=channel_head)

    return full_dataset, channel_len, data_idx


def split_dataset(full_dataset:Dataset, split_ratio=0.9):
    # split into train & develop_set
    train_size = int(split_ratio * len(full_dataset))
    dev_size = len(full_dataset) - train_size
    train_dataset, dev_dataset = random_split(full_dataset, [train_size, dev_size])

    return train_dataset, dev_dataset


def load_raw_labeled_data(folder_path='autoencoder/data/sgp'):
    data_path = f'{folder_path}/training_file_8_bit.csv'
    training_file = pd.read_csv(data_path)

    channel_head = training_file.columns[4:]

    y_true = training_file['class_name'].to_numpy()
    channel = training_file[channel_head].to_numpy()

    channel_len = len(channel_head)

    return channel, y_true, channel_len


def cal_accuracy_given_pred(prediction, truth):
    precision = precision_score(truth, prediction, average='micro')

    return precision



def plot_roc(fpr:list, tpr:list, roc_area:list):
    plt.figure()
    lw = 2
    colors = ['darkorange', 'green']
    notes = ['train', 'validat']
    for i in range(len(colors)):
        f = fpr[i]
        t = tpr[i]
        auc = roc_area[i]
        c = colors[i]
        n = notes[i]
        plt.plot(f, t, color=c,
            lw=lw, label=n+' - ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()