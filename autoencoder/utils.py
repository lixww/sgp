import torch
from torch.utils.data import Dataset

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