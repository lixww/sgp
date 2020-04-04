import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import normalize
from sklearn.metrics import precision_score



class FolioDataset(Dataset):
    def __init__(self, location, channel, grdtruth, 
                 location_head=None, channel_head=None):
        # normalize
        self.location = torch.FloatTensor(normalize(location, axis=0, norm='max'))
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