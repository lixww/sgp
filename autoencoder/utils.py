import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import normalize



class FolioDataset(Dataset):
    def __init__(self, location, channel, grdtruth):
        # normalize
        self.location = torch.FloatTensor(location)
        self.channel = torch.FloatTensor(channel)
        self.grdtruth = torch.FloatTensor(grdtruth)

    def __getitem__(self, index):
        location = self.location[index]
        channel = self.channel[index]
        grdtruth = self.grdtruth[index]
        return location, channel, grdtruth

    def __len__(self):
        return len(self.grdtruth)