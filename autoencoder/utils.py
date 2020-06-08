import pandas as pd
import math
import numpy as np

import torch
from torch.utils.data import Dataset, random_split

from skimage.io import imread_collection
from skimage.io import imread
from skimage.transform import rescale

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

class PatchDataset(Dataset):
    def __init__(self, location, channel, grdtruth):
        ''' patch: [n,neighbor_num,c] 
            patch_num: n '''
        self.location = location
        self.patch = []
        for p in channel:
            self.patch.append(torch.FloatTensor(normalize(p, axis=1, norm='max')))
        self.grdtruth = torch.LongTensor(grdtruth)

        self.patch_num = len(grdtruth)
        self.patch_size = int(math.sqrt(len(channel[0])))

    def __getitem__(self, index):
        return self.location[index], self.patch[index], self.grdtruth[index]

    def __len__(self):
        return self.patch_num


def load_patch_dataset(data_path='autoencoder/data/sgp/folio_8_bit_extended_5x5.csv'):
    training_file = pd.read_csv(data_path)

    location_head = training_file.columns[2:4]
    channel_head = training_file.columns[4:-1]
    channel_len = len(channel_head)

    patch_id = training_file['center_pxl_id'].unique()

    locations = []
    patches = []
    y_true = []
    for p_id in patch_id:
        pxls_index = training_file.loc[training_file['center_pxl_id']==p_id].index
        location = training_file.loc[pxls_index][location_head].to_numpy()
        patch = training_file.loc[pxls_index][channel_head].to_numpy()
        label = training_file.loc[pxls_index[0]]['class_name']

        locations.append(location)
        patches.append(patch)
        y_true.append(label)

    # load data
    full_dataset = PatchDataset(locations, patches, y_true)

    return full_dataset, channel_len


def load_patch_dataset_from_imgs(imgs_data:list, patch_size=5):
    ''' imgs_data: [c,h,w] 
        patch_size: odd'''
    locations = []
    patches = []
    y_true = []

    channel_len = len(imgs_data)
    img_height, img_width = imgs_data[0].shape

    radius = int((patch_size-1)*0.5)
    center_pxl_count = 0
    for h in range(radius, img_height-radius):
        for w in range(radius, img_width-radius):
            center_pxl_count += 1
            patch = []
            # find neighbors
            neighbors_loc = get_window(radius, x=w, y=h)
            for (neigh_x, neigh_y) in neighbors_loc:
                channel_data = []
                for c in range(channel_len):
                    channel_data.append(imgs_data[c][neigh_y][neigh_x])
                # add one neighbor
                patch.append(channel_data)
            # add one patch
            locations.append(neighbors_loc)
            patches.append(patch)
    y_true = [-1] * center_pxl_count

    # load data
    full_dataset = PatchDataset(locations, patches, y_true)

    return full_dataset, channel_len


def load_labeled_dataset(data_path='autoencoder/data/sgp/training_file_8_bit.csv'):
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


def load_raw_labeled_data(data_path='autoencoder/data/sgp/training_file_8_bit.csv'):
    training_file = pd.read_csv(data_path)

    channel_head = training_file.columns[4:]

    y_true = training_file['class_name'].to_numpy()
    channel = training_file[channel_head].to_numpy()

    channel_len = len(channel_head)

    return channel, y_true, channel_len


def load_images_data(data_path, rescale_ratio=1):
    ic = imread_collection(data_path)
    imgs = []
    for f in ic.files:
        image = imread(f, as_gray=True)
        if rescale_ratio != 1:
            image = rescale(image, rescale_ratio, anti_aliasing=False, preserve_range=True)
        imgs.append(image)    

    sample_img = imgs[0]
    img_height, img_width = imgs[0].shape
    pxl_num = img_height * img_width
    channel_len = len(imgs)

    # prepare dataset
    channel = []
    location = []
    y_true = [-1] * pxl_num
    for h in range(img_height):
        for w in range(img_width):
            data = []
            for i in range(channel_len):
                data.append(imgs[i][h][w])
            channel.append(data)
            location.append((w+1, h+1))

    test_dataset = FolioDataset(location, channel, y_true)

    return test_dataset, sample_img


def load_raw_images_data(data_path, rescale_ratio=1, preserve_range_after_rescale=False):
    ic = imread_collection(data_path)
    imgs = []
    for f in ic.files:
        image = imread(f, as_gray=True)
        if rescale_ratio != 1:
            image = rescale(image, rescale_ratio, 
                            anti_aliasing=False, 
                            preserve_range=preserve_range_after_rescale)
        imgs.append(image)  

    return imgs


def get_sample_image(data_path, img_idx=0, rescale_ratio=1, preserve_range_after_rescale=True):
    ic = imread_collection(data_path)
    image = imread(ic.files[img_idx], as_gray=True)
    if rescale_ratio != 1:
        image = rescale(image, rescale_ratio,
                        anti_aliasing=False,
                        preserve_range=preserve_range_after_rescale)
                        
    return image


def flatten_images(imgs):
    img_height, img_width = imgs[0].shape
    channel_len = len(imgs)
    intensities = []
    locations = []
    for h in range(img_height):
        for w in range(img_width):
            data = []
            for i in range(channel_len):
                data.append(imgs[i][h][w])
            intensities.append(data)
            locations.append((w+1, h+1))

    return intensities, locations


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



def reconstruct_image(image, pxl_labels, enhance_intensity=20, count_note=False):
    img_height, img_width = image.shape
    count_2 = 0
    count_01 = 0
    for h in range(img_height):
        for w in range(img_width):
            index = (img_width)*h + w
            if pxl_labels[index] == 2:
                image[h][w] -= enhance_intensity
                count_2 += 1
            else:
                image[h][w] += enhance_intensity
                count_01 += 1

    if count_note:
        print('count_2 = ', count_2)
        print('count_01 = ', count_01)

    return image


def get_window(radius, x=0, y=0):
    window = []
    for i in range(-radius, radius+1):
        for j in range(-radius, radius+1):
            window.append((j+x,i+y))

    return window


def pad_prediction(preds, sample_img, patch_size):
    img_height, img_width = sample_img.shape
    radius = int((patch_size-1)*0.5)
    cur_height = img_height - (radius*2)
    cur_width = img_width - (radius*2)

    preds = torch.reshape(preds, (cur_height, cur_width)).numpy()
    preds = np.pad(preds, radius, mode='edge')
    preds = torch.reshape(torch.LongTensor(preds), (-1,))

    return preds
