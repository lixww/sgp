import pandas as pd
import numpy as np

from skimage.io import imread_collection
from skimage.io import imread, imsave

import torch
from torch.utils.data import DataLoader
from torch import nn

import models
from utils import FolioDataset



data_class = 'allClass'
data_id = '214v_221r'
data_type = 'cropped_roi'

img_width, img_height = (699, 684)

# file paths
data_path = f'autoencoder/data/sgp/{data_id}/cropped_roi/*'
model_path = 'autoencoder/model'
img_save_path = 'autoencoder/reconstructed_roi'

channel_len = 23
pxl_num = img_width * img_height


# load test data
print('Load test data..')
ic = imread_collection(data_path)
imgs = []
for f in ic.files:
    imgs.append(imread(f, as_gray=True))


# load model
print('Load model..')
autoencoder = models.sdae(
    dimensions=[channel_len, 10, 10, 20, 3]
)
model = models.sdae_lr(autoencoder)
model.load_state_dict(torch.load(f'{model_path}/ae_on_{data_class}.pth', map_location='cpu'))
model.eval()


print('Prepare dataset..')
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

print('Model predict..')
predictions = models.predict_class(test_dataset, model)


print('Reconstruct..')
sample_img = imgs[0]
for h in range(img_height):
    for w in range(img_width):
        index = (img_width)*h + w
        if predictions[index] == 2:
            sample_img[h][w] -= 20
        else:
            sample_img[h][w] += 20
imsave(f'{img_save_path}/{data_id}.tif', sample_img)