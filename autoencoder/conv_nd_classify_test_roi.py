import pandas as pd
import numpy as np

from skimage.io import imsave
from skimage.transform import rescale

from sklearn.metrics import precision_score

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

from sklearn.preprocessing import normalize

import models
from utils import FolioDataset, load_raw_images_data
from utils import reconstruct_image, get_sample_image



data_class = 'allClass'
model_data_id = '102v_107r'
data_id = '214v_221r'
data_type = 'cropped_roi'
conv_nd = 3


# file paths
data_path = f'autoencoder/data/sgp/{data_id}/cropped_roi/*'
model_path = 'autoencoder/model'
img_save_path = 'autoencoder/reconstructed_roi'


# load images
print('-load images..')

imgs_norm = load_raw_images_data(data_path, rescale_ratio=0.25)
sample_img = get_sample_image(data_path, rescale_ratio=0.25)
img_height, img_width = sample_img.shape


channel_len = 23


# conv model
if conv_nd == 2:
    conv_model = models.conv2d_net(channel_len, img_width, img_height, 3)
elif conv_nd == 3:
    conv_model = models.conv3d_net(channel_len, img_width, img_height, 3)
conv_model.load_state_dict(torch.load(f'{model_path}/conv{conv_nd}d_on_{data_class}_{model_data_id}.pth', map_location='cpu'))
conv_model.eval()



print('Reconstruct..')
with torch.no_grad():
    output = conv_model(torch.FloatTensor(imgs_norm))
    _, conv_pred = torch.max(output.data, 1)

    print('-max in conv_pred: ', torch.max(conv_pred.data).item())
    print('-min in conv_pred: ', torch.min(conv_pred.data).item())


imsave(f'{img_save_path}/{data_id}_orig_eval.png', sample_img)

sample_img_conv = reconstruct_image(sample_img, conv_pred, count_note=True)
imsave(f'{img_save_path}/{data_id}_conv{conv_nd}d_eval_model_{model_data_id}.png', sample_img_conv)

