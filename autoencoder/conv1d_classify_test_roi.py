import pandas as pd
import numpy as np

from skimage.io import imsave

import torch
from torch.utils.data import DataLoader
from torch import nn

import models
from utils import FolioDataset, load_images_data, reconstruct_image



data_class = 'allClass'
data_id = '102v_107r'
data_type = 'cropped_roi'


# file paths
data_path = f'autoencoder/data/sgp/{data_id}/cropped_roi/*'
model_path = 'autoencoder/model'
img_save_path = 'autoencoder/reconstructed_roi'

channel_len = 23


# load test data
print('Load test data..')
test_dataset, sample_img = load_images_data(data_path, rescale_ratio=0.25)
img_height, img_width = sample_img.shape



# load model
print('Load model..')
model = models.conv1d_net(channel_len, 3)
model.load_state_dict(torch.load(f'{model_path}/conv1d_on_{data_class}.pth', map_location='cpu'))
model.eval()


print('Model predict..')
predictions = models.predict_class(test_dataset, model)


print('Reconstruct..')
sample_img = reconstruct_image(sample_img, predictions, enhance_intensity=20)
imsave(f'{img_save_path}/{data_id}_conv1d.png', sample_img)