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

img_width, img_height = (699, 684)

# file paths
data_path = f'autoencoder/data/sgp/{data_id}/cropped_roi/*'
model_path = 'autoencoder/model'
img_save_path = 'autoencoder/reconstructed_roi'

channel_len = 23
pxl_num = img_width * img_height


# load test data
print('Load test data..')

test_dataset, sample_img = load_images_data(data_path)
img_height, img_width = sample_img.shape

# load model
print('Load model..')
autoencoder = models.sdae(
    dimensions=[channel_len, 10, 10, 20, 3]
)
model = models.sdae_lr(autoencoder)
model.load_state_dict(torch.load(f'{model_path}/ae_on_{data_class}.pth', map_location='cpu'))
model.eval()


print('Model predict..')
predictions = models.predict_class(test_dataset, model)


print('Reconstruct..')
sample_img = reconstruct_image(sample_img, predictions)
imsave(f'{img_save_path}/{data_id}.png', sample_img)