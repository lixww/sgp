import pandas as pd
import numpy as np

from skimage.io import imsave

import torch
from torch.utils.data import DataLoader
from torch import nn

import models
from utils import PatchDataset, load_raw_images_data, reconstruct_image
from utils import load_patch_dataset_from_imgs, pad_prediction



data_class = 'allClass'
data_id = '102v_107r'
data_type = 'cropped_roi'


# file paths
data_path = f'autoencoder/data/sgp/{data_id}/cropped_roi/*'
model_path = 'autoencoder/model'
img_save_path = 'autoencoder/reconstructed_roi/conv_patch'


# load test data
print('Load test data..')
test_imgs = load_raw_images_data(data_path, rescale_ratio=0.25, preserve_range_after_rescale=True)
sample_img = test_imgs[0]
test_dataset, channel_len = load_patch_dataset_from_imgs(test_imgs)


# load model
print('Load model..')
model = models.conv_on_patch(channel_len, 3)
model.load_state_dict(torch.load(f'{model_path}/conv_patch_on_{data_class}.pth', map_location='cpu'))
model.eval()

print('Model predict..')
predictions = models.predict_class(test_dataset, model)

predictions = pad_prediction(predictions, sample_img, test_dataset.patch_size)


print('Reconstruct..')
sample_img = reconstruct_image(sample_img, predictions, enhance_intensity=20, count_note=True)
imsave(f'{img_save_path}/{data_id}_conv_patch.png', sample_img)