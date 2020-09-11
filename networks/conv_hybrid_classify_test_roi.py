import pandas as pd
import numpy as np

from skimage.io import imsave

import torch
from torch.utils.data import DataLoader
from torch import nn

from pathlib import Path

import models
from utils import PatchDataset, load_raw_images_data, reconstruct_image
from utils import load_patch_dataset_from_imgs, pad_prediction


# test conv_hybrid: enhance undertext in the given input folio image & save the reconstruction

data_class = 'allClass'
folio_ids = ['024r_029v', '102v_107r', '214v_221r']
data_id = folio_ids[2]
data_type = 'cropped_roi'
is_consider_residual = True


# file paths
data_path = f'networks/data/sgp/{data_id}/cropped_roi/*'
model_path = 'networks/model'
img_save_path = 'networks/reconstructed_roi/conv_hybrid'
# mkdir if not exists
Path(f'{img_save_path}').mkdir(parents=True, exist_ok=True)


# load test data
print('Load test data..')
test_imgs = load_raw_images_data(data_path, rescale_ratio=0.25, preserve_range_after_rescale=True)
sample_img = test_imgs[0]
test_dataset, channel_len = load_patch_dataset_from_imgs(test_imgs, patch_size=3)


# load model
print('Load model..')
model = models.conv_hybrid(channel_len, 3)
if is_consider_residual:
    model.load_state_dict(torch.load(f'{model_path}/conv_hybrid_on_{data_class}.pth', map_location='cpu'))
else:
    model.load_state_dict(torch.load(f'{model_path}/conv_hybrid_no_res_on_{data_class}.pth', map_location='cpu'))
model.eval()

print('Model predict..')
predictions = models.predict_class(test_dataset, model)

predictions = pad_prediction(predictions, sample_img, test_dataset.patch_size)


print('Reconstruct..')
sample_img = reconstruct_image(sample_img, predictions, enhance_intensity=20, count_note=True)
if is_consider_residual:
    imsave(f'{img_save_path}/{data_id}_conv_hybrid.png', sample_img)
else:
    imsave(f'{img_save_path}/{data_id}_conv_hybrid_no_res.png', sample_img)