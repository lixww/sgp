import pandas as pd
import numpy as np

from skimage.io import imsave

import torch
from torch.utils.data import DataLoader
from torch import nn

from pathlib import Path

import models
from utils import FolioDataset, load_images_data, reconstruct_image


# test sae: enhance undertext in the given input folio image & save the reconstruction

data_class = 'allClass'
folio_ids = ['024r_029v', '102v_107r', '214v_221r']
data_id = folio_ids[0]
data_type = 'cropped_roi'


# file paths
data_path = f'networks/data/sgp/{data_id}/cropped_roi/*'
model_path = 'networks/model'
img_save_path = 'networks/reconstructed_roi/ae'
# mkdir if not exists
Path(f'{img_save_path}').mkdir(parents=True, exist_ok=True)

channel_len = 23


# load test data
print('Load test data..')

test_dataset, sample_img = load_images_data(data_path, rescale_ratio=0.25)
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
imsave(f'{img_save_path}/{data_id}_2trainset_moreEpochs.png', sample_img)