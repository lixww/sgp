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

from pathlib import Path

import models
from utils import FolioDataset, load_raw_images_data
from utils import reconstruct_image, get_sample_image


# test nd[normal/f/hyb]_conv: enhance undertext in the given input folio image & save the reconstruction
# n: [2/3]

def enhanced_roi(data_id):
    data_class = 'allClass'
    folio_ids = ['024r_029v', '102v_107r', '214v_221r']
    model_data_id = folio_ids[0]
    data_type = 'cropped_roi'
    conv_nd = 2
    # net_style {'normal': 0, 'fconv': 1, 'hybrid': 2}
    net_style = 2


    # file paths
    data_path = f'networks/data/sgp/{data_id}/cropped_roi/*'
    model_path = 'networks/model'
    img_save_path = 'networks/reconstructed_roi'
    # mkdir if not exists
    Path(f'{img_save_path}').mkdir(parents=True, exist_ok=True)


    # load images
    print('-load images..')

    imgs_norm = load_raw_images_data(data_path, rescale_ratio=0.25)
    sample_img = get_sample_image(data_path, rescale_ratio=0.25)
    img_height, img_width = sample_img.shape


    channel_len = 23


    # conv model
    if net_style == 2:
        if conv_nd == 2:
            conv_model = models.conv2d_hyb_net(channel_len, img_width, img_height, 3)
        elif conv_nd == 3:
            conv_model = models.conv3d_hyb_net(channel_len, img_width, img_height, 3)
        model_name = f'{model_path}/conv{conv_nd}d_hyb_on_{data_class}_{model_data_id}.pth'
    elif net_style == 1:
        if conv_nd == 2:
            conv_model = models.fconv2d_net(channel_len, img_width, img_height, 3)
        elif conv_nd == 3:
            conv_model = models.fconv3d_net(channel_len, img_width, img_height, 3)
        model_name = f'{model_path}/fconv{conv_nd}d_on_{data_class}_{model_data_id}.pth'
    elif net_style == 0:
        if conv_nd == 2:
            conv_model = models.conv2d_net(channel_len, img_width, img_height, 3)
        elif conv_nd == 3:
            conv_model = models.conv3d_net(channel_len, img_width, img_height, 3)
        model_name = f'{model_path}/conv{conv_nd}d_on_{data_class}_{model_data_id}.pth'
    conv_model.load_state_dict(torch.load(model_name, map_location='cpu'))
    conv_model.eval()



    print('Reconstruct..')
    with torch.no_grad():
        output = conv_model(torch.FloatTensor(imgs_norm))
        _, conv_pred = torch.max(output.data, 1)

        print('-max in conv_pred: ', torch.max(conv_pred.data).item())
        print('-min in conv_pred: ', torch.min(conv_pred.data).item())


    imsave(f'{img_save_path}/{data_id}_orig_eval.png', sample_img)

    sample_img_conv = reconstruct_image(sample_img, conv_pred, count_note=True)
    if net_style == 2:
        img_name = f'{img_save_path}/conv{conv_nd}d_hyb/{data_id}_conv{conv_nd}d_hyb_eval_model_{model_data_id}.png'
    elif net_style == 1:
        img_name = f'{img_save_path}/fconv{conv_nd}d/{data_id}_fconv{conv_nd}d_eval_model_{model_data_id}.png'
    elif net_style == 0:
        img_name = f'{img_save_path}/conv{conv_nd}d/{data_id}_conv{conv_nd}d_eval_model_{model_data_id}.png'
    imsave(img_name, sample_img_conv)


folio_ids = ['024r_029v', '102v_107r', '214v_221r']

for data_id in folio_ids:
    enhanced_roi(data_id)