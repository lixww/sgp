import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imsave

import torch
from torch.utils.data import DataLoader
from torch import nn

import models
from utils import PatchDataset, load_raw_images_data, reconstruct_image
from utils import load_patch_dataset_from_imgs, pad_prediction
from utils import get_sample_image
from utils import plot_loss_acc_one_model



data_class = 'allClass'
folio_ids = ['024r_029v', '102v_107r', '214v_221r']
data_id = folio_ids[0]
data_type = 'cropped_roi'


# file paths
data_path = f'autoencoder/data/sgp/{data_id}/cropped_roi/*'
model_path = 'autoencoder/model'
log_path = 'autoencoder/training_log'

channel_len = 23

# compare two models
def plot_two_models():
    # load model
    print('Load models..')
    model1 = models.conv_hybrid(channel_len, 3)
    model2 = models.conv_hybrid(channel_len, 3)

    model1.load_state_dict(torch.load(f'{model_path}/conv_hybrid_on_{data_class}.pth', map_location='cpu'))
    model2.load_state_dict(torch.load(f'{model_path}/conv_hybrid_no_res_on_{data_class}.pth', map_location='cpu'))

    model1.eval()
    model2.eval()

    # load llog dataframe
    log_df1 = pd.read_pickle(f'{log_path}/conv_hybrid_loss_acc_log.pkl')
    log_df2 = pd.read_pickle(f'{log_path}/conv_hybrid_no_res_loss_acc_log.pkl')

    ax = log_df1.plot(x='epoch', y='loss', label='w_res')
    ax.locator_params(integer=True)
    log_df2.plot(x='epoch', y='loss', ax=ax, label='wo_res', alpha=0.7)
    plt.ylim([0.0, 1.3])
    plt.show()


# plot one model
def plot_one_model():
    model_data_id = folio_ids[2]
    conv_nd = 3
    # net_style {'normal': 0, 'tconv': 1, 'hybrid': 2}
    net_style = 1
    # get image width & height
    sample_img = get_sample_image(data_path, rescale_ratio=0.25)
    img_height, img_width = sample_img.shape

    # load model
    if net_style == 2:
        if conv_nd == 3:
            conv_model = models.conv3d_hyb_net(channel_len, img_width, img_height, 3)
        model_name = f'conv{conv_nd}d_hyb_on_{data_class}_{model_data_id}.pth'
    elif net_style == 1:
        if conv_nd == 2:
            conv_model = models.fconv2d_net(channel_len, img_width, img_height, 3)
        elif conv_nd == 3:
            conv_model = models.fconv3d_net(channel_len, img_width, img_height, 3)
        model_name = f'fconv{conv_nd}d_on_{data_class}_{model_data_id}.pth'
    elif net_style == 0:
        if conv_nd == 2:
            conv_model = models.conv2d_net(channel_len, img_width, img_height, 3)
        elif conv_nd == 3:
            conv_model = models.conv3d_net(channel_len, img_width, img_height, 3)
        model_name = f'conv{conv_nd}d_on_{data_class}_{model_data_id}.pth'
    conv_model.load_state_dict(torch.load(f'{model_path}/{model_name}', map_location='cpu'))
    conv_model.eval()

    # load log dataframe
    log_df = pd.read_pickle(f'{log_path}/{model_name}_loss_acc_log.pkl')

    # plot
    plot_loss_acc_one_model(log_df)

plot_one_model()