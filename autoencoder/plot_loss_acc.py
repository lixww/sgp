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



# file paths
# [conv1d-depth, conv1d-depth-vs-kernel, conv1d-kernel, conv1d-width]
log_path = 'autoencoder/training_log/conv2ds/expandable'

# compare two models
def plot_two_models():
    # load llog dataframe
    log_df1 = pd.read_pickle(f'{log_path}/conv2d_hyb_on_allClass_102v_107r.pth_loss_acc_log-adam-depth4.pkl')
    log_df2 = pd.read_pickle(f'{log_path}/conv2d_hyb_on_allClass_102v_107r.pth_loss_acc_log-sgd-depth4.pkl')

    _, ax = plt.subplots()
    # set colors
    cm = plt.get_cmap('gist_rainbow')
    NUM_COLORS = 4
    ax.set_prop_cycle('color', [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

    ax = log_df1.plot(x='epoch', y=['loss', 'acc'], ax=ax, label=['adam-loss', 'adam-acc'], alpha=0.7)
    ax.locator_params(integer=True)
    log_df2.plot(x='epoch', y=['loss', 'acc'], ax=ax, label=['sgd-loss', 'sgd-acc'], alpha=0.7)
    
    ax.locator_params(integer=True)
    # Put a legend to the right of the current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)

    plt.axhline(y=1, color='grey', alpha=0.8, linestyle='dashdot')
    plt.axhline(y=0.9, color='grey', alpha=0.8, linestyle='dashdot')
    plt.ylim([0.0, 1.3])
    plt.show()


# plot one model
def plot_one_model():
    # load log dataframe
    log_df = pd.read_pickle(f'{log_path}/conv2d_on_allClass_102v_107r.pth_loss_acc_log.pkl')

    # plot
    plot_loss_acc_one_model(log_df)

def plot_one_model_cv(cv=5):
    # load log dataframe
    log_df = pd.read_pickle(f'{log_path}/conv1d_loss_acc_baseline_log.pkl')
    total_epoch, _ = log_df.shape
    each_epoch = int(total_epoch/cv)
    _, ax = plt.subplots()
    loss_cv_mean = np.mean(log_df['loss'].values.reshape(each_epoch, -1), axis=1)
    loss_cv_std = np.std(log_df['loss'].values.reshape(each_epoch, -1), axis=1)
    acc_cv_mean = np.mean(log_df['acc'].values.reshape(each_epoch, -1), axis=1)
    acc_cv_std = np.std(log_df['acc'].values.reshape(each_epoch, -1), axis=1)
    ax.grid()
    ax.plot(log_df.iloc[:each_epoch]['epoch'], loss_cv_mean)
    ax.fill_between(log_df.iloc[:each_epoch]['epoch'], loss_cv_mean - loss_cv_std,
                        loss_cv_mean + loss_cv_std, alpha=0.1)
    ax.plot(log_df.iloc[:each_epoch]['epoch'], acc_cv_mean)
    ax.fill_between(log_df.iloc[:each_epoch]['epoch'], acc_cv_mean - acc_cv_std,
                        acc_cv_mean + acc_cv_std, alpha=0.1)
    # for i in range(cv):
    #     idx_from = 0 + i*each_epoch
    #     idx_to = each_epoch + i*each_epoch
    #     try:
    #         ax
    #     except NameError:
    #         ax = log_df.iloc[idx_from:idx_to].plot(x='epoch', y=['loss', 'acc'])
    #     else:
    #         log_df.iloc[idx_from:idx_to].plot(x='epoch', y=['loss', 'acc'], ax=ax)
    #     ax = df.plot(x='epoch', y=['loss', 'acc_train', 'acc_dev'])
    ax.locator_params(integer=True)
    plt.axhline(y=1, color='red', alpha=0.6, linestyle='dashdot')
    plt.axhline(y=0.9, color='red', alpha=0.6, linestyle='dashdot')
    plt.xlim([1, each_epoch])
    plt.show()


def plot_multiple_model():
    # log list
    # for depth vs. kernel
    # logs = {'conv1d_loss_acc_baseline_log.pkl': 'baseline',
    #         'conv1d_loss_acc_1_5x5_log.pkl': '1-5x5', 
    #         'conv1d_loss_acc_1_7x7_log.pkl': '1-7x7',
    #         'conv1d_loss_acc_2_3x3_log.pkl': '2-3x3',
    #         'conv1d_loss_acc_3_2x2_log.pkl': '3-2x2',
    #         'conv1d_loss_acc_5_1x1_log.pkl': '5-1x1'}
    # for depth
    # logs = {'conv1d_loss_acc_baseline_log.pkl': 'baseline',
    #         'conv1d_loss_acc_depth2_log.pkl': 'depth-2', 
    #         'conv1d_loss_acc_depth3_log.pkl': 'depth-3', 
    #         'conv1d_loss_acc_depth5_log.pkl': 'depth-5', 
    #         'conv1d_loss_acc_depth7_log.pkl': 'depth-7', 
    #         'conv1d_loss_acc_depth9_log.pkl': 'depth-9', }
    # for kernel
    # logs = {'conv1d_loss_acc_baseline_log.pkl': 'baseline',
    #         'conv1d_loss_acc_1x1_log.pkl': '1x1', 
    #         'conv1d_loss_acc_2x2_log.pkl': '2x2',
    #         'conv1d_loss_acc_3x3_log.pkl': '3x3',
    #         'conv1d_loss_acc_5x5_log.pkl': '5x5',
    #         'conv1d_loss_acc_7x7_log.pkl': '7x7'}
    # for width
    logs = {'conv1d_loss_acc_baseline_log.pkl': 'baseline',
            'conv1d_loss_acc_w20_log.pkl': 'w20', 
            'conv1d_loss_acc_w32_log.pkl': 'w32',
            'conv1d_loss_acc_w64_log.pkl': 'w64',
            'conv1d_loss_acc_w128_log.pkl': 'w128',
            'conv1d_loss_acc_w256_log.pkl': 'w256'}

    _, ax = plt.subplots()
    # set colors
    cm = plt.get_cmap('gist_rainbow')
    NUM_COLORS = len(logs)*2
    ax.set_prop_cycle('color', [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

    for log_name in logs.keys():
        # load log dataframe
        log_df = pd.read_pickle(f'{log_path}/{log_name}')
        # try:
        #     ax
        # except NameError:
        #     # ax = log_df.plot(x='epoch', y='loss', label=logs[log_name], alpha=0.7)
        #     ax = log_df.iloc[:101].plot(x='epoch', y=['loss', 'acc'], label=[logs[log_name]+'-loss', logs[log_name]+'-acc'], alpha=0.7)
        #     # ax = log_df.iloc[:51].plot(x='epoch', y='acc', label=logs[log_name], alpha=0.7)
        # else:
        #     # log_df.plot(x='epoch', y='loss', ax=ax, label=logs[log_name], alpha=0.7)
        #     log_df.iloc[:101].plot(x='epoch', y=['loss', 'acc'], ax=ax, label=[logs[log_name]+'-loss', logs[log_name]+'-acc'], alpha=0.7)
        #     # log_df.iloc[:51].plot(x='epoch', y='acc', ax=ax, label=logs[log_name], alpha=0.7)
        
        # log_df.iloc[:101].plot(x='epoch', y=['loss', 'acc'], ax=ax, label=[logs[log_name]+'-loss', logs[log_name]+'-acc'], alpha=0.7)
        log_df.plot(x='epoch', y=['loss', 'acc'], ax=ax, label=[logs[log_name]+'-loss', logs[log_name]+'-acc'], alpha=0.7)
    
    ax.locator_params(integer=True)
    # Put a legend to the right of the current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)

    plt.axhline(y=1, color='grey', alpha=0.8, linestyle='dashdot')
    plt.axhline(y=0.9, color='grey', alpha=0.8, linestyle='dashdot')
    plt.ylim([0.0, 1.3])
    plt.show()

def plot_multiple_model_cv(cv=5):
    # log list
    # for depth vs. kernel
    # logs = {'conv1d_loss_acc_baseline_log.pkl': 'baseline',
    #         'conv1d_loss_acc_1_5x5_log.pkl': '1-5x5', 
    #         'conv1d_loss_acc_1_7x7_log.pkl': '1-7x7',
    #         'conv1d_loss_acc_2_3x3_log.pkl': '2-3x3',
    #         'conv1d_loss_acc_3_2x2_log.pkl': '3-2x2',
    #         'conv1d_loss_acc_5_1x1_log.pkl': '5-1x1'}
    # for depth
    logs = {'conv1d_loss_acc_baseline_log.pkl': 'baseline',
            'conv1d_loss_acc_depth2_log.pkl': 'depth-2', 
            'conv1d_loss_acc_depth3_log.pkl': 'depth-3', 
            'conv1d_loss_acc_depth5_log.pkl': 'depth-5', 
            'conv1d_loss_acc_depth7_log.pkl': 'depth-7', 
            'conv1d_loss_acc_depth9_log.pkl': 'depth-9', }
    # for kernel
    # logs = {'conv1d_loss_acc_baseline_log.pkl': 'baseline',
    #         'conv1d_loss_acc_1x1_log.pkl': '1x1', 
    #         'conv1d_loss_acc_2x2_log.pkl': '2x2',
    #         'conv1d_loss_acc_3x3_log.pkl': '3x3',
    #         'conv1d_loss_acc_5x5_log.pkl': '5x5',
    #         'conv1d_loss_acc_7x7_log.pkl': '7x7'}
    # for width
    # logs = {'conv1d_loss_acc_baseline_log.pkl': 'baseline',
    #         'conv1d_loss_acc_w20_log.pkl': 'w20', 
    #         'conv1d_loss_acc_w32_log.pkl': 'w32',
    #         'conv1d_loss_acc_w64_log.pkl': 'w64',
    #         'conv1d_loss_acc_w128_log.pkl': 'w128',
    #         'conv1d_loss_acc_w256_log.pkl': 'w256'}

    _, ax = plt.subplots()
    # set colors
    cm = plt.get_cmap('gist_rainbow')
    NUM_COLORS = len(logs)*2
    ax.set_prop_cycle('color', [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

    for log_name in logs.keys():
        # load log dataframe
        log_df = pd.read_pickle(f'{log_path}/{log_name}')
        total_epoch, _ = log_df.shape
        each_epoch = int(total_epoch/cv)
        
        loss_cv_mean = np.mean(log_df['loss'].values.reshape(each_epoch, -1), axis=1)
        loss_cv_std = np.std(log_df['loss'].values.reshape(each_epoch, -1), axis=1)
        acc_cv_mean = np.mean(log_df['acc'].values.reshape(each_epoch, -1), axis=1)
        acc_cv_std = np.std(log_df['acc'].values.reshape(each_epoch, -1), axis=1)
        
        ax.plot(log_df.iloc[:each_epoch]['epoch'], loss_cv_mean, label=logs[log_name]+'-loss', alpha=0.7)
        ax.fill_between(log_df.iloc[:each_epoch]['epoch'], loss_cv_mean - loss_cv_std,
                            loss_cv_mean + loss_cv_std, alpha=0.1)
        ax.plot(log_df.iloc[:each_epoch]['epoch'], acc_cv_mean, label=logs[log_name]+'-acc', alpha=0.7)
        ax.fill_between(log_df.iloc[:each_epoch]['epoch'], acc_cv_mean - acc_cv_std,
                            acc_cv_mean + acc_cv_std, alpha=0.1)
    
    ax.grid()
    ax.locator_params(integer=True)
    # Put a legend to the right of the current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)

    plt.axhline(y=1, color='grey', alpha=0.8, linestyle='dashdot')
    plt.axhline(y=0.9, color='grey', alpha=0.8, linestyle='dashdot')
    plt.ylim([0.0, 1.3])
    plt.xlim([1, each_epoch])
    plt.show()

def plot_multiple_model_for_diff_training_data(num_training_set=3):
    # for 2/3dconv
    # logs = [{'conv2ds/conv2d_on_allClass_024r_029v.pth_loss_acc_log.pkl': '2dconv-basic', 
    #         'conv2ds/conv2d_on_allClass_102v_107r.pth_loss_acc_log.pkl': '2dconv-basic', 
    #         'conv2ds/conv2d_on_allClass_214v_221r.pth_loss_acc_log.pkl': '2dconv-basic', },
    #         {'conv2ds/conv2d_hyb_on_allClass_024r_029v.pth_loss_acc_log.pkl': '2dconv-hyb', 
    #         'conv2ds/conv2d_hyb_on_allClass_102v_107r.pth_loss_acc_log.pkl': '2dconv-hyb', 
    #         'conv2ds/conv2d_hyb_on_allClass_214v_221r.pth_loss_acc_log.pkl': '2dconv-hyb', },
    #         {'conv3ds/conv3d_on_allClass_024r_029v.pth_loss_acc_log.pkl': '3dconv-basic', 
    #         'conv3ds/conv3d_on_allClass_102v_107r.pth_loss_acc_log.pkl': '3dconv-basic', 
    #         'conv3ds/conv3d_on_allClass_214v_221r.pth_loss_acc_log.pkl': '3dconv-basic', },
    #         {'conv3ds/conv3d_hyb_on_allClass_024r_029v.pth_loss_acc_log.pkl': '3dconv-hyb', 
    #         'conv3ds/conv3d_hyb_on_allClass_102v_107r.pth_loss_acc_log.pkl': '3dconv-hyb', 
    #         'conv3ds/conv3d_hyb_on_allClass_214v_221r.pth_loss_acc_log.pkl': '3dconv-hyb', },]
    # for 2dconv_optimizer
    logs = [{'conv2ds/conv2d_on_allClass_024r_029v.pth_loss_acc_log.pkl': '2dconv-adam', 
            'conv2ds/conv2d_on_allClass_102v_107r.pth_loss_acc_log.pkl': '2dconv-adam', 
            'conv2ds/conv2d_on_allClass_214v_221r.pth_loss_acc_log.pkl': '2dconv-adam', },
            {'conv2ds/sgd/conv2d_on_allClass_024r_029v.pth_loss_acc_log.pkl': '2dconv-sgd', 
            'conv2ds/sgd/conv2d_on_allClass_102v_107r.pth_loss_acc_log.pkl': '2dconv-sgd', 
            'conv2ds/sgd/conv2d_on_allClass_214v_221r.pth_loss_acc_log.pkl': '2dconv-sgd', },]

    _, ax = plt.subplots()
    # set colors
    cm = plt.get_cmap('gist_rainbow')
    NUM_COLORS = len(logs)*2
    ax.set_prop_cycle('color', [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

    for log_set in logs:
        # load log dataframe
        sub_dfs = []
        for log_name in log_set.keys():
            sub_log_df = pd.read_pickle(f'{log_path}/{log_name}')
            sub_dfs.append(sub_log_df)
        log_df = pd.concat(sub_dfs)
        total_epoch, _ = log_df.shape
        each_epoch = int(total_epoch/num_training_set)
    
        loss_mean = np.mean(log_df['loss'].values.reshape(-1,each_epoch), axis=0)
        loss_std = np.std(log_df['loss'].values.reshape(-1,each_epoch), axis=0)
        acc_mean = np.mean(log_df['acc'].values.reshape(-1,each_epoch), axis=0)
        acc_std = np.std(log_df['acc'].values.reshape(-1,each_epoch), axis=0)

        ax.plot(log_df.iloc[:each_epoch]['epoch'], loss_mean, label=log_set[log_name]+'-loss', alpha=0.7)
        ax.fill_between(log_df.iloc[:each_epoch]['epoch'], loss_mean - loss_std,
                            loss_mean + loss_std, alpha=0.1)
        ax.plot(log_df.iloc[:each_epoch]['epoch'], acc_mean, label=log_set[log_name]+'-acc', alpha=0.7)
        ax.fill_between(log_df.iloc[:each_epoch]['epoch'], acc_mean - acc_std,
                            acc_mean + acc_std, alpha=0.1)
    
    ax.grid()
    ax.locator_params(integer=True)
    # Put a legend to the right of the current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)

    plt.axhline(y=1, color='grey', alpha=0.8, linestyle='dashdot')
    plt.axhline(y=0.9, color='grey', alpha=0.8, linestyle='dashdot')
    plt.ylim([0.0, 1.3])
    plt.xlim([1, 301])
    plt.show()

plot_two_models()
