import matplotlib.pyplot as plt

import torch

from torchvision.utils import make_grid

import models
from utils import load_patch_dataset_from_imgs, load_raw_images_data
from utils import get_sample_image


data_class = 'allClass'
model_data_id = '024r_029v'
data_id = '102v_107r'
conv_nd = 2

# file paths
data_path = f'networks/data/sgp/{data_id}/cropped_roi/*'
model_path = 'networks/model'

def for_img_input():
    # load test data
    print('Load test data..')
    imgs_norm = load_raw_images_data(data_path, rescale_ratio=0.25)
    sample_img = get_sample_image(data_path, rescale_ratio=0.25)
    img_height, img_width = sample_img.shape

    model = models.conv2d_net(23, img_width, img_height, 3)
    model.load_state_dict(torch.load(f'{model_path}/conv{conv_nd}d_on_{data_class}_{model_data_id}.pth', map_location='cpu'))
    model.eval()

    for name, param in model.named_parameters():
        print(name, param.shape)

    # Visualize feature maps
    activation = {}
    model.conv[0].register_forward_hook(get_activation('conv1', activation))
    output = model(torch.FloatTensor(imgs_norm))
    act = activation['conv1'].squeeze()
    view_feature_map(act)
    # Visualize conv filter
    kernels = model.conv[0].weight.detach()
    # normalize filter values to 0-1 so we can visualize them
    kernels = kernels - kernels.min()
    kernels = kernels / kernels.max()
    view_filter(kernels)


def for_pixel_input():
    # load test data
    print('Load test data..')
    test_imgs = load_raw_images_data(data_path, rescale_ratio=0.25, preserve_range_after_rescale=True)
    img_height, img_width = test_imgs[0].shape
    test_dataset, channel_len = load_patch_dataset_from_imgs(test_imgs, patch_size=3)

    model = models.conv_hybrid(channel_len, 3)
    model.load_state_dict(torch.load(f'{model_path}/conv_hybrid_on_{data_class}.pth', map_location='cpu'))
    model.eval()

    for name, param in model.named_parameters():
        print(name, param.shape)

    # Visualize feature maps
    activation = {}
    model.residual[0].register_forward_hook(get_activation('conv1', activation))
    output = models.get_model_output(test_dataset, model)
    act = activation['conv1']
    filter_num = act.size(1)
    reconstruct_feature_map(act, filter_num, img_height-2, img_width-2)
    # Visualize conv filter
    kernels = model.residual[0].weight.detach()
    # normalize filter values to 0-1 so we can visualize them
    kernels = kernels - kernels.min()
    kernels = kernels / kernels.max()
    # view_filter(kernels)


def get_activation(name, activation):
    def hook(model, input, output):
        if name not in activation:
            activation[name] = output.detach()
        else:
            activation[name] = torch.cat((activation[name], output.detach()))
    return hook


def reconstruct_feature_map(act, n_filter, img_h, img_w):
    act = torch.reshape(act, (-1, img_h, img_w, n_filter))
    act = act[:,:,:,0]
    rows = act.size(0)
    fig1, axarr = plt.subplots(rows, 1)
    for idx in range(rows):
        axarr[idx].imshow(act[idx], cmap='gray')


def view_feature_map(act):
    rows, cols = int(act.size(0)/4), 4
    fig1, axarr = plt.subplots(rows, 4)
    for idx in range(rows):
        for j in range(4):
            axarr[idx, j].imshow(act[idx*cols+j], cmap='gray')

def view_filter(kernels):
    rows, cols = int(kernels.size(0)/4), 4
    fig2, axarr = plt.subplots(rows, cols)
    for idx in range(rows):
        for j in range(4):
            if kernels.size(-1) != 1 and kernels.size(-2) != 1:
                axarr[idx, j].imshow(kernels[idx*cols+j][0].squeeze(), cmap='gray')
            else:
                axarr[idx, j].imshow(kernels[idx*cols+j][0], cmap='gray')
        if idx*cols >= 20:
            break

# # plot first few filters
# fig3 = plt.figure()
# rows, cols = 5, 4
# for i in range(rows):
#     for j in range(cols):
#         f = kernels[i*cols+j, 0, :, :]
#         # specify subplot and turn of axis
#         ax = plt.subplot(rows, cols, i*cols+j+1)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         # plot filter channel in grayscale
#         plt.imshow(f, cmap='gray')


for_pixel_input()
# show the figure
plt.show()