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
from utils import FolioDataset, load_images_data, load_raw_images_data
from utils import reconstruct_image
from utils import initialize_log_dataframe
from utils import plot_loss_acc_one_model

import time


data_class = 'allClass'
folio_ids = ['024r_029v', '102v_107r', '214v_221r']
data_id = folio_ids[2]
data_type = 'cropped_roi'
conv_nd = 2
# net_style {'normal': 0, 'fconv': 1, 'hybrid': 2}
net_style = 0


# file paths
data_path = f'autoencoder/data/sgp/{data_id}/cropped_roi/*'
model_path = 'autoencoder/model'
img_save_path = 'autoencoder/reconstructed_roi'
log_path = f'autoencoder/training_log/conv{conv_nd}ds/sgd'


# load training data
print('Load training data..')

# load images
print('-load images..')

imgs_norm = load_raw_images_data(data_path, rescale_ratio=0.25)
test_dataset, sample_img = load_images_data(data_path, rescale_ratio=0.25)
img_height, img_width = sample_img.shape


channel_len = 23
pxl_num = img_width * img_height


# load estimator
print('-load estimator..')
autoencoder = models.sdae(
    dimensions=[channel_len, 10, 10, 20, 3]
)
ae_model = models.sdae_lr(autoencoder)
ae_model.load_state_dict(torch.load(f'{model_path}/ae_on_{data_class}.pth', map_location='cpu'))
ae_model.eval()


# predict labels
print('-predict labels..')
ae_pred = models.predict_class(test_dataset, ae_model)

# reshape
y_true = np.reshape(ae_pred, (img_height, img_width))
# normalize
ae_pred_norm = torch.FloatTensor(normalize(torch.reshape(ae_pred,(1,-1)), norm='max')[0])


# conv model
print('Train conv net..')

num_epochs = 300
learning_rate = 1e-3

if net_style == 2:
    if conv_nd == 2:
        conv_model = models.conv2d_hyb_net(channel_len, img_width, img_height, 3)
    elif conv_nd == 3:
        conv_model = models.conv3d_hyb_net(channel_len, img_width, img_height, 3)
elif net_style == 1:
    if conv_nd == 2:
        conv_model = models.fconv2d_net(channel_len, img_width, img_height, 3)
    elif conv_nd == 3:
        conv_model = models.fconv3d_net(channel_len, img_width, img_height, 3)
elif net_style == 0:
    if conv_nd == 2:
        conv_model = models.conv2d_net(channel_len, img_width, img_height, 3)
    elif conv_nd == 3:
        conv_model = models.conv3d_net(channel_len, img_width, img_height, 3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(conv_model.parameters(), lr=learning_rate, momentum=0.9)
# optimizer = optim.Adam(conv_model.parameters(), lr=learning_rate, weight_decay=1e-5)
conv_model.train()

# log loss & acc in dataframe
log_df = initialize_log_dataframe()
# log time
start_time = time.time()

for epoch in range(num_epochs):
    output = conv_model(torch.FloatTensor(imgs_norm))
    _, conv_pred = torch.max(output.data, 1)
    loss = criterion(output, ae_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # log
    # if epoch % 10 == 0:
    acc = precision_score(ae_pred, conv_pred, average='micro')
    print('epoch [{}/{}], loss:{:.4f}, accuracy:{:.4f}' 
            .format(epoch + 1, num_epochs, loss.data.item(), acc))
    log_df.loc[epoch] = [epoch + 1, loss.data.item(), acc]

# time log
print("--- %s seconds ---" % (time.time() - start_time))

# save model
if net_style == 2:
    model_name = f'conv{conv_nd}d_hyb_on_{data_class}_{data_id}.pth'
elif net_style == 1:
    model_name = f'fconv{conv_nd}d_on_{data_class}_{data_id}.pth'
elif net_style == 0:
    model_name = f'conv{conv_nd}d_on_{data_class}_{data_id}.pth'
torch.save(conv_model.state_dict(), f'{model_path}/{model_name}')

# save log df
log_df.to_pickle(f'{log_path}/{model_name}_loss_acc_log.pkl')


print('Reconstruct..')
with torch.no_grad():
    output = conv_model(torch.FloatTensor(imgs_norm))
    _, conv_pred = torch.max(output.data, 1)

    print('-max in conv_pred: ', torch.max(conv_pred.data).item())
    print('-min in conv_pred: ', torch.min(conv_pred.data).item())


imsave(f'{img_save_path}/{data_id}_orig.png', sample_img)

sample_img_conv = reconstruct_image(sample_img, conv_pred, count_note=True)
if net_style == 2:
    img_name = f'{img_save_path}/conv{conv_nd}d_hyb/{data_id}_conv{conv_nd}d_hyb.png'
elif net_style == 1:
    img_name = f'{img_save_path}/fconv{conv_nd}d/{data_id}_fconv{conv_nd}d.png'
elif net_style == 0:
    img_name = f'{img_save_path}/conv{conv_nd}d/{data_id}_conv{conv_nd}d.png'
imsave(img_name, sample_img_conv)

# plot learning curve
plot_loss_acc_one_model(log_df)

