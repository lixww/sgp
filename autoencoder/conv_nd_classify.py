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



data_class = 'allClass'
data_id = '024r_029v'
data_type = 'cropped_roi'
conv_nd = 3
is_fcnet = True


# file paths
data_path = f'autoencoder/data/sgp/{data_id}/cropped_roi/*'
model_path = 'autoencoder/model'
img_save_path = 'autoencoder/reconstructed_roi'


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

num_epochs = 1000
learning_rate = 0.01

if is_fcnet:
    if conv_nd == 2:
        conv_model = models.fconv2d_net(channel_len, img_width, img_height, 3)
    elif conv_nd == 3:
        conv_model = models.fconv3d_net(channel_len, img_width, img_height, 3)
else:
    if conv_nd == 2:
        conv_model = models.conv2d_net(channel_len, img_width, img_height, 3)
    elif conv_nd == 3:
        conv_model = models.conv3d_net(channel_len, img_width, img_height, 3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(conv_model.parameters(), lr=learning_rate, momentum=0.9)
conv_model.train()

for epoch in range(num_epochs):
    output = conv_model(torch.FloatTensor(imgs_norm))
    _, conv_pred = torch.max(output.data, 1)
    loss = criterion(output, ae_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # log
    if epoch % 10 == 0:
        acc = precision_score(ae_pred, conv_pred, average='micro')
        print('epoch [{}/{}], loss:{:.4f}, accuracy:{:.4f}' 
                .format(epoch + 1, num_epochs, loss.data.item(), acc))


# save model
if is_fcnet:
    model_name = f'{model_path}/fconv{conv_nd}d_on_{data_class}_{data_id}.pth'
else:
    model_name = f'{model_path}/conv{conv_nd}d_on_{data_class}_{data_id}.pth'
torch.save(conv_model.state_dict(), model_name)


print('Reconstruct..')
with torch.no_grad():
    output = conv_model(torch.FloatTensor(imgs_norm))
    _, conv_pred = torch.max(output.data, 1)

    print('-max in conv_pred: ', torch.max(conv_pred.data).item())
    print('-min in conv_pred: ', torch.min(conv_pred.data).item())


imsave(f'{img_save_path}/{data_id}_orig.png', sample_img)

sample_img_conv = reconstruct_image(sample_img, conv_pred, count_note=True)
if is_fcnet:
    img_name = f'{img_save_path}/{data_id}_fconv{conv_nd}d.png'
else:
    img_name = f'{img_save_path}/{data_id}_conv{conv_nd}d.png'
imsave(img_name, sample_img_conv)

