import pandas as pd
import numpy as np

from skimage.io import imread_collection
from skimage.io import imread, imsave
from skimage.transform import rescale

from sklearn.metrics import precision_score

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

from sklearn.preprocessing import normalize

import models
from utils import FolioDataset



data_class = 'allClass'
data_id = '102v_107r'
data_type = 'cropped_roi'

img_width, img_height = (699, 684)

# file paths
data_path = f'autoencoder/data/sgp/{data_id}/cropped_roi/*'
model_path = 'autoencoder/model'
img_save_path = 'autoencoder/reconstructed_roi'


# load training data
print('Load training data..')

# load images
print('-load images..')
ic = imread_collection(data_path)
imgs_norm = []
imgs = []
for f in ic.files:
    image = imread(f, as_gray=True)
    imgs_norm.append(rescale(image, 0.25, anti_aliasing=False))
    imgs.append(rescale(image, 0.25, anti_aliasing=False, preserve_range=True))
img_height, img_width = imgs_norm[0].shape


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

# prepare images
print('-prepare images..')
channel = []
location = []
y_true = [-1] * pxl_num
for h in range(img_height):
    for w in range(img_width):
        data = []
        for i in range(channel_len):
            data.append(imgs[i][h][w])
        channel.append(data)
        location.append((w+1, h+1))

test_dataset = FolioDataset(location, channel, y_true)

# predict labels
print('-predict labels..')
ae_pred = models.predict_class(test_dataset, ae_model)

# reshape
y_true = np.reshape(ae_pred, (img_height, img_width))
# normalize
ae_pred_norm = torch.FloatTensor(normalize(torch.reshape(ae_pred,(1,-1)), norm='max')[0])


# conv model
print('Train conv net..')

num_epochs = 2000
learning_rate = 0.01

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
torch.save(conv_model.state_dict(), f'{model_path}/conv_on_{data_class}.pth')


print('Reconstruct..')
with torch.no_grad():
    output = conv_model(torch.FloatTensor(imgs_norm))
    _, conv_pred = torch.max(output.data, 1)

    print('-max in conv_pred: ', torch.max(conv_pred.data).item())
    print('-min in conv_pred: ', torch.min(conv_pred.data).item())

sample_img_conv = imgs[0]
sample_img_ae = imgs[0]
imsave(f'{img_save_path}/{data_id}_orig.png', imgs[0])
tmp_count_conv_2 = 0
tmp_count_conv_01 = 0
tmp_count_ae_2 = 0
tmp_count_ae_01 = 0
for h in range(img_height):
    for w in range(img_width):
        index = (img_width)*h + w
        if conv_pred[index] == 2:
            sample_img_conv[h][w] -= 20
            tmp_count_conv_2 += 1
        else:
            sample_img_conv[h][w] += 20
            tmp_count_conv_01 += 1
        if ae_pred[index] == 2:
            # sample_img_ae[h][w] -= 20
            tmp_count_ae_2 += 1
        else:
            # sample_img_ae[h][w] += 20
            tmp_count_ae_01 += 1
imsave(f'{img_save_path}/{data_id}_conv.png', sample_img_conv)
imsave(f'{img_save_path}/{data_id}_ae.png', sample_img_ae)

print('tmp_count_conv_2 = ', tmp_count_conv_2)
print('tmp_count_conv_01 = ', tmp_count_conv_01)
print('tmp_count_ae_2 = ', tmp_count_ae_2)
print('tmp_count_ae_01 = ', tmp_count_ae_01)

