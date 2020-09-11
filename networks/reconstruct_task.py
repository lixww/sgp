import os
from skimage.io import imread_collection
from skimage.io import imread
from skimage.io import imsave

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from pathlib import Path

from models import simple_ae

# train a simple_autoencoder

data_id = '102v_107r'
data_type = 'snip'
file_type = 'tif'
w_h_no = 1
preprocess_type = 'processed'

data_path = f'networks/data/sgp/{data_id}/{preprocess_type}_{data_type}/*.{file_type}'
img_save_path = f'networks/reconstructed_{data_type}'
model_path = 'networks/model'
# mkdir if not exists
Path(f'{img_save_path}').mkdir(parents=True, exist_ok=True)

width_height_list = ((576, 380), (1164, 1088))
img_width, img_height = width_height_list[w_h_no]

num_epochs = 100
batch_size = 20
img_size = img_width * img_height
learning_rate = 1e-2



def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, img_height, img_width)
    return x


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

ic = imread_collection(data_path)
imgs = []
for f in ic.files:
    imgs.append(torch.tensor(img_transform(imread(f, as_gray=True)), dtype=torch.float32))
dataloader = DataLoader(imgs, batch_size=batch_size, shuffle=False)

# imsave('after_norm.jpg', imgs[23].view(img_height, img_width).numpy())



# model = autoencoder().cuda()
model = simple_ae(img_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in dataloader:
        img = data
        img = img.view(img.size(0), -1)
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data.item()))
    if epoch % 10 == 0:
        pic = to_img(output.data)
        save_image(pic, '{}/image_{}.png'.format(img_save_path, epoch))

torch.save(model.state_dict(), f'{model_path}/simae_on_{preprocess_type}_{data_type}_{data_id}.pth')
# torch.load('./sim_autoencoder.pth')