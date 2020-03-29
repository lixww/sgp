import torch
from torch.utils.data import DataLoader

from torchvision.utils import save_image
from torchvision import transforms

from skimage.io import imread_collection
from skimage.io import imread
from skimage.io import imsave

from models import simple_ae


data_id = '102v_107r'
data_type = 'snip'
w_h_no = 1
preprocess_type = 'processed'

data_path = f'autoencoder/data/sgp/{data_id}/{preprocess_type}_{data_type}/*'
img_save_path = f'autoencoder/reconstructed_{data_type}'
model_path = 'autoencoder/model'

width_height_list = ((576, 380), (1164, 1088))
img_width, img_height = width_height_list[w_h_no]

batch_size = 26
img_size = img_width * img_height


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, img_height, img_width)
    return x


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


model = simple_ae(img_size)
model.load_state_dict(torch.load(f'{model_path}/simae_on_{preprocess_type}_{data_type}_{data_id}.pth', map_location='cpu'))

ic = imread_collection(data_path)
imgs = []
for f in ic.files:
    imgs.append(torch.tensor(img_transform(imread(f, as_gray=True)), dtype=torch.float32))
batch_size = len(imgs)
dataloader = DataLoader(imgs, batch_size=batch_size, shuffle=False)

# view output
count = 0
for data in dataloader:
    img = data.view(data.size(0), -1)

    pic = to_img(img)
    save_image(pic, '{}/{}_{}_input_{}.png'.format(img_save_path, data_id, preprocess_type, count))

    output = model(img)
    
    pic = to_img(output.data)
    save_image(pic, '{}/{}_{}_output_{}.png'.format(img_save_path, data_id, preprocess_type, count))
    count += 1