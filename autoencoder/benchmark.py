import pandas as pd
import numpy as np

from sklearn.model_selection import ShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

from skimage.io import imread, imsave
from skimage.io import imread_collection



data_id = '214v_221r'
data_type = 'cropped_roi'

img_width, img_height = (699, 684)

# file paths
train_data_path = 'autoencoder/data/sgp'
test_data_path = f'autoencoder/data/sgp/{data_id}/cropped_roi/*'
img_save_path = 'autoencoder/reconstructed_roi'

channel_len = 23
pxl_num = img_width * img_height


# prepare training set
print('Prepare training data..')
training_file = pd.read_csv(f'{train_data_path}/training_file_8_bit.csv')

location_head = training_file.columns[2:4]
channel_head = training_file.columns[4:]

y_true = training_file['class_name'].to_numpy()
location_train = training_file[location_head].to_numpy()
channel_train = training_file[channel_head].to_numpy()

data_idx = training_file.index


# fit lda
print('Model training..')
classifier = lda()
classifier.fit(channel_train, y_true)
precision_clf = classifier.score(channel_train, y_true)
print('train-accuracy: ', precision_clf)


# prepare test data
print('Prepare test data..')
ic = imread_collection(test_data_path)
imgs = []
for f in ic.files:
    imgs.append(imread(f, as_gray=True))

channel_test = []
location_test = []
for h in range(img_height):
    for w in range(img_width):
        data = []
        for i in range(channel_len):
            data.append(imgs[i][h][w])
        channel_test.append(data)
        location_test.append((w+1, h+1))


print('Model predict..')
predictions = classifier.predict(channel_test)


print('Reconstruct..')
sample_img = imgs[0]
for h in range(img_height):
    for w in range(img_width):
        index = (img_width)*h + w
        if predictions[index] == 2:
            sample_img[h][w] -= 20
imsave(f'{img_save_path}/{data_id}_lda.tif', sample_img)