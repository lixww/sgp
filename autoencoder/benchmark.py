import pandas as pd
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

from skimage.io import imsave

from utils import load_raw_labeled_data
from utils import load_raw_images_data, flatten_images, reconstruct_image



data_id = '102v_107r'
data_type = 'cropped_roi'


# file paths
test_data_path = f'autoencoder/data/sgp/{data_id}/cropped_roi/*'
img_save_path = 'autoencoder/reconstructed_roi'



# prepare training set
print('Prepare training data..')

channel_train, y_true, channel_len = load_raw_labeled_data()


# fit lda
print('Model training..')
classifier = lda()
classifier.fit(channel_train, y_true)
precision_clf = classifier.score(channel_train, y_true)
print('train-accuracy: ', precision_clf)


# prepare test data
print('Prepare test data..')
imgs = load_raw_images_data(test_data_path)
channel_test, _ = flatten_images(imgs)

print('Model predict..')
predictions = classifier.predict(channel_test)


print('Reconstruct..')
sample_img = imgs[0]
sample_img = reconstruct_image(sample_img, predictions)
imsave(f'{img_save_path}/{data_id}_lda.png', sample_img)