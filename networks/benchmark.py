import pandas as pd
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score

from skimage.io import imsave

from pathlib import Path

from utils import load_raw_labeled_data
from utils import load_raw_images_data, flatten_images, reconstruct_image
from utils import plot_learning_curve


# use sklearn.lda as benchmark model to train data in training_file_8_bit.csv

data_id = '102v_107r'
data_type = 'cropped_roi'


# file paths
test_data_path = f'networks/data/sgp/{data_id}/{data_type}/*'
img_save_path = 'networks/reconstructed_roi/lda'
# mkdir if not exists
Path(f'{img_save_path}').mkdir(parents=True, exist_ok=True)



# prepare training set
print('Prepare training data..')

channel_train, y_true, channel_len = load_raw_labeled_data()


# fit lda
print('Model training..')
classifier = lda()
classifier.fit(channel_train, y_true)
precision_clf = classifier.score(channel_train, y_true)
prediction = classifier.predict(channel_train)
balanced_acc = balanced_accuracy_score(y_true, prediction)
kappa = cohen_kappa_score(y_true, prediction)
# plot learning curve
plot_learning_curve(classifier, 'learning curve of LDA', channel_train, y_true)

print('train-accuracy: ', precision_clf)
print('balanced-accuracy: ', balanced_acc)
print('kappa: ', kappa)


# prepare test data
print('Prepare test data..')
imgs = load_raw_images_data(test_data_path, rescale_ratio=0.25, preserve_range_after_rescale=True)
channel_test, _ = flatten_images(imgs)

print('Model predict..')
predictions = classifier.predict(channel_test)


print('Reconstruct..')
sample_img = imgs[0]
sample_img = reconstruct_image(sample_img, predictions)
imsave(f'{img_save_path}/{data_id}_lda.png', sample_img)