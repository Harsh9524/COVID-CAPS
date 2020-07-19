# Saving images and labels into numpy arrays

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
# Set parameters here 
INPUT_SIZE = (224, 224)
mapping = {'covid': 0, 'non-covid': 1}

# train/valid/test .txt files
train_filepath = '/kaggle/working/covid_ct_dataset/train/train.csv'
valid_filepath = '/kaggle/working/covid_ct_dataset/valid/valid.csv'
test_filepath = '/kaggle/working/covid_ct_dataset/test/test.csv'

# load in the train and test files
trainfiles = pd.read_csv(train_filepath,header=None)
validfiles = pd.read_csv(valid_filepath,header=None)
testfiles = pd.read_csv(test_filepath,header=None)

print('Total samples for train: ', len(trainfiles))
print('Total samples for valid: ', len(validfiles))
print('Total samples for test: ', len(testfiles))

# Total samples for train:  5310
# Total samples for test:  639

# load in images
# resize to input size and normalize to 0 - 1
x_train = []
x_valid = []
x_test = []
y_train = []
y_valid = []
y_test = []


# Create ./data/test - ./data/train - ./data/valid directories yourself
for i in range(len(testfiles)):
    # test_i = testfiles[i]
    imgpath = testfiles[1][i]
    img = cv2.imread(os.path.join(r'/kaggle/working/covid_ct_dataset/test', imgpath))
    img = cv2.resize(img, INPUT_SIZE) # resize
    img = img.astype('float32') / 255.0
    x_test.append(img)
    y_test.append(mapping[testfiles[2][i]])

print('Shape of test images: ', x_test[0].shape)

for i in range(len(validfiles)):
    # valid_i = validfiles[i].split()
    imgpath = validfiles[1][i]
    img = cv2.imread(os.path.join(r'/kaggle/working/covid_ct_dataset/valid', imgpath))
    img = cv2.resize(img, INPUT_SIZE) # resize
    img = img.astype('float32') / 255.0
    x_valid.append(img)
    y_valid.append(mapping[validfiles[2][i]])

print('Shape of valid images: ', x_valid[0].shape)

for i in range(len(trainfiles)):
    # train_i = trainfiles[i].split()
    imgpath = trainfiles[1][i]
    img = cv2.imread(os.path.join(r'/kaggle/working/covid_ct_dataset/train', imgpath))
    img = cv2.resize(img, INPUT_SIZE) # resize
    img = img.astype('float32') / 255.0
    x_train.append(img)
    y_train.append(mapping[trainfiles[2][i]])

print('Shape of train images: ', x_train[0].shape)

# Shape of test images:  (224, 224, 3)
# Shape of train images:  (224, 224, 3)
# export to npy to load in for training
np.save('x_train.npy', x_train)
np.save('y_train.npy', y_train)
np.save('x_valid.npy', x_valid)
np.save('y_valid.npy', y_valid)
np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)

