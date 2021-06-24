import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from random import shuffle
from collections import deque

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


path='C:/Users/SamSung/Desktop/uni/y4/year_project/img_stats'# appropriate file path (.lif file)

names=os.listdir(path)
shuffle(names)
split_no=12 # 12 to 23 train/test를 어디서 나눌건지
control_aug_no=0
mutant_aug_no=0

train_names=names[:split_no]
test_names=names[split_no:]

train_points=[]
train_labels=[]
test_points=[]
test_labels=[]

def augment(points):
    # jitter points
    points += tf.random.uniform(points.shape, -2, 2, dtype=tf.float64)
    # shuffle points
    points = tf.random.shuffle(points)
    return points

def augment2(points):
    # jitter points
    a = tf.random.uniform(points.shape, -2, 2, dtype=tf.float64)
    points += a
    # shuffle points
    points = tf.random.shuffle(points)
    return points
def count_samples(trainset):
    plus=0
    for filename in trainset:
        if '+' in filename:
            plus+=1
        else:
            pass
    return plus
#print('there are ',count_samples(train_names),'control samples in the trainset')
N=count_samples(train_names)# number of healthy in INITIAL training set.
control_aug_no=60
mutant_aug_no=int((N/(12-N))*(control_aug_no+1)-1)# balance no. of samples in each class

for name in train_names:
    if '+' in name:
        xl=tf.convert_to_tensor(pd.read_csv(os.path.join(path,name))[['X','Y','Z']].sample(n=119).to_numpy())
        train_points.append(xl)
        train_labels.append(1)# healthy
        for i in range(control_aug_no):
            #augment healthy
            #train_points.append(augment(xl))
            #train_points.append(augment2(xl))
            train_points.append(augment(augment2(xl)))
            train_labels.append(1)# healthy
    else:
        xl=tf.convert_to_tensor(pd.read_csv(os.path.join(path,name))[['X','Y','Z']].sample(n=119).to_numpy())
        train_points.append(xl)
        train_labels.append(0)# mutant
        #augment mutant n times
        for i in range(mutant_aug_no):
            #train_points.append(augment(xl))
            #train_points.append(augment2(xl))
            train_points.append(augment(augment2(xl)))
            train_labels.append(0)# mutant

for name in test_names:
    if '+' in name:
        xl=tf.convert_to_tensor(pd.read_csv(os.path.join(path,name))[['X','Y','Z']].sample(n=119).to_numpy())
        test_points.append(xl)
        test_labels.append(1)
    else:
        xl=tf.convert_to_tensor(pd.read_csv(os.path.join(path,name))[['X','Y','Z']].sample(n=119).to_numpy())
        test_points.append(xl)
        test_labels.append(0)
from collections import Counter


NUM_POINTS = 119
NUM_CLASSES = 2
BATCH_SIZE = 32
EPOCHS=50


#create dataset
trainset=tf.data.Dataset.from_tensor_slices((train_points, train_labels))
#print('d')

testset=tf.data.Dataset.from_tensor_slices((test_points, test_labels))
trainset = trainset.shuffle(len(train_points)).batch(BATCH_SIZE)
testset = testset.shuffle(len(test_points)).batch(BATCH_SIZE)
#print('done')


def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

def tnet(inputs, num_features):

    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])

inputs = keras.Input(shape=(NUM_POINTS, 3))# none, 119, 3
#print(inputs)

x = tnet(inputs, 3)
x = conv_bn(x, 32)
x = conv_bn(x, 32)
x = tnet(x, 32)
x = conv_bn(x, 32)
x = conv_bn(x, 64)
x = conv_bn(x, 128)# 원래 512
x = layers.GlobalMaxPooling1D()(x)
x = dense_bn(x, 64)
x = layers.Dropout(0.4)(x)
x = dense_bn(x, 32)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
#model.summary()
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.03),
    metrics=["binary_accuracy"],
)
history=model.fit(trainset,epochs=EPOCHS,validation_data=testset)


# summarize history for accuracy
trainacc=list(history.history['binary_accuracy'])
valacc=list(history.history['val_binary_accuracy'])

def mvavg(iter, ws):
    i=0
    moving_averages=[]
    while i < len(iter) - ws+ 1:
        this_window = iter[i : i + ws]
        window_average = sum(this_window) / ws
        moving_averages.append(window_average)
        i += 1
    return moving_averages
#trainacc=mvavg(trainacc,5)
#valacc=mvavg(valacc,5)

plt.plot(trainacc)
plt.plot(valacc)
plt.legend(['acc', 'val_acc'], loc='upper left')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.show()
from math import log
# show loss graph
trainloss=list(map(log, history.history['loss']))
valoss=list(map(log,history.history['val_loss']))
plt.plot(trainloss)
plt.plot(valoss)
plt.title('loss')
plt.ylabel('losses')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss'], loc='upper left')
plt.show()
