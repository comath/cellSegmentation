import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
import os
from skimage.io import imread
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, UpSampling2D, Lambda

import tensorflow as tf

train_img_df = pd.read_pickle("cell.df")
n_img = 6

X = tf.placeholder(tf.float32,shape=[None,500,500,3],name="Images_IN")
Y = tf.placeholder(tf.float32,shape=[None,500,500],name="Masks_IN")
Y_count = tf.placeholder(tf.float32,shape=[None,],name="Masks_Count_IN")

zeroes = tf.zeros([500*500,3])

#for i in range(1,400):
two = tf.constant(2, dtype=tf.float32)

mask = tf.equal(Y,0)

results = tf.boolean_mask(X, mask)


#mask = tf.reshape(mask,[None,500,500,3])

sess = tf.Session()
d_row = train_img_df.sample(n_img)
print sess.run(results,feed_dict={Y:d_row["masks"],X:d_row["images"]})