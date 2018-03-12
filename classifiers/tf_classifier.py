import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
import os
from skimage.io import imread
import matplotlib.pyplot as plt

import tensorflow as tf

from tf_loss import swarmEmbeddingLoss

X = tf.placeholder(tf.float32,shape=[5,500,500,3],name="Images_IN")
Y = tf.placeholder(tf.float32,shape=[5,500,500],name="Masks_IN")
Y_count = tf.placeholder(tf.float32,shape=[5,],name="Masks_Count_IN")

def convModel1(X):
	def convolution(x,input_size,num_filters,kernel_shape,dilation_rate=None,padding='same'):
		print input_size, num_filters, kernel_shape
		filters = tf.Variable(tf.random_normal(kernel_shape + [input_size, num_filters],0,0.05)) 
		convolveOp = tf.nn.convolution(x,  filters, "SAME",dilation_rate=dilation_rate)
		return convolveOp,filters

	IMG_CHANNELS = 3

	normalizer = tf.layers.BatchNormalization(axis=-1,name="batchNorm")
	x_normalized = normalizer(X)

	conv1,filters1 = convolution(x_normalized,3,8, [3,3], padding = 'same')
	conv1,filters2 = convolution(conv1,8,8, [3,3], padding = 'same')
	# use dilations to get a slightly larger field of view
	conv1,filters3 = convolution(conv1,8,16, [3,3], dilation_rate = [2,2], padding = 'same')
	conv1,filters4 = convolution(conv1,16,16, [3,3], dilation_rate = [2,2], padding = 'same')
	conv1,filters5 = convolution(conv1,16,32, [3,3], dilation_rate = [3,3], padding = 'same')

	# the final processing
	conv1,filters6 = convolution(conv1,32,16, [1,1], padding = 'same')
	conv1,filters7 = convolution(conv1,16,10, [1,1], padding = 'same')	
	return conv1


if __name__ == '__main__':
	model = convModel1(X)
	se_loss = swarmEmbeddingLoss(model,Y,Y_count, 0.1, 1)

	optimizer = tf.train.RMSPropOptimizer(0.0001,
	    decay=0.98,
	    momentum=0.001,
	    centered=True,
	    name='RMSProp')
	globalStep = tf.Variable(0, name='globalStep', trainable=False)

	trainOp = optimizer.minimize(se_loss, global_step=globalStep)

	sess = tf.Session()
	summBaseString = 'tfLogs/basic_test'
	writer = tf.summary.FileWriter(summBaseString)
	testWriter = tf.summary.FileWriter(summBaseString + '_test')

	init = tf.global_variables_initializer()
	sess.run(init)
