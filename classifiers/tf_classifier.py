import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
import os
from skimage.io import imread
import matplotlib.pyplot as plt

import tensorflow as tf

from tf_loss import swarmEmbeddingLoss, dice_loss

def convLayerToStr(input_size,num_filters,kernel_shape,dilation_rate=None):
	convFilterName = "convFilter_"
	convFilterName = convFilterName + "_input-" + str(input_size) + "_"
	convFilterName = convFilterName + "_output-" + str(num_filters) + "_"

	convFilterName = convFilterName + "_kernelShape-" 

	for k in kernel_shape[:-1]:
		convFilterName = convFilterName + str(k) + "-"
	convFilterName = convFilterName + str(kernel_shape[-1]) + "_"

	convFilterName = convFilterName + "_dialationRate-" 

	if not dilation_rate is None:
		for k in dilation_rate[:-1]:
			convFilterName = convFilterName + str(k) + "-"
		convFilterName = convFilterName + str(dilation_rate[-1]) + "_"

	return convFilterName


def convModel1(X):
	
	def convolution(x,input_size,num_filters,kernel_shape,dilation_rate=None,padding='same'):
		print input_size, num_filters, kernel_shape
		filters = tf.get_variable(convLayerToStr(input_size,num_filters,kernel_shape,dilation_rate),kernel_shape + [input_size, num_filters]) 
		convolveOp = tf.nn.convolution(x,  filters, "SAME",dilation_rate=dilation_rate)
		return convolveOp,filters

	IMG_CHANNELS = 3

	normalizer = tf.layers.BatchNormalization(axis=-1,name="batchNorm")
	x_normalized = normalizer(X)
	tf.summary.scalar('X_normalized', tf.reduce_mean(x_normalized))


	conv_preSplit,filters1 = convolution(X,3,8, [3,3], padding = 'same')
	tf.summary.scalar('split', tf.reduce_mean(conv_preSplit))
	tf.summary.scalar('filters', tf.reduce_mean(filters1))

	conv_preSplit,filters2 = convolution(conv_preSplit,8,8, [3,3], padding = 'same')
	# use dilations to get a slightly larger field of view
	conv_preSplit,filters3 = convolution(conv_preSplit,8,16, [3,3], dilation_rate = [2,2], padding = 'same')
	conv_split,filters4 = convolution(conv_preSplit,16,16, [3,3], dilation_rate = [2,2], padding = 'same')



	conv1,filters5 = convolution(conv_split,16,32, [3,3], dilation_rate = [3,3], padding = 'same')
	conv1,filters6 = convolution(conv1,32,16, [1,1], padding = 'same')
	embedding,filters7 = convolution(conv1,16,10, [1,1], padding = 'same')

	conv2,filters5 = convolution(conv_split,16,31, [3,3], dilation_rate = [3,3], padding = 'same')
	conv2,filters6 = convolution(conv2,31,16, [1,1], padding = 'same')
	mask_prediction,filters7 = convolution(conv2,16,1, [1,1], padding = 'same')	
	return embedding,mask_prediction


if __name__ == '__main__':
	train_img_df = pd.read_pickle("cell.df")
	def simple_gen():
		while True:
			for _, c_row in train_img_df.iterrows():
				yield np.expand_dims(c_row['images'],0), np.expand_dims(c_row['masks'],0),np.expand_dims(c_row['masksCount'],0)
	X = tf.placeholder(tf.float32,shape=[1,500,500,3],name="Images_IN")
	Y = tf.placeholder(tf.float32,shape=[1,500,500],name="Masks_IN")
	Y_count = tf.placeholder(tf.float32,shape=[1,],name="Masks_Count_IN")

	embeddings,mask_pred = convModel1(X)
	se_loss = swarmEmbeddingLoss(embeddings,Y,Y_count, 0.1, 1)
	mask_loss = dice_loss(mask_pred,Y)

	tf.summary.scalar('embedded_average', tf.reduce_mean(embeddings))
	tf.summary.scalar('mask_average', tf.reduce_mean(mask_loss))

	tf.summary.scalar('X_average', tf.reduce_mean(X))
	tf.summary.scalar('Y_average', tf.reduce_mean(Y))

	tf.summary.scalar('swarmEmbeddingLoss', se_loss)
	tf.summary.scalar('dice_loss', mask_loss)

	summaries = tf.summary.merge_all()

	batcher = simple_gen()

	optimizer = tf.train.RMSPropOptimizer(0.0001,
	    decay=0.98,
	    momentum=0.001,
	    centered=True,
	    name='RMSProp')
	globalStep = tf.Variable(0, name='globalStep', trainable=False)

	trainOp = optimizer.minimize(2*se_loss + mask_loss, global_step=globalStep)

	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)


	summBaseString = 'tfLogs/basic_test'
	writer = tf.summary.FileWriter(summBaseString)
	testWriter = tf.summary.FileWriter(summBaseString + '_test')



	for i in range(100):
		images, masks, mask_count = next(batcher)

		trainNP, summariesNP = sess.run([trainOp,summaries],feed_dict={Y:masks,Y_count:mask_count,X:images})
		writer.add_summary(summariesNP, i)