import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
import os
from skimage.io import imread
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, UpSampling2D, Lambda

import tensorflow as tf


 
def swarmEmbeddingLoss(X,Y,Y_count, delta1, delta2):
	secondLoss_mask = tf.ones([399,399])-tf.eye(399)
	losses = []
	
	perImage = tf.unstack(X)
	perMask = tf.unstack(Y)
	perMaskCount = tf.unstack(Y_count)
	for (embeddings,masks),masks_count in zip(zip(perImage,perMask),perMaskCount):
		# Variance Distance, makes sure the embedded points are attracted to their own center
		avgs = []
		norm1 = []
		for i in range(1,400):
			maskedEmbedding = tf.boolean_mask(embeddings, tf.equal(masks,i))
			avgEmbedding = tf.reduce_mean(maskedEmbedding,axis=0)
			avgs.append(avgEmbedding)
			l2Norm = tf.reduce_mean(tf.squared_difference(maskedEmbedding,avgEmbedding),axis = 1)
			norm1.append(tf.reduce_mean(tf.nn.relu(l2Norm -delta1)))
		norm1 = (1/masks_count)*tf.reduce_sum(norm1)
		# Embedding Distance, makes sure the embeddings are seperated from each other
		avgs = tf.stack(avgs)
		expanded_a = tf.expand_dims(avgs, 1)
		expanded_b = tf.expand_dims(avgs, 0)
		distMatrix = tf.nn.relu(tf.square(expanded_a - expanded_b) )
		distMatrix = tf.reduce_sum(distMatrix,2)
		distMatrix = tf.multiply(secondLoss_mask,distMatrix)
		norm2 = (1/(masks_count*(masks_count-1)))*tf.reduce_sum(distMatrix)

		# L1 regularizer
		regulizers = tf.reduce_sum(tf.abs(expanded_a - expanded_b))

		losses.append(norm1 + norm2 + 0.001*regulizers)


	final_loss = tf.reduce_sum(tf.stack(losses))

	return final_loss

if __name__ == '__main__':
	train_img_df = pd.read_pickle("cell.df")
	n_img = 6
	batchSize = 5
	embeddingSize = 3

	X = tf.placeholder(tf.float32,shape=[batchSize,500,500,3],name="Images_IN")
	Y = tf.placeholder(tf.float32,shape=[batchSize,500,500],name="Masks_IN")
	Y_count = tf.placeholder(tf.float32,shape=[batchSize,],name="Masks_Count_IN")

	sampledImages = []
	sampledMasks = []
	sampledMaskCount = []
	samples = train_img_df.sample(batchSize)
	for _, c_row in samples.iterrows():
		sampledImages.append(c_row['images'])
		sampledMasks.append(c_row['masks'])
		sampledMaskCount.append(c_row['masksCount'])
	sampledImages = np.stack(sampledImages)
	sampledMasks = np.stack(sampledMasks)
	sampledMaskCount = np.stack(sampledMaskCount)

	se_loss = swarmEmbeddingLoss(X,Y,Y_count, 0.1, 1.0)

	optimizer = tf.train.RMSPropOptimizer(0.0001,
	    decay=0.98,
	    momentum=0.001,
	    centered=True,
	    name='RMSProp')
	globalStep = tf.Variable(0, name='globalStep', trainable=False)

	trainOp = optimizer.minimize(se_loss, global_step=globalStep)
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)

	print sess.run(se_loss,feed_dict={Y:sampledMasks,Y_count:sampledMaskCount,X:sampledImages})

	