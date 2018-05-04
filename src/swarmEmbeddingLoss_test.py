import tensorflow as tf
import numpy as np

class SwarmLossTest(tf.test.TestCase):
	def setUp(self):
		self.batch = 1
		self.imgX = 5
		self.imgY = 10
		self.embed = 1

		# Known Numpy implementation:
		np.random.seed(0)
		self.X = np.random.random((self.batch, self.imgX, self.imgY, self.embed))  # 10 points in 3 dimensions
		self.Y = np.random.random((self.batch, self.imgX, self.imgY))  # 10 points in 3 dimensions

		self.rle_mask = np.array([[[[0,10],[0,0],[0,0],[0,0],[0,0]],[[0,0],[0,1],[0,1],[2,3],[0,0]]]])
		self.masked_averages = np.zeros((self.batch, 2, self.embed))  # 10 points in 3 dimensions

		for batch in range(self.batch):
			for embedding in range(2):
				emb_count = 0
				for x in range(self.imgX):
					y_1 = self.rle_mask[batch,embedding,x,0]
					y_2 = self.rle_mask[batch,embedding,x,1]
					if y_2 > y_1:
						self.masked_averages[batch,embedding] += np.sum(self.X[batch,x,y_1:y_2,:],axis=0)
						emb_count += y_2-y_1

				self.masked_averages[batch,embedding] /= emb_count

		self.differences = np.zeros((self.batch, self.imgX, self.imgY, self.embed))  # 10 points in 3 dimensions
		self.attractiveLoss = np.zeros((self.batch, self.imgX, self.imgY))  # 10 points in 3 dimensions

		for batch in range(self.batch):
			for embedding in range(2):
				emb_count = 0
				for x in range(self.imgX):
					y_1 = self.rle_mask[batch,embedding,x,0]
					y_2 = self.rle_mask[batch,embedding,x,1]
					if y_2 > y_1:
						for y in range(y_1,y_2):
							self.differences[batch,x,y] = self.X[batch,x,y] - self.masked_averages[batch,embedding]
							self.attractiveLoss[batch,x,y] = (0.5)*np.linalg.norm(self.differences[batch,x,y])**2

		self.feeder = tf.placeholder(tf.float32,[self.batch, self.imgX, self.imgY, self.embed])
		self.rle_feeder = tf.placeholder(tf.int32,[self.batch, 2, self.imgX, 2])
		self.average_feeder = tf.placeholder(tf.int32,[self.batch, 2, self.embed])



	def testMaskedAverages(self):
		swarmAverage_module = tf.load_op_library('../bin/swarmAverage.so')

		with self.test_session() as sess:
			result = swarmAverage_module.masked_averages(self.feeder,self.rle_feeder)
			masked_results = sess.run(result,feed_dict={self.feeder:self.X,self.rle_feeder:self.rle_mask})
			self.assertAllClose(self.masked_averages, masked_results)

	def testAttractiveLoss(self):
		swarmAverage_module = tf.load_op_library('../bin/swarmAverage.so')
		swarmAttractiveLoss_module = tf.load_op_library('../bin/swarmAttractiveLoss.so')

		with self.test_session() as sess:
			averages = swarmAverage_module.masked_averages(self.feeder,self.rle_feeder)
			result = swarmAttractiveLoss_module.attractive_loss(self.feeder,self.rle_feeder,averages)
			loss_results = sess.run(result,feed_dict={self.feeder:self.X,self.rle_feeder:self.rle_mask})
			
			self.assertAllClose(self.attractiveLoss, loss_results)

	def testAttractiveLossBackprop(self):
		swarmAverage_module = tf.load_op_library('../bin/swarmAverage.so')
		swarmAttractiveLoss_module = tf.load_op_library('../bin/swarmAttractiveLoss.so')

		with self.test_session() as sess:
			averages = swarmAverage_module.masked_averages(self.feeder,self.rle_feeder)
			result = swarmAttractiveLoss_module.derived_attractive_loss(self.feeder,self.rle_feeder,averages)
			backprop_results = sess.run(result,feed_dict={self.feeder:self.X,self.rle_feeder:self.rle_mask})
			
			self.assertAllClose(self.differences, backprop_results)



if __name__ == "__main__":
  tf.test.main()
