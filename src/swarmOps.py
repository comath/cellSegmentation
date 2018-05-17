import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

swarmAverage_module = tf.load_op_library('../bin/swarmAverage.so')
swarmAttractiveLoss_module = tf.load_op_library('../bin/swarmAttractiveLoss.so')
swarmRepulsiveLoss_module = tf.load_op_library('../bin/swarmRepulsiveLoss.so')

def masked_averages(x, rle_masks):
	return swarmAverage_module.masked_averages(x,rle_masks)

ops.NotDifferentiable("MaskedAverages")

def attractive_loss(x,rle_masks,averages):
	return swarmAttractiveLoss_module.attractive_loss(x,rle_masks,averages)

@ops.RegisterGradient("AttractiveLoss")
def _attractive_loss_grad(op, grad):
	x = op.inputs[0]
	rle_masks = op.inputs[1]
	averages = op.inputs[2]

	return [swarmAttractiveLoss_module.derived_attractive_loss(x,rle_masks,averages,grad),
			None,
			None]

def repulsive_loss(x,rle_masks,averages):
	return swarmRepulsiveLoss_module.repulsive_loss(x,rle_masks,averages)

@ops.RegisterGradient("RepulsiveLoss")
def _repulsive_loss_grad(op, grad):
	x = op.inputs[0]
	rle_masks = op.inputs[1]
	averages = op.inputs[2]

	return [swarmRepulsiveLoss_module.derived_repulsive_loss(x,rle_masks,averages,grad),
			None,
			None]