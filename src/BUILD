# Description:
# An example for custom op and kernel defined as a TensorFlow plugin.

package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "swarmAverage.so",
    srcs = ["swarmAverage.cc"],
)

tf_custom_op_library(
    name = "swarmAttractiveLoss.so",
    srcs = ["swarmAttractiveLoss.cc"],
)

tf_custom_op_library(
    name = "swarmRepulsiveLoss.so",
    srcs = ["swarmRepulsiveLoss.cc"],
)