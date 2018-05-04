# Cell Segmentation

To build this first download the tensorflow source and put the three swarm op source files and the bazel BUILD file in `tensorflow/core/user_ops` and run:

```
bazel build --config opt //tensorflow/core/user_ops:swarmRepulsiveLoss.so
bazel build --config opt //tensorflow/core/user_ops:swarmAttractiveLoss.so
bazel build --config opt //tensorflow/core/user_ops:swarmAverage.so

cp -rf bazel-bin/tensorflow/core/user_ops/* ../cellDetection/bin

```