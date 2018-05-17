#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

/*
For a image mask with run length encoding with shape (batch_size,num_masks,dim1,2) where 
the last dimension holds the dim2_min, dim2_max coordinates, and a set of average points 
(batch_size,num_masks,embedding_size). 

This computes

 L_(batch,x,y) = (1/(n-1))\Sum_{j!=i} 1/2 || E_{batch,x,y} - M_{batch,j} ||_2^2   if y \in {y_{batch,i,x,1},...y_{batch,i,x,2}} 
                 0 								                                    else  
*/
REGISTER_OP("RepulsiveLoss")
	.Attr("T: {float32, float64}")
	.Attr("S: {int32, int64}")
  .Input("embeddings: T")
  .Input("rle_mask: S")
  .Input("averages: T")
  .Output("repulsive_losses: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle input;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));

    shape_inference::ShapeHandle rle_input;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &rle_input));

    shape_inference::DimensionHandle rle_end_dim;
    TF_RETURN_IF_ERROR(c->WithValue(c->DimKnownRank(rle_input, 3), 2, &rle_end_dim));

    shape_inference::ShapeHandle mean_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 3, &mean_shape));

    shape_inference::ShapeHandle loss_shape;
    TF_RETURN_IF_ERROR(c->Subshape(input,0,-1,&loss_shape));

    c->set_output(0,loss_shape);
    return Status::OK();
  }
);


// CPU specialization of actual computation.
template <typename Device, typename T, typename S>
struct RepulsiveLossFunctor {
  void operator()(const Device& d,
      const Eigen::Tensor<T,4, Eigen::RowMajor> logits_embeddings, 
      const Eigen::Tensor<S,4, Eigen::RowMajor> rle_mask, 
      const Eigen::Tensor<T,3, Eigen::RowMajor> mean_embeddings,
      T* loss_outputs) {

    const int batch_size = logits_embeddings.dimension(0);
    const int size_x = logits_embeddings.dimension(1);
    const int size_y = logits_embeddings.dimension(2);
    const int embedding_size = logits_embeddings.dimension(3);
    const int embedding_count = rle_mask.dimension(1);

    Eigen::array<int, 4> window_logits(1,1,1,embedding_size);
    Eigen::array<int, 4> offset_logits(0,0,0,0);

    Eigen::array<int, 3> window_mean(1,1,embedding_size);
    Eigen::array<int, 3> offset_mean(0,0,0);
    Eigen::array<int, 1> dims2({2});
    Eigen::array<int, 1> dims3({2});

    // Create the average embedding for each mask
    for (int batch = 0; batch < batch_size; ++batch) {
      offset_logits[0] = batch;
      offset_mean[0] = batch;

      for(int embedding_current = 0; embedding_current < embedding_count; ++embedding_current) {
        for(int x = 0; x < size_x; ++x) {
          offset_logits[1] = x;
          S min_y = std::max(0,rle_mask(batch,embedding_current,x,0));
          S max_y = std::min(size_y,rle_mask(batch,embedding_current,x,1));
          for(int y = min_y; y < max_y; ++y) {
            offset_logits[2] = y;
            for(int embedding_other = 0; embedding_other < embedding_current; ++embedding_other) {
              offset_mean[1] = embedding_other;
              auto mean_scratch = mean_embeddings.slice(offset_mean,window_mean);
              auto my_mean_embedding = (mean_scratch.reshape(window_logits)); // Delays the computation and allows the compiler to optimize
              Eigen::Tensor<T,3, Eigen::RowMajor> middle = (logits_embeddings.slice(offset_logits,window_logits) - my_mean_embedding).square().sum(dims3);
              loss_outputs[batch*size_x*size_y + x*size_y + y] += 0.5*middle.data()[0]; 
            }
            for(int embedding_other = embedding_current + 1; embedding_other < embedding_count; ++embedding_other) {
              offset_mean[1] = embedding_other;
              auto mean_scratch = mean_embeddings.slice(offset_mean,window_mean);
              auto my_mean_embedding = (mean_scratch.reshape(window_logits));
              Eigen::Tensor<T,3, Eigen::RowMajor> middle = (logits_embeddings.slice(offset_logits,window_logits) - my_mean_embedding).square().sum(dims3);
              loss_outputs[batch*size_x*size_y + x*size_y + y] += 0.5*middle.data()[0]; 
            }
            loss_outputs[batch*size_x*size_y + x*size_y + y] /= embedding_count - 1;
          }
          
        } 
      }
    }
  }
};

template <typename Device, typename T, typename S>
class RepulsiveLossOp : public OpKernel {
 public:
  explicit RepulsiveLossOp(OpKernelConstruction* context) : OpKernel(context) {}


  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    const Tensor& rle_tensor = context->input(1);
    const Tensor& mean_tensor = context->input(2);

    TensorShape input_shape = input_tensor.shape();
    TensorShape loss_shape = input_shape;
    loss_shape.RemoveLastDims(1);
   
    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, loss_shape,
                                                     &output_tensor));

    // Do the computation.
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));

    RepulsiveLossFunctor<Device, T, S>()(
        context->eigen_device<Device>(),
        input_tensor.tensor<T,4>(),
        rle_tensor.tensor<S,4>(),
        mean_tensor.tensor<T,3>(),
        output_tensor->flat<T>().setZero().data());
  }
};

// Register the CPU kernels.
#define REGISTER_CPU_RL(T,S)                                                \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("RepulsiveLoss").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      RepulsiveLossOp<CPUDevice, T,S>);
REGISTER_CPU_RL(float,int32);


REGISTER_OP("DerivedRepulsiveLoss")
  .Attr("T: {float32, float64}")
  .Attr("S: {int32, int64}")
  .Input("embeddings: T")
  .Input("rle_mask: S")
  .Input("averages: T")
  .Input("gradient_in: T")
  .Output("gradient_out: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle input;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));

    shape_inference::ShapeHandle rle_input;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &rle_input));

    shape_inference::DimensionHandle rle_end_dim;
    TF_RETURN_IF_ERROR(c->WithValue(c->DimKnownRank(rle_input, 3), 2, &rle_end_dim));

    shape_inference::ShapeHandle mean_input;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 3, &mean_input));

    shape_inference::ShapeHandle up_input;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 3, &up_input));

    c->set_output(0,input);
    return Status::OK();
  }
);


// CPU specialization of actual computation.
template <typename Device, typename T, typename S>
struct DerivedRepulsiveLossFunctor {
  void operator()(const Device& d,
    const Eigen::Tensor<T,4, Eigen::RowMajor> logits_embeddings, 
    const Eigen::Tensor<S,4, Eigen::RowMajor> rle_mask, 
    const Eigen::Tensor<T,3, Eigen::RowMajor> mean_embeddings,
    const Eigen::Tensor<T,3, Eigen::RowMajor> grad_up,
    T* grad_down_outputs) {
    
    const int batch_size = logits_embeddings.dimension(0);
    const int size_x = logits_embeddings.dimension(1);
    const int size_y = logits_embeddings.dimension(2);
    const int embedding_size = logits_embeddings.dimension(3);
    const int embedding_count = rle_mask.dimension(1);

    Eigen::array<int, 4> window_logits(1,1,1,embedding_size);
    Eigen::array<int, 4> offset_logits(0,0,0,0);

    Eigen::array<int, 3> window_mean(1,1,embedding_size);
    Eigen::array<int, 3> offset_mean(0,0,0);
    Eigen::array<int, 1> dims2({2});
    Eigen::array<int, 1> dims3({2});

    // Create the average embedding for each mask
    for (int batch = 0; batch < batch_size; ++batch) {
      offset_logits[0] = batch;
      offset_mean[0] = batch;

       for(int embedding_current = 0; embedding_current < embedding_count; ++embedding_current) {
        for(int x = 0; x < size_x; ++x) {
          offset_logits[1] = x;
          S min_y = std::max(0,rle_mask(batch,embedding_current,x,0));
          S max_y = std::min(size_y,rle_mask(batch,embedding_current,x,1));

          Eigen::Tensor<T, 4, Eigen::RowMajor> average_difference(1,1,1,embedding_size);

          for(int y = min_y; y < max_y; ++y) {
            average_difference.setZero();
            offset_logits[2] = y;
            for(int embedding_other = 0; embedding_other < embedding_current; ++embedding_other) {
              offset_mean[1] = embedding_other;
              auto mean_scratch = mean_embeddings.slice(offset_mean,window_mean);
              auto my_mean_embedding = (mean_scratch.reshape(window_logits)); // Delays the computation and allows the compiler to optimize
              auto middle = (logits_embeddings.slice(offset_logits,window_logits) - my_mean_embedding);
              average_difference = average_difference + middle;
            }
            for(int embedding_other = embedding_current + 1; embedding_other < embedding_count; ++embedding_other) {
              offset_mean[1] = embedding_other;
              auto mean_scratch = mean_embeddings.slice(offset_mean,window_mean);
              auto my_mean_embedding = (mean_scratch.reshape(window_logits));
              auto middle = (logits_embeddings.slice(offset_logits,window_logits) - my_mean_embedding);
              average_difference = average_difference + middle;
            }
            average_difference = (1.0/(embedding_count-1))*average_difference;
            memcpy(
              grad_down_outputs + (batch*size_x*size_y + x*size_y + y)*embedding_size,
              average_difference.data(),
              sizeof(T)*embedding_size);
          }
          
        } 
      }
    }
  }
};

template <typename Device, typename T, typename S>
class DerivedRepulsiveLossOp : public OpKernel {
 public:
  explicit DerivedRepulsiveLossOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    const Tensor& rle_tensor = context->input(1);
    const Tensor& mean_tensor = context->input(2);
    const Tensor& grad_up_tensor = context->input(3);

    TensorShape input_shape = input_tensor.shape();
    TensorShape loss_shape = input_shape;
    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, loss_shape,
                                                     &output_tensor));
    // Do the computation.
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));
    
    DerivedRepulsiveLossFunctor<Device, T, S>()(
        context->eigen_device<Device>(),
        input_tensor.tensor<T,4>(),
        rle_tensor.tensor<S,4>(),
        mean_tensor.tensor<T,3>(),
        grad_up_tensor.tensor<T,3>(),
        output_tensor->flat<T>().setZero().data());
    
  }
};

// Register the CPU kernels.
#define REGISTER_CPU_DRL(T,S)                                                \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("DerivedRepulsiveLoss").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      DerivedRepulsiveLossOp<CPUDevice, T,S>);
REGISTER_CPU_DRL(float,int32);
