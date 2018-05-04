#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;



/*
For a image mask with run length encoding with shape (batch_size,num_masks,dim1,2) where 
the last dimension holds the dim2_min, dim2_max coordinates, it computes the average 
value of the input `embedding` for each mask. 
*/
REGISTER_OP("MaskedAverages")
  .Attr("T: {float32, float64}")
  .Attr("S: {int32, int64}")
  .Input("embeddings: T")
  .Input("rle_mask: S")
  .Output("masked_averages: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
  shape_inference::ShapeHandle input;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));

  shape_inference::ShapeHandle rle_input;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &rle_input));

  shape_inference::DimensionHandle rle_end_dim;
  TF_RETURN_IF_ERROR(c->WithValue(c->DimKnownRank(rle_input, 3), 2, &rle_end_dim));

  shape_inference::ShapeHandle mean_shape;
  TF_RETURN_IF_ERROR(c->Subshape(rle_input,0,-1,&mean_shape));
  TF_RETURN_IF_ERROR(c->ReplaceDim(mean_shape, -1, c->DimKnownRank(input, 3),
                      &mean_shape));
  c->set_output(0,mean_shape);
  return Status::OK();
});


// Sernel_swarmEmbeddingLoss.cc
// #include "swarmEmbeddingLoss.h"


// CPU specialization of actual computation.
template <typename Device, typename T, typename S>
struct getMaskedAveragesFunctor {
  void operator()(const Device& d,
  		const Eigen::Tensor<T,4, Eigen::RowMajor> logits_embeddings, 
  		const Eigen::Tensor<S,4, Eigen::RowMajor> rle_mask, 
  		T* mean_scratch_buffer) {
  	
	const int batch_size = logits_embeddings.dimension(0);
	const int size_x = logits_embeddings.dimension(1);
	const int size_y = logits_embeddings.dimension(2);
	const int embedding_size = logits_embeddings.dimension(3);
	const int embedding_count = rle_mask.dimension(1);

    Eigen::array<int, 4> window(1,1,1,embedding_size);
    Eigen::array<int, 4> offset_logits(0,0,0,0);
    Eigen::array<int, 1> dims({2});

    Eigen::Tensor<T,3, Eigen::RowMajor> mean_scratch(1,1,embedding_size);
    // Create the average embedding for each mask
    for (int batch = 0; batch < batch_size; ++batch) {
    	offset_logits[0] = batch;
    	
    	for(int embedding = 0; embedding < embedding_count; ++embedding) {
		    mean_scratch.setZero();
	    	int num_elements = 0;
	    	
	    	for(int x = 0; x < size_x; ++x){
    			offset_logits[1] = x;
    			S min_y = std::max(0,rle_mask(batch,embedding,x,0));
    			S max_y = std::min(size_y,rle_mask(batch,embedding,x,1));
    			if(max_y > min_y){
	    			window[2] = max_y - min_y;
	    			offset_logits[2] = min_y;
	    			
			    	mean_scratch += (logits_embeddings.slice(offset_logits,window)).sum(dims);
			    	num_elements += max_y - min_y;
		    	}    			
    		}
    		if(num_elements > 0)
	    		mean_scratch = (1.0f/num_elements)* mean_scratch;
		    memcpy(mean_scratch_buffer + batch*embedding*embedding_size + embedding*embedding_size,mean_scratch.data(),sizeof(T)*embedding_size);
    	}
    }
  }
};


//REGISTER_CPU(double,int32);
// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T, typename S>
class MaskedAveragesOp : public OpKernel {
 public:
  explicit MaskedAveragesOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    const Tensor& rle_tensor = context->input(1);


    TensorShape input_shape = input_tensor.shape();
    TensorShape rle_shape = rle_tensor.shape();
    TensorShape mean_shape = input_shape;
    mean_shape.RemoveDimRange(1,-2);
    mean_shape.InsertDim(1,rle_shape.dim_size(1));

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, mean_shape,
                                                     &output_tensor));
    // Do the computation.
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));

    getMaskedAveragesFunctor<Device, T, S>()(
        context->eigen_device<Device>(),
        input_tensor.tensor<T,4>(),
        rle_tensor.tensor<S,4>(),
        output_tensor->flat<T>().data());
  }
};

// Register the CPU kernels.
#define REGISTER_CPU_MA(T,S)                                          			\
  REGISTER_KERNEL_BUILDER(                                       			\
      Name("MaskedAverages").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      MaskedAveragesOp<CPUDevice, T,S>);
REGISTER_CPU_MA(float,int32);