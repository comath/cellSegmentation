# Cell Segmentation

I had to write a new tensorflow op for this as if we tried to implement the chosen loss function at the python level we will have far too many little ops glued together. This is numerically unstable, I don't know why but I think it's because the loss is highly unstable due to the numerous averaging steps taken. Even if that weren't true, it's slow and unwieldy.   


# Registering the Operation

## How does it work? Why?

``
REGISTER_OP("SwarmEmbeddingLoss")
	.Attr("T: {float32, float64}")
	.Attr("S: {int32, int64}")
    .Input("embedding: T")
    .Input("rle_mask: S")
    .Output("swarm_loss: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
});

``

This registers the op with tensorflow. This op has 2 inputs and 1 output. The `embedding` input has to be either a float or a double, and the `rle_mask` has to be either a int or a long. We also determine the output as a float. The current description of this in the [tensorflow documentation](https://www.tensorflow.org/extend/adding_an_op#list_inputs_and_outputs). This is all I've needed to do with this so far. If I run into more issues I will update this later. However, the `SetShapeFn` is the part I need to look into more and is probably more important for most custom ops so it's featured here. 

## How do I make a certain output?

You have to define a lambda, that computes the shape using `shape_inference.h`. These are computed using the InferenceContext class object that is passed to the lambda. All computational routines have a pointer as the last argument so that you can pass as a reference the output object. For example, if I wanted a op whose output whose dimensions are all but the last element of an input op we would use the following: 

```
    SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle loss_shape;								
      TF_RETURN_IF_ERROR(c->Subshape(c->input(0),0,-1,&loss_shape));							
      c->set_output(0,loss_shape);
      return Status::OK();
    });
```

To walk this though, step by step, the first line is the function handle that the REGISTER_OP calls to determine the output shapes. It starts a lambda that's expressed in the following few lines. The next line creates a ShapeHandle for the output shape that's passed to the next line. `c` is the inference context, holding all the information needed to determine the type and shape of the input and output tensors. It has several functions attached to it that don't interact with it's internal data, such as Subshape. This function takes in the start and end that you want to keep from the first input to the op, `c-input(0)`, and outputs it into `loss_shape`. It actually returns a `Status` (defined in lib/core/status.h). The `TF_RETURN_IF_ERROR`  is for tensorflow's internal error handling methods. We finally set the first output to have the same shape as the loss_shape and return an OK status indicating that there's been no errors.

## How do I require a certain input?

```
    SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));

      ShapeHandle rle_input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &rle_input));

      DimensionHandle rle_end_dim;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(rle_input, 3), 2, &rle_end_dim));
      return Status::OK();
    });
```

I want the input to always be 4 dimensional, I want a batch size, and an image with an X and a Y dimension that each holds N dimensions. So, the input has to be a rank 4 tensor. To do this we have several 
The above two can be combines and if we do so we can use `input` as a placeholder for the first input tensor.


# Output

## How do I allocate for output?
```
Tensor* output_tensor = NULL;
OP_REQUIRES_OK(context, context->allocate_output(0, loss_shape,
                                                 &output_tensor));
```

The output tensors are already set up in the `OpKernelContext`, with the correct type though they are not allocated and have no given shape. For the first tensor (the 0), we pass a shape, and a reference to a pointer. I don't know if it error checks to ensure that the output loss is of the correct shape, but don't screw it up!

## How do I allocate for scratch space?

```
Tensor mean_tensor_temp;
OP_REQUIRES_OK(context, context->allocate_temp(input_tensor.dtype(), 
								mean_shape_temp, &mean_tensor_temp));
```

This is the temporary version of above. The type cannot be inferred from context, so it needs to be passed. There's an enumeration in `types.pb.h` that holds all the types. The most useful ones are `DT_FLOAT` and `DT_INT32`. Remember the `Tensor` object is just a reference, so this function provides the tensor with the type and the shape and then it allocates the appropriate internal buffer.