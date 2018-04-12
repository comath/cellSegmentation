# Cell Segmentation

I had to write a new tensorflow op for this as if we tried to implement the chosen loss function at the python level we will have far too many little ops glued together. This is numerically unstable, I don't know why but I think it's because the loss is highly unstable due to the numerous averaging steps taken. Even if that weren't true, it's slow and unwieldy.   


#Shape Inference: 

There are two main questions that need to be answered:
- How does it work? 
- If I want an input of a certain shape and an output that is computable from that shape how do i do it?

You have to define a lambda, that computes the shape using ''shape_inference.h''. These are computed using the InferenceContext class object that is passed to the lambda. All computational routines have a pointer as the last argument so that you can pass as a reference the output object. For example, if I wanted a op whose output whose dimensions are all but the last element of an input op we would use the following: 

'''
    SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle loss_shape;								
      TF_RETURN_IF_ERROR(c->Subshape(c->input(0),0,-1,&loss_shape));							
      c->set_output(0,loss_shape);
      return Status::OK();
    });
'''

To walk this though, step by step, the first line is the function handle that the REGISTER_OP calls to determine the output shapes. It starts a lambda that's expressed in the following few lines. The next line creates a ShapeHandle for the output shape that's passed to the next line. ''c'' is the inference context, holding all the information needed to determine the type and shape of the input and output tensors. It has several functions attached to it that don't interact with it's internal data, such as Subshape. This function takes in the start and end that you want to keep from the first input to the op, ''c-input(0)'', and outputs it into ''loss_shape''. It actually returns a ''Status'' (defined in lib/core/status.h). The ''TF_RETURN_IF_ERROR''  is for tensorflow's internal error handling methods. We finally set the first output to have the same shape as the loss_shape and return an OK status indicating that there's been no errors.

