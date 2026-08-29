# Learnings
## Encountered functions

* [torch.stack](https://docs.pytorch.org/docs/2.13/generated/torch.stack.html): To stack tensors along a new dimension
* [torch.repeat](https://docs.pytorch.org/docs/2.13/generated/torch.repeat.html): To repeat a tensor along a new dimension
* [torch.unsqueeze](https://docs.pytorch.org/docs/2.13/generated/torch.unsqueeze.html): To modify the dimensions of a tensor
* [torch.expand](https://docs.pytorch.org/docs/2.13/generated/torch.expand.html): To add more values to a tensor based on the tensor's existing values
* [functional.conv2d](https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.conv2d.html): To perform a 2D convolution on a tensor. Takes in a NCHW where N is the batch size, C is the number of channels, H is the height, and W is the width.
* [torch.arange](https://docs.pytorch.org/docs/2.13/generated/torch.arange.html): To create a 1D tensor of values from 0 to a given number
* [torch.hypot](https://docs.pytorch.org/docs/2.13/generated/torch.hypot.html): To compute the hypotenuse of two tensors. Use it for computing distances between points in 2D space.
* [torch.meshgrid](https://docs.pytorch.org/docs/2.13/generated/torch.meshgrid.html): To create a 2D grid of points out of two 1D tensors. Look at `hello-circle.py` for an example.
* [torch.isclose](https://docs.pytorch.org/docs/2.13/generated/torch.isclose.html): To compare two tensors element-wise and check if they are close within a given tolerance. Basically epsiloned comparison.

## Interesting tricks
* You can unsqueeze when indexing by adding a None in the index position to expand the dimensions of the tensor before indexing. Skipping the need for `torch.unsqueeze`. For example, look at `checkerboard.py`.
