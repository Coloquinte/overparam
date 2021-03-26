# overparam

Over-parameterization of linear and convolution layers in Pytorch.
Overparameterization replaces a linear layer with several larger layers during training, and collapses them at inference time.

It implements the ideas presented in [ExpandNets: Linear Over-parameterization to Train Compact Convolutional Networks](https://arxiv.org/abs/1811.10495).

This code is based on the [Pytorch code for weight normalization](https://pytorch.org/docs/stable/generated/torch.nn.utils.weight_norm.html?highlight=weight%20norm#torch.nn.utils.weight_norm).
It adds the overparameterization as a forward hook, so that the performance penalty during training is minimal.


