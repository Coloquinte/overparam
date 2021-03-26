r"""
Overparameterization from https://arxiv.org/abs/1811.10495
"""
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.nn.init import kaiming_uniform_
from typing import Any, TypeVar


def _overparameterization(expand_w, kernel_w, reduce_w):
    return torch.matmul(torch.matmul(reduce_w, kernel_w), expand_w)


class OverParameterization(object):
    name: str
    expansion: int

    def __init__(self, name: str) -> None:
        self.name = name

    # TODO Make return type more specific
    def compute_weight(self, module: Module) -> Any:
        expand_w = getattr(module, self.name + '_expand')
        kernel_w = getattr(module, self.name + '_kernel')
        reduce_w = getattr(module, self.name + '_reduce')
        return _overparameterization(expand_w, kernel_w, reduce_w)

    @staticmethod
    def apply(module, name: str, expansion: float) -> 'OverParameterization':
        if expansion < 1.0:
            raise RuntimeError("Cannot make overparameterization with expansion less "
                               "than 1.0; got {}".format(expansion))
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, OverParameterization) and hook.name == name:
                raise RuntimeError("Cannot register two overparameterization hooks on "
                                   "the same parameter {}".format(name))


        fn = OverParameterization(name)

        weight = getattr(module, name)
        # remove w from parameter list
        del module._parameters[name]

        in_channels = weight.shape[1]
        out_channels = weight.shape[0]

        expand_w = torch.Tensor(int(expansion*in_channels), in_channels)
        kaiming_uniform_(expand_w)
        reduce_w = torch.Tensor(out_channels, int(expansion*out_channels))
        kaiming_uniform_(reduce_w)

        # TODO: find a kernel so that the expanded layer matches the original values
        # This could be done with two least-square solutions
        kernel_w = torch.Tensor(int(expansion*out_channels), int(expansion*in_channels))
        kaiming_uniform_(kernel_w)

        # add the new parameters
        module.register_parameter(name + '_expand', Parameter(expand_w.data))
        module.register_parameter(name + '_kernel', Parameter(kernel_w.data))
        module.register_parameter(name + '_reduce', Parameter(reduce_w.data))
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module: Module) -> None:
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_expand']
        del module._parameters[self.name + '_kernel']
        del module._parameters[self.name + '_reduce']
        setattr(module, self.name, Parameter(weight.data))

    def __call__(self, module: Module, inputs: Any) -> None:
        setattr(module, self.name, self.compute_weight(module))


T_module = TypeVar('T_module', bound=Module)

def overparameterization(module: T_module, name: str = 'weight', expansion: float = 2.0) -> T_module:
    r"""Applies overparameterization to a parameter in the given module.

    .. math::
         \mathbf{w} = \mathbf{a}\mathbf{k}\mathbf{b}

    Overparameterization is a reparameterization that replaces a linear layer by
    several consecutive linear layers. This replaces the parameter specified
    by :attr:`name` (e.g. ``'weight'``) with three parameters: one that expands
    the number of channels (e.g. ``'weight_expand'``) and one that shrinks them
    (e.g. ``'weight_reduce'``) with an third parameter in the middle
    (e.g. ``'weight_kernel'``). Overparameterization is implemented via a hook
    that recomputes the weight tensor before every :meth:`~Module.forward` call.

    See https://arxiv.org/abs/1811.10495

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter
        expansion (float, optional): expansion ratio

    Returns:
        The original module with the overparameterization hook

    Example::

        >>> m = overparameterization(nn.Linear(20, 40), name='weight', expansion=4)
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_expand.size()
        torch.Size([80, 20])
        >>> m.weight_kernel.size()
        torch.Size([160, 80])
        >>> m.weight_reduce.size()
        torch.Size([40, 160])
    """
    OverParameterization.apply(module, name, expansion)
    return module



#[docs]
def remove_overparameterization(module: T_module, name: str = 'weight') -> T_module:
    r"""Removes the overparameterization from a module.

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter

    Example:
        >>> m = overparameterization(nn.Linear(20, 40))
        >>> remove_overparameterization(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, OverParameterization) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("overparameterization of '{}' not found in {}"
                     .format(name, module))

