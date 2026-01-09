# Copyright (c) 2023, Tri Dao.
# Wrapper for causal_conv1d_cuda C++ functions

import torch
import os
import sys

# Import causal_conv1d_cuda from the same directory
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

try:
    import causal_conv1d_cuda
except ImportError:
    # Try importing from parent directory (where .so might be)
    _parent_dir = os.path.dirname(_current_dir)
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    import causal_conv1d_cuda


class CausalConv1dFwdFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None, activation=None):
        if activation not in [None, "silu", "swish"]:
            raise NotImplementedError("activation must be None, silu, or swish")
        if x.stride(2) != 1 and x.stride(1) != 1:
            x = x.contiguous()
        bias = bias.contiguous() if bias is not None else None
        ctx.save_for_backward(x, weight, bias)
        ctx.activation = activation in ["silu", "swish"]
        out = causal_conv1d_cuda.causal_conv1d_fwd(x, weight, bias, ctx.activation)
        return out

    @staticmethod
    def backward(ctx, dout):
        x, weight, bias = ctx.saved_tensors
        if dout.stride(2) != 1 and dout.stride(1) != 1:
            dout = dout.contiguous()
        dx, dweight, dbias = causal_conv1d_cuda.causal_conv1d_bwd(
            x, weight, bias, dout, None, ctx.activation
        )
        return dx, dweight, dbias if bias is not None else None, None


class CausalConv1dBwdFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        raise NotImplementedError("CausalConv1dBwdFunction should only be used in backward pass")

    @staticmethod
    def backward(ctx, *args):
        # This is typically called from the forward function's backward
        pass


# Export the functions that mamba_ssm expects
causal_conv1d_fwd_function = CausalConv1dFwdFunction.apply
causal_conv1d_bwd_function = CausalConv1dBwdFunction.apply

