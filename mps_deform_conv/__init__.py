"""
MPS Deformable Convolution - torchvision.ops.deform_conv2d for Apple Silicon

Drop-in replacement for torchvision's deform_conv2d that works on MPS devices.
Used in DETR, Deformable DETR, BasicVSR++, EDVR, optical flow models, etc.
"""

import torch
from torch import nn, Tensor
from typing import Optional, Tuple
import math

__version__ = "0.1.1"


def _load_library():
    """Load the native extension."""
    try:
        from mps_deform_conv import _C as _lib
        return _lib
    except ImportError:
        import os
        from torch.utils.cpp_extension import load

        src_dir = os.path.join(os.path.dirname(__file__), "csrc")
        _lib = load(
            name="mps_deform_conv",
            sources=[os.path.join(src_dir, "deform_conv2d_mps.mm")],
            extra_cflags=["-std=c++17"],
            extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
            verbose=False,
        )
        return _lib


_lib_cache = None


def _get_lib():
    global _lib_cache
    if _lib_cache is None:
        _lib_cache = _load_library()
    return _lib_cache


class _DeformConv2dFunction(torch.autograd.Function):
    """Autograd function for deformable convolution with full backward pass."""

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        offset: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        dilation: Tuple[int, int],
        mask: Optional[Tensor],
        n_weight_grps: int,
        n_offset_grps: int,
        use_mask: bool,
    ) -> Tensor:
        lib = _get_lib()

        stride_h, stride_w = stride
        pad_h, pad_w = padding
        dilation_h, dilation_w = dilation

        bias_t = bias if bias is not None else torch.empty(0, device=input.device, dtype=input.dtype)
        mask_t = mask if mask is not None else torch.empty(0, device=input.device, dtype=input.dtype)

        output = lib.deform_conv2d_forward(
            input, weight, offset, mask_t, bias_t,
            stride_h, stride_w, pad_h, pad_w,
            dilation_h, dilation_w,
            n_weight_grps, n_offset_grps, use_mask,
        )

        ctx.save_for_backward(input, offset, weight, mask_t if use_mask else None)
        ctx.bias_exists = bias is not None
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.n_weight_grps = n_weight_grps
        ctx.n_offset_grps = n_offset_grps
        ctx.use_mask = use_mask

        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        input, offset, weight, mask = ctx.saved_tensors
        stride_h, stride_w = ctx.stride
        pad_h, pad_w = ctx.padding
        dilation_h, dilation_w = ctx.dilation
        n_weight_grps = ctx.n_weight_grps
        n_offset_grps = ctx.n_offset_grps
        use_mask = ctx.use_mask

        batch_sz, n_in_channels, in_h, in_w = input.shape
        n_out_channels = weight.size(0)
        kernel_h, kernel_w = weight.size(2), weight.size(3)

        out_h = (in_h + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
        out_w = (in_w + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1

        lib = _get_lib()
        mask_t = mask if mask is not None else torch.empty(0, device=input.device, dtype=input.dtype)

        # Compute columns via Metal im2col (fast)
        columns = lib.deformable_im2col(
            input, offset, mask_t,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w,
            dilation_h, dilation_w,
            n_offset_grps, use_mask
        )

        # grad_output: [B, C_out, H_out, W_out] -> [C_out, B*H_out*W_out]
        grad_out_flat = grad_output.permute(1, 0, 2, 3).contiguous().view(n_out_channels, -1)

        # Weight gradient: dL/dW = grad_out @ columns.T
        grad_weight = None
        if weight.requires_grad:
            grad_weight = torch.mm(grad_out_flat, columns.t()).view_as(weight)

        # Bias gradient
        grad_bias = None
        if ctx.bias_exists:
            grad_bias = grad_output.sum(dim=(0, 2, 3))

        # For input/offset gradients, compute grad_col = W.T @ grad_out
        weight_flat = weight.view(n_out_channels, -1)
        grad_col = torch.mm(weight_flat.t(), grad_out_flat)

        # Input gradient via Metal col2im kernel
        grad_input = None
        if input.requires_grad:
            grad_input = lib.deform_conv2d_backward_input(
                grad_col, input, offset, mask_t,
                in_h, in_w,
                stride_h, stride_w,
                pad_h, pad_w,
                dilation_h, dilation_w,
                kernel_h, kernel_w,
                n_offset_grps, use_mask
            )

        # Offset/mask gradient via Metal kernel
        grad_offset = None
        grad_mask = None
        if offset.requires_grad or (use_mask and mask is not None):
            grad_offset, grad_mask = lib.deform_conv2d_backward_offset_mask(
                grad_col, input, offset, mask_t,
                kernel_h, kernel_w,
                stride_h, stride_w,
                pad_h, pad_w,
                dilation_h, dilation_w,
                n_offset_grps, use_mask
            )
            if not offset.requires_grad:
                grad_offset = None
            if not use_mask or mask is None:
                grad_mask = None

        return grad_input, grad_offset, grad_weight, grad_bias, None, None, None, grad_mask, None, None, None


def deform_conv2d(
    input: Tensor,
    offset: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Deformable 2D convolution.

    Args:
        input: Input tensor (N, C_in, H, W)
        offset: Offset tensor (N, 2*K*K*offset_groups, H_out, W_out)
        weight: Weight tensor (C_out, C_in/groups, K, K)
        bias: Optional bias (C_out,)
        stride: Convolution stride
        padding: Input padding
        dilation: Kernel dilation
        mask: Optional DCNv2 mask (N, K*K*offset_groups, H_out, W_out)

    Returns:
        Output tensor (N, C_out, H_out, W_out)
    """
    if not input.device.type == "mps":
        raise ValueError(f"Input must be on MPS device, got {input.device}")

    kernel_h, kernel_w = weight.size(2), weight.size(3)
    in_channels = input.size(1)

    weight_in_channels = weight.size(1)
    n_weight_grps = in_channels // weight_in_channels

    offset_channels = offset.size(1)
    n_offset_grps = offset_channels // (2 * kernel_h * kernel_w)

    use_mask = mask is not None

    return _DeformConv2dFunction.apply(
        input, offset, weight, bias,
        stride, padding, dilation, mask,
        n_weight_grps, n_offset_grps, use_mask,
    )


class DeformConv2d(nn.Module):
    """Deformable 2D Convolution Module."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        offset_groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = groups
        self.offset_groups = offset_groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor, offset: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return deform_conv2d(
            input, offset, self.weight, self.bias,
            self.stride, self.padding, self.dilation, mask,
        )


class ModulatedDeformConv2d(nn.Module):
    """Modulated Deformable Convolution (DCNv2) with built-in offset prediction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        offset_groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = groups
        self.offset_groups = offset_groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        kernel_h, kernel_w = self.kernel_size
        offset_channels = 2 * kernel_h * kernel_w * offset_groups
        mask_channels = kernel_h * kernel_w * offset_groups

        self.offset_conv = nn.Conv2d(
            in_channels,
            offset_channels + mask_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True,
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

    def forward(self, input: Tensor) -> Tensor:
        kernel_h, kernel_w = self.kernel_size
        offset_channels = 2 * kernel_h * kernel_w * self.offset_groups

        offset_mask = self.offset_conv(input)
        offset = offset_mask[:, :offset_channels]
        mask = torch.sigmoid(offset_mask[:, offset_channels:])

        return deform_conv2d(
            input, offset, self.weight, self.bias,
            self.stride, self.padding, self.dilation, mask,
        )


def is_available() -> bool:
    """Check if MPS is available."""
    return torch.backends.mps.is_available()
