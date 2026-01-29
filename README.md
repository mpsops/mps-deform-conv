# MPS Deformable Convolution

Deformable Convolution 2D for PyTorch on Apple Silicon (M1/M2/M3/M4).

**Drop-in replacement** for `torchvision.ops.deform_conv2d` that actually works on MPS.

## Why?

Deformable convolutions are used everywhere:
- **Detection**: DETR, Deformable DETR, mmdetection models
- **Video**: BasicVSR++, EDVR, optical flow models
- **Segmentation**: Mask R-CNN with DCN backbones

But torchvision's implementation is **CUDA-only**. On Mac you get:
```
NotImplementedError: deform_conv2d not implemented for MPS
```

This package provides a native Metal implementation.

## Installation

```bash
pip install mps-deform-conv
```

Or from source:

```bash
git clone https://github.com/imperatormk/mps-deform-conv
cd mps-deform-conv
pip install -e .
```

## Quick Start

### Basic Usage

```python
import torch
from mps_deform_conv import deform_conv2d

# Input: (batch, channels, height, width)
input = torch.randn(1, 64, 32, 32, device='mps')

# Weight: (out_channels, in_channels, kernel_h, kernel_w)
weight = torch.randn(64, 64, 3, 3, device='mps')

# Offset: (batch, 2 * kernel_h * kernel_w, out_h, out_w)
# 2 values (dy, dx) for each position in the 3x3 kernel
offset = torch.randn(1, 2*9, 32, 32, device='mps')

# Run deformable convolution
output = deform_conv2d(input, offset, weight, padding=(1, 1))
```

### DeformConv2d Module

```python
from mps_deform_conv import DeformConv2d

# Create layer
conv = DeformConv2d(
    in_channels=64,
    out_channels=128,
    kernel_size=3,
    padding=1
).to('mps')

# Forward pass requires input and offset
x = torch.randn(1, 64, 32, 32, device='mps')
offset = torch.randn(1, 2*9, 32, 32, device='mps')
output = conv(x, offset)
```

### ModulatedDeformConv2d (DCNv2)

Includes the offset predictor - a complete conv layer replacement:

```python
from mps_deform_conv import ModulatedDeformConv2d

# DCNv2 with internal offset/mask prediction
conv = ModulatedDeformConv2d(
    in_channels=64,
    out_channels=128,
    kernel_size=3,
    padding=1
).to('mps')

# Just pass input - offsets learned internally
x = torch.randn(1, 64, 32, 32, device='mps')
output = conv(x)
```

## API Reference

### `deform_conv2d(input, offset, weight, bias, stride, padding, dilation, mask)`

Functional interface matching `torchvision.ops.deform_conv2d`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | Tensor | Input tensor (N, C_in, H, W) |
| `offset` | Tensor | Offset tensor (N, 2\*K\*K\*groups, H_out, W_out) |
| `weight` | Tensor | Weight tensor (C_out, C_in/groups, K, K) |
| `bias` | Tensor | Optional bias (C_out,) |
| `stride` | tuple | Convolution stride (default: (1, 1)) |
| `padding` | tuple | Input padding (default: (0, 0)) |
| `dilation` | tuple | Kernel dilation (default: (1, 1)) |
| `mask` | Tensor | Optional DCNv2 mask (N, K\*K\*groups, H_out, W_out) |

### `DeformConv2d`

Module wrapping `deform_conv2d`. Takes `input` and `offset` in forward.

### `ModulatedDeformConv2d`

Self-contained DCNv2 module with internal offset prediction. Takes only `input` in forward.

## How It Works

Standard convolution samples on a fixed grid:
```
[•] [•] [•]
[•] [x] [•]
[•] [•] [•]
```

Deformable convolution learns offsets to sample from arbitrary positions:
```
    [•]
[•]     [•]
    [x]     [•]
[•]     [•]
    [•]
```

This lets the network adapt its receptive field to the input content - useful for detecting objects at different scales, handling geometric transformations, etc.

## Compatibility

- **PyTorch**: 2.0+
- **macOS**: 12.0+ (Monterey)
- **Hardware**: Apple Silicon (M1/M2/M3/M4)

## Features

- Full forward and backward pass (training supported)
- Gradients verified against torchvision (< 0.00001 error)
- fp32 and fp16 supported
- Grouped convolutions supported

## Credits

- [torchvision](https://github.com/pytorch/vision) - Reference CUDA implementation
- [Deformable Convolutional Networks](https://arxiv.org/abs/1703.06211) - Original paper
- [Deformable ConvNets v2](https://arxiv.org/abs/1811.11168) - Modulated DCN

## License

MIT
