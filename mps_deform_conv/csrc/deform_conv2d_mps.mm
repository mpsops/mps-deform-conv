/**
 * Deformable Convolution 2D - MPS Backend
 *
 * Port of torchvision's deform_conv2d for Apple Silicon.
 * Implements forward and backward passes using Metal compute shaders.
 */

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

// =============================================================================
// Metal Kernel Cache
// =============================================================================

static id<MTLDevice> g_device = nil;
static id<MTLLibrary> g_library = nil;
static id<MTLComputePipelineState> g_im2col_fp32 = nil;
static id<MTLComputePipelineState> g_im2col_fp16 = nil;
static id<MTLComputePipelineState> g_col2im_fp32 = nil;
static id<MTLComputePipelineState> g_col2im_coord_fp32 = nil;

static NSString* get_metal_source() {
    return @R"(
#include <metal_stdlib>
using namespace metal;

// Bilinear interpolation
template<typename T>
inline T bilinear_interpolate(
    device const T* in,
    int height,
    int width,
    T h,
    T w
) {
    if (h <= T(-1) || T(height) <= h || w <= T(-1) || T(width) <= w) {
        return T(0);
    }

    int h_low = int(floor(h));
    int w_low = int(floor(w));
    int h_high = h_low + 1;
    int w_high = w_low + 1;

    T lh = h - T(h_low);
    T lw = w - T(w_low);
    T hh = T(1) - lh;
    T hw = T(1) - lw;

    T v1 = (h_low >= 0 && w_low >= 0) ? in[h_low * width + w_low] : T(0);
    T v2 = (h_low >= 0 && w_high <= width - 1) ? in[h_low * width + w_high] : T(0);
    T v3 = (h_high <= height - 1 && w_low >= 0) ? in[h_high * width + w_low] : T(0);
    T v4 = (h_high <= height - 1 && w_high <= width - 1) ? in[h_high * width + w_high] : T(0);

    return hh * hw * v1 + hh * lw * v2 + lh * hw * v3 + lh * lw * v4;
}

// Coordinate weight for backward
template<typename T>
inline T get_coordinate_weight(
    device const T* im_data,
    int height,
    int width,
    T y,
    T x,
    bool is_y_direction
) {
    int y_l = int(floor(y));
    int x_l = int(floor(x));
    int y_h = y_l + 1;
    int x_h = x_l + 1;

    bool valid_y_l = 0 <= y_l && y_l < height;
    bool valid_y_h = 0 <= y_h && y_h < height;
    bool valid_x_l = 0 <= x_l && x_l < width;
    bool valid_x_h = 0 <= x_h && x_h < width;

    T v_yx = (valid_y_l && valid_x_l) ? im_data[y_l * width + x_l] : T(0);
    T v_yX = (valid_y_l && valid_x_h) ? im_data[y_l * width + x_h] : T(0);
    T v_Yx = (valid_y_h && valid_x_l) ? im_data[y_h * width + x_l] : T(0);
    T v_YX = (valid_y_h && valid_x_h) ? im_data[y_h * width + x_h] : T(0);

    if (is_y_direction) {
        T dx = x - T(x_l);
        return dx * (v_YX - v_yX) + (T(1) - dx) * (v_Yx - v_yx);
    } else {
        T dy = y - T(y_l);
        return dy * (v_YX - v_Yx) + (T(1) - dy) * (v_yX - v_yx);
    }
}

// Forward: im2col with deformable offsets
kernel void deformable_im2col_fp32(
    device const float* input       [[buffer(0)]],
    device const float* offset      [[buffer(1)]],
    device const float* mask        [[buffer(2)]],
    device float* columns           [[buffer(3)]],
    constant int& height            [[buffer(4)]],
    constant int& width             [[buffer(5)]],
    constant int& weight_h          [[buffer(6)]],
    constant int& weight_w          [[buffer(7)]],
    constant int& pad_h             [[buffer(8)]],
    constant int& pad_w             [[buffer(9)]],
    constant int& stride_h          [[buffer(10)]],
    constant int& stride_w          [[buffer(11)]],
    constant int& dilation_h        [[buffer(12)]],
    constant int& dilation_w        [[buffer(13)]],
    constant int& batch_sz          [[buffer(14)]],
    constant int& n_in_channels     [[buffer(15)]],
    constant int& n_offset_grps     [[buffer(16)]],
    constant int& out_h             [[buffer(17)]],
    constant int& out_w             [[buffer(18)]],
    constant int& use_mask          [[buffer(19)]],
    uint gid [[thread_position_in_grid]]
) {
    int n = n_in_channels * out_h * out_w * batch_sz;
    if (int(gid) >= n) return;

    int index = int(gid);
    int out_x = index % out_w;
    int out_y = (index / out_w) % out_h;
    int out_b = (index / (out_w * out_h)) % batch_sz;
    int in_c = index / (out_w * out_h * batch_sz);
    int out_c = in_c * weight_h * weight_w;

    int c_per_offset_grp = n_in_channels / n_offset_grps;
    int grp_idx = in_c / c_per_offset_grp;

    device float* col_ptr = columns +
        (out_c * (batch_sz * out_h * out_w) + out_b * (out_h * out_w) +
         out_y * out_w + out_x);

    device const float* in_ptr = input +
        (out_b * (n_in_channels * height * width) + in_c * (height * width));

    device const float* offset_ptr = offset +
        (out_b * n_offset_grps + grp_idx) * 2 * weight_h * weight_w * out_h * out_w;

    device const float* mask_ptr = mask +
        (out_b * n_offset_grps + grp_idx) * weight_h * weight_w * out_h * out_w;

    for (int i = 0; i < weight_h; ++i) {
        for (int j = 0; j < weight_w; ++j) {
            int mask_idx = i * weight_w + j;
            int offset_idx = 2 * mask_idx;

            float mask_value = 1.0f;
            if (use_mask) {
                mask_value = mask_ptr[mask_idx * (out_h * out_w) + out_y * out_w + out_x];
            }

            float offset_h = offset_ptr[offset_idx * (out_h * out_w) + out_y * out_w + out_x];
            float offset_w = offset_ptr[(offset_idx + 1) * (out_h * out_w) + out_y * out_w + out_x];

            float y = float(out_y * stride_h - pad_h) + float(i * dilation_h) + offset_h;
            float x = float(out_x * stride_w - pad_w) + float(j * dilation_w) + offset_w;

            *col_ptr = mask_value * bilinear_interpolate(in_ptr, height, width, y, x);
            col_ptr += batch_sz * out_h * out_w;
        }
    }
}

kernel void deformable_im2col_fp16(
    device const half* input        [[buffer(0)]],
    device const half* offset       [[buffer(1)]],
    device const half* mask         [[buffer(2)]],
    device half* columns            [[buffer(3)]],
    constant int& height            [[buffer(4)]],
    constant int& width             [[buffer(5)]],
    constant int& weight_h          [[buffer(6)]],
    constant int& weight_w          [[buffer(7)]],
    constant int& pad_h             [[buffer(8)]],
    constant int& pad_w             [[buffer(9)]],
    constant int& stride_h          [[buffer(10)]],
    constant int& stride_w          [[buffer(11)]],
    constant int& dilation_h        [[buffer(12)]],
    constant int& dilation_w        [[buffer(13)]],
    constant int& batch_sz          [[buffer(14)]],
    constant int& n_in_channels     [[buffer(15)]],
    constant int& n_offset_grps     [[buffer(16)]],
    constant int& out_h             [[buffer(17)]],
    constant int& out_w             [[buffer(18)]],
    constant int& use_mask          [[buffer(19)]],
    uint gid [[thread_position_in_grid]]
) {
    int n = n_in_channels * out_h * out_w * batch_sz;
    if (int(gid) >= n) return;

    int index = int(gid);
    int out_x = index % out_w;
    int out_y = (index / out_w) % out_h;
    int out_b = (index / (out_w * out_h)) % batch_sz;
    int in_c = index / (out_w * out_h * batch_sz);
    int out_c = in_c * weight_h * weight_w;

    int c_per_offset_grp = n_in_channels / n_offset_grps;
    int grp_idx = in_c / c_per_offset_grp;

    device half* col_ptr = columns +
        (out_c * (batch_sz * out_h * out_w) + out_b * (out_h * out_w) +
         out_y * out_w + out_x);

    device const half* in_ptr = input +
        (out_b * (n_in_channels * height * width) + in_c * (height * width));

    device const half* offset_ptr = offset +
        (out_b * n_offset_grps + grp_idx) * 2 * weight_h * weight_w * out_h * out_w;

    device const half* mask_ptr = mask +
        (out_b * n_offset_grps + grp_idx) * weight_h * weight_w * out_h * out_w;

    for (int i = 0; i < weight_h; ++i) {
        for (int j = 0; j < weight_w; ++j) {
            int mask_idx = i * weight_w + j;
            int offset_idx = 2 * mask_idx;

            half mask_value = half(1.0);
            if (use_mask) {
                mask_value = mask_ptr[mask_idx * (out_h * out_w) + out_y * out_w + out_x];
            }

            half offset_h = offset_ptr[offset_idx * (out_h * out_w) + out_y * out_w + out_x];
            half offset_w = offset_ptr[(offset_idx + 1) * (out_h * out_w) + out_y * out_w + out_x];

            half y = half(out_y * stride_h - pad_h) + half(i * dilation_h) + offset_h;
            half x = half(out_x * stride_w - pad_w) + half(j * dilation_w) + offset_w;

            *col_ptr = mask_value * bilinear_interpolate(in_ptr, height, width, y, x);
            col_ptr += batch_sz * out_h * out_w;
        }
    }
}

// Backward: col2im for input gradients
kernel void deformable_col2im_fp32(
    device const float* col         [[buffer(0)]],
    device const float* offset      [[buffer(1)]],
    device const float* mask        [[buffer(2)]],
    device atomic_float* grad_im    [[buffer(3)]],
    constant int& channels          [[buffer(4)]],
    constant int& height            [[buffer(5)]],
    constant int& width             [[buffer(6)]],
    constant int& kernel_h          [[buffer(7)]],
    constant int& kernel_w          [[buffer(8)]],
    constant int& pad_h             [[buffer(9)]],
    constant int& pad_w             [[buffer(10)]],
    constant int& stride_h          [[buffer(11)]],
    constant int& stride_w          [[buffer(12)]],
    constant int& dilation_h        [[buffer(13)]],
    constant int& dilation_w        [[buffer(14)]],
    constant int& batch_sz          [[buffer(15)]],
    constant int& n_offset_grps     [[buffer(16)]],
    constant int& out_h             [[buffer(17)]],
    constant int& out_w             [[buffer(18)]],
    constant int& use_mask          [[buffer(19)]],
    uint gid [[thread_position_in_grid]]
) {
    int n = channels * kernel_h * kernel_w * out_h * out_w * batch_sz;
    if (int(gid) >= n) return;

    int index = int(gid);
    int out_x = index % out_w;
    int out_y = (index / out_w) % out_h;
    int b = (index / (out_w * out_h)) % batch_sz;
    int j = (index / (out_w * out_h * batch_sz)) % kernel_w;
    int i = (index / (out_w * out_h * batch_sz * kernel_w)) % kernel_h;
    int c = index / (out_w * out_h * batch_sz * kernel_w * kernel_h);

    int c_per_offset_grp = channels / n_offset_grps;
    int offset_grp = c / c_per_offset_grp;

    device const float* offset_ptr = offset +
        (b * n_offset_grps + offset_grp) * 2 * kernel_h * kernel_w * out_h * out_w;
    device const float* mask_ptr = mask +
        (b * n_offset_grps + offset_grp) * kernel_h * kernel_w * out_h * out_w;

    int mask_idx = i * kernel_w + j;
    int offset_idx = 2 * mask_idx;

    float offset_h = offset_ptr[offset_idx * out_h * out_w + out_y * out_w + out_x];
    float offset_w = offset_ptr[(offset_idx + 1) * out_h * out_w + out_y * out_w + out_x];

    float mask_value = 1.0f;
    if (use_mask) {
        mask_value = mask_ptr[mask_idx * out_h * out_w + out_y * out_w + out_x];
    }

    float y = float(out_y * stride_h - pad_h) + float(i * dilation_h) + offset_h;
    float x = float(out_x * stride_w - pad_w) + float(j * dilation_w) + offset_w;

    float col_val = col[index];

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int yp = int(y) + dy;
            int xp = int(x) + dx;

            if (0 <= yp && yp < height && 0 <= xp && xp < width &&
                abs(y - float(yp)) < 1.0f && abs(x - float(xp)) < 1.0f) {

                int grad_pos = ((b * channels + c) * height + yp) * width + xp;
                float weight = (1.0f - abs(y - float(yp))) * (1.0f - abs(x - float(xp)));

                atomic_fetch_add_explicit(&grad_im[grad_pos], mask_value * weight * col_val, memory_order_relaxed);
            }
        }
    }
}

// Backward: gradient for offsets and mask
kernel void deformable_col2im_coord_fp32(
    device const float* col         [[buffer(0)]],
    device const float* im          [[buffer(1)]],
    device const float* offset      [[buffer(2)]],
    device const float* mask        [[buffer(3)]],
    device float* grad_offset       [[buffer(4)]],
    device float* grad_mask         [[buffer(5)]],
    constant int& channels          [[buffer(6)]],
    constant int& height            [[buffer(7)]],
    constant int& width             [[buffer(8)]],
    constant int& weight_h          [[buffer(9)]],
    constant int& weight_w          [[buffer(10)]],
    constant int& pad_h             [[buffer(11)]],
    constant int& pad_w             [[buffer(12)]],
    constant int& stride_h          [[buffer(13)]],
    constant int& stride_w          [[buffer(14)]],
    constant int& dilation_h        [[buffer(15)]],
    constant int& dilation_w        [[buffer(16)]],
    constant int& batch_sz          [[buffer(17)]],
    constant int& n_offset_grps     [[buffer(18)]],
    constant int& out_h             [[buffer(19)]],
    constant int& out_w             [[buffer(20)]],
    constant int& use_mask          [[buffer(21)]],
    uint gid [[thread_position_in_grid]]
) {
    int offset_channels = 2 * weight_h * weight_w * n_offset_grps;
    int n = out_h * out_w * offset_channels * batch_sz;
    if (int(gid) >= n) return;

    int index = int(gid);

    float grad_offset_val = 0.0f;
    float grad_mask_val = 0.0f;

    int w = index % out_w;
    int h = (index / out_w) % out_h;
    int w_w = (index / (out_w * out_h * 2)) % weight_w;
    int w_h = (index / (out_w * out_h * 2 * weight_w)) % weight_h;
    int c = (index / (out_w * out_h)) % offset_channels;
    int b = index / (out_w * out_h * offset_channels);

    int offset_grp = c / (2 * weight_h * weight_w);
    int col_step = weight_h * weight_w;
    int c_per_offset_grp = channels / n_offset_grps;

    device const float* col_ptr = col +
        offset_grp * c_per_offset_grp * weight_h * weight_w * batch_sz * out_w * out_h;
    device const float* im_ptr = im +
        (b * n_offset_grps + offset_grp) * c_per_offset_grp * height * width;
    device const float* offset_ptr = offset +
        (b * n_offset_grps + offset_grp) * 2 * weight_h * weight_w * out_h * out_w;
    device const float* mask_ptr = mask +
        (b * n_offset_grps + offset_grp) * weight_h * weight_w * out_h * out_w;

    int offset_c = c - offset_grp * 2 * weight_h * weight_w;
    bool is_y_direction = (offset_c % 2) == 0;

    int c_bound = c_per_offset_grp * weight_h * weight_w;

    for (int col_c = offset_c / 2; col_c < c_bound; col_c += col_step) {
        int col_pos = (((col_c * batch_sz + b) * out_h) + h) * out_w + w;

        int out_x = col_pos % out_w;
        int out_y = (col_pos / out_w) % out_h;
        int jj = (col_pos / (out_w * out_h * batch_sz)) % weight_w;
        int ii = (col_pos / (out_w * out_h * batch_sz * weight_w)) % weight_h;

        int mask_idx = ii * weight_w + jj;

        float offset_h = offset_ptr[(2 * mask_idx) * out_h * out_w + out_y * out_w + out_x];
        float offset_w = offset_ptr[(2 * mask_idx + 1) * out_h * out_w + out_y * out_w + out_x];

        float mask_value = 1.0f;
        if (use_mask) {
            mask_value = mask_ptr[mask_idx * out_h * out_w + out_y * out_w + out_x];
        }

        float y = float(out_y * stride_h - pad_h) + float(ii * dilation_h) + offset_h;
        float x = float(out_x * stride_w - pad_w) + float(jj * dilation_w) + offset_w;

        float weight = get_coordinate_weight(im_ptr, height, width, y, x, is_y_direction);
        grad_offset_val += mask_value * weight * col_ptr[col_pos];

        if (use_mask && is_y_direction) {
            grad_mask_val += col_ptr[col_pos] * bilinear_interpolate(im_ptr, height, width, y, x);
        }

        im_ptr += height * width;
    }

    grad_offset[index] = grad_offset_val;

    if (use_mask && is_y_direction) {
        int idx = ((((b * n_offset_grps + offset_grp) * weight_h + w_h) * weight_w + w_w) * out_h + h) * out_w + w;
        grad_mask[idx] = grad_mask_val;
    }
}
)";
}

static bool init_metal() {
    if (g_device) return true;

    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            throw std::runtime_error("Failed to create Metal device");
        }

        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.mathMode = MTLMathModeFast;

        g_library = [g_device newLibraryWithSource:get_metal_source() options:options error:&error];
        if (!g_library) {
            throw std::runtime_error("Failed to compile Metal library: " +
                                     std::string([[error localizedDescription] UTF8String]));
        }

        // Create pipelines
        auto create_pipeline = [&](NSString* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn = [g_library newFunctionWithName:name];
            if (!fn) {
                throw std::runtime_error("Failed to find function: " +
                                         std::string([name UTF8String]));
            }
            id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
            if (!pipeline) {
                throw std::runtime_error("Failed to create pipeline: " +
                                         std::string([[error localizedDescription] UTF8String]));
            }
            return pipeline;
        };

        g_im2col_fp32 = create_pipeline(@"deformable_im2col_fp32");
        g_im2col_fp16 = create_pipeline(@"deformable_im2col_fp16");
        g_col2im_fp32 = create_pipeline(@"deformable_col2im_fp32");
        g_col2im_coord_fp32 = create_pipeline(@"deformable_col2im_coord_fp32");
    }

    return true;
}

// =============================================================================
// Forward Pass
// =============================================================================

at::Tensor deform_conv2d_forward_mps(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const at::Tensor& mask,
    const at::Tensor& bias,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dilation_h, int64_t dilation_w,
    int64_t n_weight_grps,
    int64_t n_offset_grps,
    bool use_mask
) {
    TORCH_CHECK(input.device().is_mps(), "input must be on MPS");
    TORCH_CHECK(weight.device().is_mps(), "weight must be on MPS");
    TORCH_CHECK(offset.device().is_mps(), "offset must be on MPS");

    init_metal();

    const int64_t batch_sz = input.size(0);
    const int64_t n_in_channels = input.size(1);
    const int64_t in_h = input.size(2);
    const int64_t in_w = input.size(3);

    const int64_t n_out_channels = weight.size(0);
    const int64_t weight_h = weight.size(2);
    const int64_t weight_w = weight.size(3);

    const int64_t out_h = (in_h + 2 * pad_h - (dilation_h * (weight_h - 1) + 1)) / stride_h + 1;
    const int64_t out_w = (in_w + 2 * pad_w - (dilation_w * (weight_w - 1) + 1)) / stride_w + 1;

    // Allocate column buffer
    auto columns = at::empty(
        {n_in_channels * weight_h * weight_w, batch_sz * out_h * out_w},
        input.options()
    );

    // Handle mask
    at::Tensor mask_tensor = mask;
    if (!use_mask || mask.numel() == 0) {
        mask_tensor = at::ones(
            {batch_sz, n_offset_grps, weight_h, weight_w, out_h, out_w},
            input.options()
        );
    }

    auto input_contig = input.contiguous();
    auto offset_contig = offset.contiguous();
    auto mask_contig = mask_tensor.contiguous();

    // Use MPS stream with PyTorch's shared encoder (zero-sync)
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();

        bool use_fp16 = input.scalar_type() == at::kHalf;
        auto pipeline = use_fp16 ? g_im2col_fp16 : g_im2col_fp32;
        [encoder setComputePipelineState:pipeline];

        // Set buffers
        id<MTLBuffer> input_buf = at::native::mps::getMTLBufferStorage(input_contig);
        id<MTLBuffer> offset_buf = at::native::mps::getMTLBufferStorage(offset_contig);
        id<MTLBuffer> mask_buf = at::native::mps::getMTLBufferStorage(mask_contig);
        id<MTLBuffer> columns_buf = at::native::mps::getMTLBufferStorage(columns);

        [encoder setBuffer:input_buf offset:0 atIndex:0];
        [encoder setBuffer:offset_buf offset:0 atIndex:1];
        [encoder setBuffer:mask_buf offset:0 atIndex:2];
        [encoder setBuffer:columns_buf offset:0 atIndex:3];

        // Set constants
        int32_t height = static_cast<int32_t>(in_h);
        int32_t width = static_cast<int32_t>(in_w);
        int32_t weight_h_val = static_cast<int32_t>(weight_h);
        int32_t weight_w_val = static_cast<int32_t>(weight_w);
        int32_t pad_h_val = static_cast<int32_t>(pad_h);
        int32_t pad_w_val = static_cast<int32_t>(pad_w);
        int32_t stride_h_val = static_cast<int32_t>(stride_h);
        int32_t stride_w_val = static_cast<int32_t>(stride_w);
        int32_t dilation_h_val = static_cast<int32_t>(dilation_h);
        int32_t dilation_w_val = static_cast<int32_t>(dilation_w);
        int32_t batch_sz_val = static_cast<int32_t>(batch_sz);
        int32_t n_in_channels_val = static_cast<int32_t>(n_in_channels);
        int32_t n_offset_grps_val = static_cast<int32_t>(n_offset_grps);
        int32_t out_h_val = static_cast<int32_t>(out_h);
        int32_t out_w_val = static_cast<int32_t>(out_w);
        int32_t use_mask_val = use_mask ? 1 : 0;

        [encoder setBytes:&height length:sizeof(int32_t) atIndex:4];
        [encoder setBytes:&width length:sizeof(int32_t) atIndex:5];
        [encoder setBytes:&weight_h_val length:sizeof(int32_t) atIndex:6];
        [encoder setBytes:&weight_w_val length:sizeof(int32_t) atIndex:7];
        [encoder setBytes:&pad_h_val length:sizeof(int32_t) atIndex:8];
        [encoder setBytes:&pad_w_val length:sizeof(int32_t) atIndex:9];
        [encoder setBytes:&stride_h_val length:sizeof(int32_t) atIndex:10];
        [encoder setBytes:&stride_w_val length:sizeof(int32_t) atIndex:11];
        [encoder setBytes:&dilation_h_val length:sizeof(int32_t) atIndex:12];
        [encoder setBytes:&dilation_w_val length:sizeof(int32_t) atIndex:13];
        [encoder setBytes:&batch_sz_val length:sizeof(int32_t) atIndex:14];
        [encoder setBytes:&n_in_channels_val length:sizeof(int32_t) atIndex:15];
        [encoder setBytes:&n_offset_grps_val length:sizeof(int32_t) atIndex:16];
        [encoder setBytes:&out_h_val length:sizeof(int32_t) atIndex:17];
        [encoder setBytes:&out_w_val length:sizeof(int32_t) atIndex:18];
        [encoder setBytes:&use_mask_val length:sizeof(int32_t) atIndex:19];

        // Dispatch
        int64_t num_kernels = n_in_channels * out_h * out_w * batch_sz;
        MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
        MTLSize gridSize = MTLSizeMake((num_kernels + 255) / 256 * 256, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

        // Don't endEncoding/commit - PyTorch manages encoder lifecycle
    }

    // Reshape columns and perform matrix multiplication with weights
    // columns: [n_in_channels * weight_h * weight_w, batch_sz * out_h * out_w]
    // weight: [n_out_channels, n_in_channels / n_weight_grps, weight_h, weight_w]

    auto weight_flat = weight.view({n_out_channels, -1});  // [out_ch, in_ch * kh * kw]

    // Perform grouped matmul if needed
    at::Tensor output;
    if (n_weight_grps == 1) {
        // Simple matmul: [out_ch, in_ch*kh*kw] x [in_ch*kh*kw, batch*out_h*out_w]
        output = at::mm(weight_flat, columns);
    } else {
        // Grouped convolution
        int64_t out_ch_per_grp = n_out_channels / n_weight_grps;
        int64_t in_ch_per_grp = n_in_channels / n_weight_grps;
        int64_t col_per_grp = in_ch_per_grp * weight_h * weight_w;

        auto columns_grouped = columns.view({n_weight_grps, col_per_grp, -1});
        auto weight_grouped = weight_flat.view({n_weight_grps, out_ch_per_grp, col_per_grp});

        output = at::bmm(weight_grouped, columns_grouped).view({n_out_channels, -1});
    }

    // Reshape to [batch, out_ch, out_h, out_w]
    output = output.view({n_out_channels, batch_sz, out_h, out_w})
                   .permute({1, 0, 2, 3})
                   .contiguous();

    // Add bias
    if (bias.defined() && bias.numel() > 0) {
        output = output + bias.view({1, -1, 1, 1});
    }

    return output;
}

// =============================================================================
// im2col - exposed for backward pass
// =============================================================================

at::Tensor deformable_im2col_mps(
    const at::Tensor& input,
    const at::Tensor& offset,
    const at::Tensor& mask,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dilation_h, int64_t dilation_w,
    int64_t n_offset_grps,
    bool use_mask
) {
    TORCH_CHECK(input.device().is_mps(), "input must be on MPS");

    init_metal();

    const int64_t batch_sz = input.size(0);
    const int64_t n_in_channels = input.size(1);
    const int64_t in_h = input.size(2);
    const int64_t in_w = input.size(3);

    const int64_t out_h = (in_h + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int64_t out_w = (in_w + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    // Allocate columns
    auto columns = at::empty(
        {n_in_channels * kernel_h * kernel_w, batch_sz * out_h * out_w},
        input.options()
    );

    // Handle mask
    at::Tensor mask_tensor = mask;
    if (!use_mask || mask.numel() == 0) {
        mask_tensor = at::ones({batch_sz, n_offset_grps, kernel_h, kernel_w, out_h, out_w}, input.options());
    }

    auto input_contig = input.contiguous();
    auto offset_contig = offset.contiguous();
    auto mask_contig = mask_tensor.contiguous();

    // Use MPS stream with PyTorch's shared encoder (zero-sync)
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();

        bool use_fp16 = input.scalar_type() == at::kHalf;
        auto pipeline = use_fp16 ? g_im2col_fp16 : g_im2col_fp32;
        [encoder setComputePipelineState:pipeline];

        id<MTLBuffer> input_buf = at::native::mps::getMTLBufferStorage(input_contig);
        id<MTLBuffer> offset_buf = at::native::mps::getMTLBufferStorage(offset_contig);
        id<MTLBuffer> mask_buf = at::native::mps::getMTLBufferStorage(mask_contig);
        id<MTLBuffer> columns_buf = at::native::mps::getMTLBufferStorage(columns);

        [encoder setBuffer:input_buf offset:0 atIndex:0];
        [encoder setBuffer:offset_buf offset:0 atIndex:1];
        [encoder setBuffer:mask_buf offset:0 atIndex:2];
        [encoder setBuffer:columns_buf offset:0 atIndex:3];

        int32_t height = static_cast<int32_t>(in_h);
        int32_t width = static_cast<int32_t>(in_w);
        int32_t weight_h_val = static_cast<int32_t>(kernel_h);
        int32_t weight_w_val = static_cast<int32_t>(kernel_w);
        int32_t pad_h_val = static_cast<int32_t>(pad_h);
        int32_t pad_w_val = static_cast<int32_t>(pad_w);
        int32_t stride_h_val = static_cast<int32_t>(stride_h);
        int32_t stride_w_val = static_cast<int32_t>(stride_w);
        int32_t dilation_h_val = static_cast<int32_t>(dilation_h);
        int32_t dilation_w_val = static_cast<int32_t>(dilation_w);
        int32_t batch_sz_val = static_cast<int32_t>(batch_sz);
        int32_t n_in_channels_val = static_cast<int32_t>(n_in_channels);
        int32_t n_offset_grps_val = static_cast<int32_t>(n_offset_grps);
        int32_t out_h_val = static_cast<int32_t>(out_h);
        int32_t out_w_val = static_cast<int32_t>(out_w);
        int32_t use_mask_val = use_mask ? 1 : 0;

        [encoder setBytes:&height length:sizeof(int32_t) atIndex:4];
        [encoder setBytes:&width length:sizeof(int32_t) atIndex:5];
        [encoder setBytes:&weight_h_val length:sizeof(int32_t) atIndex:6];
        [encoder setBytes:&weight_w_val length:sizeof(int32_t) atIndex:7];
        [encoder setBytes:&pad_h_val length:sizeof(int32_t) atIndex:8];
        [encoder setBytes:&pad_w_val length:sizeof(int32_t) atIndex:9];
        [encoder setBytes:&stride_h_val length:sizeof(int32_t) atIndex:10];
        [encoder setBytes:&stride_w_val length:sizeof(int32_t) atIndex:11];
        [encoder setBytes:&dilation_h_val length:sizeof(int32_t) atIndex:12];
        [encoder setBytes:&dilation_w_val length:sizeof(int32_t) atIndex:13];
        [encoder setBytes:&batch_sz_val length:sizeof(int32_t) atIndex:14];
        [encoder setBytes:&n_in_channels_val length:sizeof(int32_t) atIndex:15];
        [encoder setBytes:&n_offset_grps_val length:sizeof(int32_t) atIndex:16];
        [encoder setBytes:&out_h_val length:sizeof(int32_t) atIndex:17];
        [encoder setBytes:&out_w_val length:sizeof(int32_t) atIndex:18];
        [encoder setBytes:&use_mask_val length:sizeof(int32_t) atIndex:19];

        int64_t num_kernels = n_in_channels * out_h * out_w * batch_sz;
        MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
        MTLSize gridSize = MTLSizeMake((num_kernels + 255) / 256 * 256, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

        // Don't endEncoding/commit - PyTorch manages encoder lifecycle
    }

    return columns;
}

// =============================================================================
// Backward Pass - Input Gradient (col2im)
// =============================================================================

at::Tensor deform_conv2d_backward_input_mps(
    const at::Tensor& grad_col,  // [C_in * K * K, B * H_out * W_out]
    const at::Tensor& input,
    const at::Tensor& offset,
    const at::Tensor& mask,
    int64_t in_h, int64_t in_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dilation_h, int64_t dilation_w,
    int64_t kernel_h, int64_t kernel_w,
    int64_t n_offset_grps,
    bool use_mask
) {
    init_metal();

    const int64_t batch_sz = input.size(0);
    const int64_t n_in_channels = input.size(1);

    // Compute output dimensions
    const int64_t out_h = (in_h + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int64_t out_w = (in_w + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    // Allocate grad_input
    auto grad_input = at::zeros({batch_sz, n_in_channels, in_h, in_w}, input.options());

    // Handle mask
    at::Tensor mask_tensor = mask;
    if (!use_mask || mask.numel() == 0) {
        mask_tensor = at::ones({batch_sz, n_offset_grps, kernel_h, kernel_w, out_h, out_w}, input.options());
    }

    auto grad_col_contig = grad_col.contiguous();
    auto offset_contig = offset.contiguous();
    auto mask_contig = mask_tensor.contiguous();

    // Use MPS stream with PyTorch's shared encoder (zero-sync)
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();

        [encoder setComputePipelineState:g_col2im_fp32];

        id<MTLBuffer> col_buf = at::native::mps::getMTLBufferStorage(grad_col_contig);
        id<MTLBuffer> offset_buf = at::native::mps::getMTLBufferStorage(offset_contig);
        id<MTLBuffer> mask_buf = at::native::mps::getMTLBufferStorage(mask_contig);
        id<MTLBuffer> grad_im_buf = at::native::mps::getMTLBufferStorage(grad_input);

        [encoder setBuffer:col_buf offset:0 atIndex:0];
        [encoder setBuffer:offset_buf offset:0 atIndex:1];
        [encoder setBuffer:mask_buf offset:0 atIndex:2];
        [encoder setBuffer:grad_im_buf offset:0 atIndex:3];

        int32_t channels_val = static_cast<int32_t>(n_in_channels);
        int32_t height_val = static_cast<int32_t>(in_h);
        int32_t width_val = static_cast<int32_t>(in_w);
        int32_t kernel_h_val = static_cast<int32_t>(kernel_h);
        int32_t kernel_w_val = static_cast<int32_t>(kernel_w);
        int32_t pad_h_val = static_cast<int32_t>(pad_h);
        int32_t pad_w_val = static_cast<int32_t>(pad_w);
        int32_t stride_h_val = static_cast<int32_t>(stride_h);
        int32_t stride_w_val = static_cast<int32_t>(stride_w);
        int32_t dilation_h_val = static_cast<int32_t>(dilation_h);
        int32_t dilation_w_val = static_cast<int32_t>(dilation_w);
        int32_t batch_sz_val = static_cast<int32_t>(batch_sz);
        int32_t n_offset_grps_val = static_cast<int32_t>(n_offset_grps);
        int32_t out_h_val = static_cast<int32_t>(out_h);
        int32_t out_w_val = static_cast<int32_t>(out_w);
        int32_t use_mask_val = use_mask ? 1 : 0;

        [encoder setBytes:&channels_val length:sizeof(int32_t) atIndex:4];
        [encoder setBytes:&height_val length:sizeof(int32_t) atIndex:5];
        [encoder setBytes:&width_val length:sizeof(int32_t) atIndex:6];
        [encoder setBytes:&kernel_h_val length:sizeof(int32_t) atIndex:7];
        [encoder setBytes:&kernel_w_val length:sizeof(int32_t) atIndex:8];
        [encoder setBytes:&pad_h_val length:sizeof(int32_t) atIndex:9];
        [encoder setBytes:&pad_w_val length:sizeof(int32_t) atIndex:10];
        [encoder setBytes:&stride_h_val length:sizeof(int32_t) atIndex:11];
        [encoder setBytes:&stride_w_val length:sizeof(int32_t) atIndex:12];
        [encoder setBytes:&dilation_h_val length:sizeof(int32_t) atIndex:13];
        [encoder setBytes:&dilation_w_val length:sizeof(int32_t) atIndex:14];
        [encoder setBytes:&batch_sz_val length:sizeof(int32_t) atIndex:15];
        [encoder setBytes:&n_offset_grps_val length:sizeof(int32_t) atIndex:16];
        [encoder setBytes:&out_h_val length:sizeof(int32_t) atIndex:17];
        [encoder setBytes:&out_w_val length:sizeof(int32_t) atIndex:18];
        [encoder setBytes:&use_mask_val length:sizeof(int32_t) atIndex:19];

        int64_t num_kernels = n_in_channels * kernel_h * kernel_w * out_h * out_w * batch_sz;
        MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
        MTLSize gridSize = MTLSizeMake((num_kernels + 255) / 256 * 256, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

        // Don't endEncoding/commit - PyTorch manages encoder lifecycle
    }

    return grad_input;
}

// =============================================================================
// Backward Pass - Offset and Mask Gradients
// =============================================================================

std::tuple<at::Tensor, at::Tensor> deform_conv2d_backward_offset_mask_mps(
    const at::Tensor& grad_col,
    const at::Tensor& input,
    const at::Tensor& offset,
    const at::Tensor& mask,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dilation_h, int64_t dilation_w,
    int64_t n_offset_grps,
    bool use_mask
) {
    init_metal();

    const int64_t batch_sz = input.size(0);
    const int64_t n_in_channels = input.size(1);
    const int64_t in_h = input.size(2);
    const int64_t in_w = input.size(3);

    const int64_t out_h = (in_h + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int64_t out_w = (in_w + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    // Allocate gradients
    auto grad_offset = at::zeros_like(offset);
    auto grad_mask = use_mask ? at::zeros_like(mask) : at::empty({0}, input.options());

    // Handle mask
    at::Tensor mask_tensor = mask;
    if (!use_mask || mask.numel() == 0) {
        mask_tensor = at::ones({batch_sz, n_offset_grps, kernel_h, kernel_w, out_h, out_w}, input.options());
    }

    auto grad_col_contig = grad_col.contiguous();
    auto input_contig = input.contiguous();
    auto offset_contig = offset.contiguous();
    auto mask_contig = mask_tensor.contiguous();

    // Use MPS stream with PyTorch's shared encoder (zero-sync)
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();

        [encoder setComputePipelineState:g_col2im_coord_fp32];

        id<MTLBuffer> col_buf = at::native::mps::getMTLBufferStorage(grad_col_contig);
        id<MTLBuffer> im_buf = at::native::mps::getMTLBufferStorage(input_contig);
        id<MTLBuffer> offset_buf = at::native::mps::getMTLBufferStorage(offset_contig);
        id<MTLBuffer> mask_buf = at::native::mps::getMTLBufferStorage(mask_contig);
        id<MTLBuffer> grad_offset_buf = at::native::mps::getMTLBufferStorage(grad_offset);

        // Create dummy buffer for grad_mask if not using mask
        at::Tensor grad_mask_tensor = use_mask ? grad_mask : at::zeros({1}, input.options());
        id<MTLBuffer> grad_mask_buf = at::native::mps::getMTLBufferStorage(grad_mask_tensor);

        [encoder setBuffer:col_buf offset:0 atIndex:0];
        [encoder setBuffer:im_buf offset:0 atIndex:1];
        [encoder setBuffer:offset_buf offset:0 atIndex:2];
        [encoder setBuffer:mask_buf offset:0 atIndex:3];
        [encoder setBuffer:grad_offset_buf offset:0 atIndex:4];
        [encoder setBuffer:grad_mask_buf offset:0 atIndex:5];

        int32_t channels_val = static_cast<int32_t>(n_in_channels);
        int32_t height_val = static_cast<int32_t>(in_h);
        int32_t width_val = static_cast<int32_t>(in_w);
        int32_t weight_h_val = static_cast<int32_t>(kernel_h);
        int32_t weight_w_val = static_cast<int32_t>(kernel_w);
        int32_t pad_h_val = static_cast<int32_t>(pad_h);
        int32_t pad_w_val = static_cast<int32_t>(pad_w);
        int32_t stride_h_val = static_cast<int32_t>(stride_h);
        int32_t stride_w_val = static_cast<int32_t>(stride_w);
        int32_t dilation_h_val = static_cast<int32_t>(dilation_h);
        int32_t dilation_w_val = static_cast<int32_t>(dilation_w);
        int32_t batch_sz_val = static_cast<int32_t>(batch_sz);
        int32_t n_offset_grps_val = static_cast<int32_t>(n_offset_grps);
        int32_t out_h_val = static_cast<int32_t>(out_h);
        int32_t out_w_val = static_cast<int32_t>(out_w);
        int32_t use_mask_val = use_mask ? 1 : 0;

        [encoder setBytes:&channels_val length:sizeof(int32_t) atIndex:6];
        [encoder setBytes:&height_val length:sizeof(int32_t) atIndex:7];
        [encoder setBytes:&width_val length:sizeof(int32_t) atIndex:8];
        [encoder setBytes:&weight_h_val length:sizeof(int32_t) atIndex:9];
        [encoder setBytes:&weight_w_val length:sizeof(int32_t) atIndex:10];
        [encoder setBytes:&pad_h_val length:sizeof(int32_t) atIndex:11];
        [encoder setBytes:&pad_w_val length:sizeof(int32_t) atIndex:12];
        [encoder setBytes:&stride_h_val length:sizeof(int32_t) atIndex:13];
        [encoder setBytes:&stride_w_val length:sizeof(int32_t) atIndex:14];
        [encoder setBytes:&dilation_h_val length:sizeof(int32_t) atIndex:15];
        [encoder setBytes:&dilation_w_val length:sizeof(int32_t) atIndex:16];
        [encoder setBytes:&batch_sz_val length:sizeof(int32_t) atIndex:17];
        [encoder setBytes:&n_offset_grps_val length:sizeof(int32_t) atIndex:18];
        [encoder setBytes:&out_h_val length:sizeof(int32_t) atIndex:19];
        [encoder setBytes:&out_w_val length:sizeof(int32_t) atIndex:20];
        [encoder setBytes:&use_mask_val length:sizeof(int32_t) atIndex:21];

        int64_t offset_channels = 2 * kernel_h * kernel_w * n_offset_grps;
        int64_t num_kernels = out_h * out_w * offset_channels * batch_sz;
        MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
        MTLSize gridSize = MTLSizeMake((num_kernels + 255) / 256 * 256, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

        // Don't endEncoding/commit - PyTorch manages encoder lifecycle
    }

    return std::make_tuple(grad_offset, grad_mask);
}

// =============================================================================
// Python Bindings
// =============================================================================

PYBIND11_MODULE(_C, m) {
    m.doc() = "MPS Deformable Convolution - torchvision.ops.deform_conv2d for Apple Silicon";

    m.def("deform_conv2d_forward", &deform_conv2d_forward_mps,
          "Deformable Convolution 2D forward pass",
          py::arg("input"),
          py::arg("weight"),
          py::arg("offset"),
          py::arg("mask"),
          py::arg("bias"),
          py::arg("stride_h"),
          py::arg("stride_w"),
          py::arg("pad_h"),
          py::arg("pad_w"),
          py::arg("dilation_h"),
          py::arg("dilation_w"),
          py::arg("n_weight_grps"),
          py::arg("n_offset_grps"),
          py::arg("use_mask"));

    m.def("deform_conv2d_backward_input", &deform_conv2d_backward_input_mps,
          "Deformable Convolution 2D backward pass for input gradient");

    m.def("deform_conv2d_backward_offset_mask", &deform_conv2d_backward_offset_mask_mps,
          "Deformable Convolution 2D backward pass for offset and mask gradients");

    m.def("deformable_im2col", &deformable_im2col_mps,
          "Deformable im2col operation (Metal accelerated)");
}
