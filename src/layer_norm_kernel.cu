#include "cuda_tensor_ops.h"

__global__ void layer_norm_kernel(
    const float* input, float* output,
    int rows, int cols,
    const float* gamma, const float* beta,
    float epsilon
) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Mean calculation
    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x)
        sum += input[row * cols + i];
    sum = warpReduceSum(sum);
    if (tid % 32 == 0) shared[tid / 32] = sum;
    __syncthreads();
    float mean = (tid < 32) ? warpReduceSum(shared[tid]) / cols : 0;

    // Variance calculation
    float var = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float diff = input[row * cols + i] - mean;
        var += diff * diff;
    }
    var = warpReduceSum(var);
    if (tid % 32 == 0) shared[tid / 32] = var;
    __syncthreads();
    float variance = (tid < 32) ? warpReduceSum(shared[tid]) / cols + epsilon : 0;

    // Normalization
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = (input[row * cols + i] - mean) * rsqrtf(variance);
        output[row * cols + i] = val * gamma[i] + beta[i];
    }
}

extern "C" void cuda_layer_norm(
    const float* input, float* output,
    int rows, int cols,
    const float* gamma, const float* beta,
    float epsilon, cudaStream_t stream
) {
    layer_norm_kernel<<<rows, 256, 0, stream>>>(input, output, rows, cols, gamma, beta, epsilon);
    CHECK_CUDA(cudaGetLastError());
}
