#pragma once
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void cuda_gemm(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    const float* lora_A, const float* lora_B,
    int lora_rank, float lora_alpha,
    cudaStream_t stream
);

void cuda_layer_norm(
    const float* input, float* output,
    int rows, int cols,
    const float* gamma, const float* beta,
    float epsilon, cudaStream_t stream
);

#ifdef __cplusplus
}
#endif
