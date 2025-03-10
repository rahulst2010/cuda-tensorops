#include "cuda_tensor_ops.h"
#include <cuda_fp16.h>
#include <stdio.h>

#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

template <int BLOCK_SIZE>
__global__ void gemm_kernel(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Coordinate in C matrix
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;

    for (int i = 0; i < K; i += BLOCK_SIZE) {
        // Load tile from A and B into shared memory
        if (row < M && (i + tx) < K)
            As[ty][tx] = A[row * K + i + tx];
        else
            As[ty][tx] = 0.0f;

        if ((i + ty) < K && col < N)
            Bs[ty][tx] = B[(i + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Compute partial product
        for (int k = 0; k < BLOCK_SIZE; ++k)
            sum += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    // Write result to C
    if (row < M && col < N)
        C[row * N + col] = sum;
}

extern "C" void cuda_gemm(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    const float* lora_A, const float* lora_B,
    int lora_rank, float lora_alpha,
    cudaStream_t stream
) {
    const int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1)/BLOCK_SIZE, (M + BLOCK_SIZE - 1)/BLOCK_SIZE);
    
    gemm_kernel<BLOCK_SIZE><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
}
