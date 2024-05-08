#include <iostream>
#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>

#define N 9

__global__ void add(float *d_A) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        for(int )
        //d_C[tid] = d_A[tid] + d_B[tid];
    }

}

int main() {
    //申请数据空间
    float *A = (float *) malloc(N * sizeof(float));
    float *d_A = NULL;

    hipMalloc((void **) &d_A, N * sizeof(float));

    //数据初始化
    for (int i = 0; i < N; i++) {
        A[i] = i;
    }
    hipMemcpy(d_A, A, sizeof(float) * N, hipMemcpyHostToDevice);

    dim3 blocksize(1, 1);
    dim3 gridsize(1, 1);
    // 进行数组相加
    add<<<gridsize, blocksize >>> (d_A);

    //释放申请空间
    free(A);
    hipFree(d_A);
}