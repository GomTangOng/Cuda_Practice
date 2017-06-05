#include "stdafx.h"

__global__ void Add(int *a, int *b, int *c)
{
	int tid = threadIdx.x;

	if (tid < N)
		c[tid] = a[tid] + b[tid];
}

extern "C" void Cuda_Sum_Init(int *a, int *b, int *c)
{
	int *d_a, *d_b, *d_c;

	cudaMalloc((void **)&d_a, sizeof(int) * N);
	cudaMalloc((void **)&d_b, sizeof(int) * N);
	cudaMalloc((void **)&d_c, sizeof(int) * N);

	for (int i = 0; i < N; ++i)
	{
		a[i] = i;
		b[i] = i * i;
		c[i] = 0;
	}

	cudaMemcpy(d_a, a, sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, sizeof(int) * N, cudaMemcpyHostToDevice);

	Add << <1, N >> >(d_a, d_b, d_c);

	cudaMemcpy(c, d_c, sizeof(int) * N, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}
