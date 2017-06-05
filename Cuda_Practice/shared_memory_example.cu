#include "stdafx.h"

__global__ void Dot(float *a, float *b, float *c)
{
	__shared__ float cache[ThreadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;
	float temp = 0;
	while (tid < N) {
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}
	cache[cacheIndex] = temp;
	__syncthreads();

	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0)
		c[blockIdx.x] = cache[0];
}

extern "C" void Cuda_Dot_Init()
{
	float *d_a, *d_b, *d_c;
	float *a, *b, *partial_c;

	a = (float *)malloc(sizeof(float) * N);
	b = (float *)malloc(sizeof(float) * N);
	partial_c = (float *)malloc(sizeof(float) * BlocksPerGrid);

	cudaMalloc((void **)&d_a, sizeof(float) * N);
	cudaMalloc((void **)&d_b, sizeof(float) * N);
	cudaMalloc((void **)&d_c, sizeof(float) * BlocksPerGrid);

	for (int i = 0; i < N; ++i)
	{
		a[i] = i;
		b[i] = i * 2;
	}

	printf("BlocksPerGrid : %d\n", BlocksPerGrid);

	cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

	Dot << <BlocksPerGrid, ThreadsPerBlock >> >(d_a, d_b, d_c);

	cudaMemcpy(partial_c, d_c, sizeof(float) * BlocksPerGrid, cudaMemcpyDeviceToHost);

	float c = 0;
	for (int i = 0; i < BlocksPerGrid; ++i)
	{
		c += partial_c[i];
	}
#define sum_squares(x)  (x*(x+1)*(2*x+1)/6)
	printf("Does GPU value %.6g = %.6g ?\n", c, 2 * sum_squares((float)(N - 1)));

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	free(a);
	free(b);
	free(partial_c);
}