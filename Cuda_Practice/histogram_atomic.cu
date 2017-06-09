#include "stdafx.h"

#define SIZE (100 * 1024 * 1024)

__global__ void histo_kernel(unsigned char *buffer, int size, unsigned int *histo)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	while (i < size)
	{
		atomicAdd(&histo[buffer[i]], 1);
		i += stride;
	}
}

extern "C" void Init_Histogram()
{
	unsigned char *buffer = new unsigned char[SIZE];

	for (int i = 0; i < SIZE; ++i)
	{
		buffer[i] = rand() % 'z' + 'a';
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	unsigned char *dev_buffer;
	unsigned int *dev_histo;

	cudaMalloc((void**)&dev_buffer, SIZE);
	cudaMemcpy(dev_buffer, buffer, SIZE, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dev_histo, 256 * sizeof(long));
	cudaMemset(dev_histo, 0, 256 * sizeof(int));

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	int blocks = prop.multiProcessorCount;

	histo_kernel << <blocks * 2, 256 >> > (dev_buffer, SIZE, dev_histo);

	unsigned int histo[256];
	cudaMemcpy(histo, dev_histo, 256 * sizeof(int), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cout << "Time to generate " << elapsedTime << endl;

	long histoCount = 0;
	for (int i = 0; i < 256; ++i)
	{
		histoCount += histo[i];
	}

	cout << "Histogram Sum : " << histoCount << endl;

	for (int i = 0; i < SIZE; ++i)
	{
		histo[buffer[i]]--;
	}
	for (int i = 0; i < 256; ++i)
	{
		if (histo[i] != 0) cout << "Failure at " << i << endl;
	}

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(dev_histo);
	cudaFree(dev_buffer);
	delete[] buffer;
}