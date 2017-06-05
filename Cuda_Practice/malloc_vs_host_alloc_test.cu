#include "stdafx.h"

extern "C" void cuda_malloc_test(int size, bool up)
{
	cudaEvent_t start, stop;
	int *a, *dev_a;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	a = (int *)malloc(size * sizeof(*a));
	cudaMalloc((void**)&dev_a, size * sizeof(*dev_a));

	cudaEventRecord(start, 0);
	for (int i = 0; i < 100; ++i)
	{
		if (up)
			cudaMemcpy(dev_a, a, size * sizeof(*dev_a), cudaMemcpyHostToDevice);
		else
			cudaMemcpy(a, dev_a, size * sizeof(*dev_a), cudaMemcpyDeviceToHost);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	free(a);
	cudaFree(dev_a);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	if (up)
		cout << "malloc_test(HOST=>DEVICE) : " << elapsedTime << endl;
	else
		cout << "malloc_test(DEVICE=>HOST) : " << elapsedTime << endl;
}

extern "C" void cuda_host_alloc_test(int size, bool up)
{
	cudaEvent_t start, stop;
	int *a, *dev_a;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//a = (int *)malloc(size * sizeof(*a));
	cudaHostAlloc((void **)&a, size * sizeof(*a), cudaHostAllocDefault);
	cudaMalloc((void**)&dev_a, size * sizeof(*dev_a));

	cudaEventRecord(start, 0);
	for (int i = 0; i < 100; ++i)
	{
		if (up)
			cudaMemcpy(dev_a, a, size * sizeof(*dev_a), cudaMemcpyHostToDevice);
		else
			cudaMemcpy(a, dev_a, size * sizeof(*dev_a), cudaMemcpyDeviceToHost);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	//free(a);
	cudaFreeHost(a);
	cudaFree(dev_a);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	if (up)
		cout << "host_alloc_test(HOST=>DEVICE) : " << elapsedTime << endl;
	else
		cout << "host_alloc_test(DEVICE=>HOST) : " << elapsedTime << endl;
}


