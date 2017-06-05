// Cuda_Practice.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

extern "C" void Cuda_Sum_Init(int *, int *, int *);
extern "C" void Cuda_Dot_Init();
extern "C" void cuda_host_alloc_test(int size, bool up);
extern "C" void cuda_malloc_test(int size, bool up);

#define SIZE (10*1024*1024)

int main()
{
	//Cuda_Dot_Init();
	cuda_malloc_test(SIZE, true);
	cuda_malloc_test(SIZE, false);
	cuda_host_alloc_test(SIZE, true);
	cuda_host_alloc_test(SIZE, false);
    return 0;
}

