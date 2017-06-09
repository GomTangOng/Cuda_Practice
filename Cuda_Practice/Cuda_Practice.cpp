// Cuda_Practice.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

extern "C" void Cuda_Sum_Init(int *, int *, int *);
extern "C" void Cuda_Dot_Init();
extern "C" void cuda_host_alloc_test(int size, bool up);
extern "C" void cuda_malloc_test(int size, bool up);
extern "C" void Init_Histogram();

#define SIZE (10*1024*1024)

int Histogram_cpu()
{
	auto start = GetTickCount();
	unsigned char *buffer = new unsigned char[SIZE];
	unsigned int histo[256]{ 0, };

	for (int i = 0; i < SIZE; ++i)
	{
		unsigned char temp = rand() % 256;
		buffer[i] = temp;
	}

	for (int i = 0; i < SIZE; ++i)
	{
		histo[buffer[i]]++;
	}

	long histoCount = 0;

	for (int i = 0; i < 256; ++i)
	{
		histoCount += histo[i];
	}
	auto end = GetTickCount();

	cout << "HistoCount : " << histoCount << endl;
	cout << "CPU Time : " << end - start << "ms" << endl;

	delete[] buffer;

	return histoCount;
}

int main()
{
	//Cuda_Dot_Init();
	//cuda_malloc_test(SIZE, true);
	//cuda_malloc_test(SIZE, false);
	//cuda_host_alloc_test(SIZE, true);
	//cuda_host_alloc_test(SIZE, false);
	Histogram_cpu();
	Init_Histogram();

    return 0;
}

