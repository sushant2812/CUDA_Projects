
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
using namespace std;
#include <stdio.h>
#include <iostream> 
#include <random>
#include <cassert>
#include <iomanip>
struct Vector {
	double x, y, z;
};
const size_t WORKINGSET = 256 * 1024;

void CPUVectorAdd(const Vector* arrA, const Vector* arrB, Vector* arrC, size_t size) {
	for (size_t i = 0; i < size; i++) {
		arrC[i].x = arrA[i].x + arrB[i].x;
		arrC[i].y = arrA[i].y + arrB[i].y;
		arrC[i].z = arrA[i].z + arrB[i].z;
	}
}

__global__ void GPUVectorArrAdd(const Vector* arrA, const Vector* arrB, Vector* arrC) {

	//Calculate the index
	size_t i = threadIdx.x + blockIdx.x * blockDim.x; //Per Block Index

	// Compute the result
	arrC[i].x = arrA[i].x + arrB[i].x;
	arrC[i].y = arrA[i].y + arrB[i].y;
	arrC[i].z = arrA[i].z + arrB[i].z;

}
int main() {
	cout << "First Cuda Program :)" << endl;
	//Prepare a workset
	cout << "Prepare a workset" << endl;
	Vector* A = (Vector*)malloc(sizeof(Vector) * WORKINGSET);
	Vector* B = (Vector*)malloc(sizeof(Vector) * WORKINGSET);
	Vector* C = (Vector*)malloc(sizeof(Vector) * WORKINGSET);
	assert(A && B && "Memory allocation failed");
	for (size_t i = 0; i < WORKINGSET; i++) {
		A[i].x = 1.0f / (rand() % 200);
		A[i].y = 1.0f / (rand() % 200);
		A[i].z = 1.0f / (rand() % 200);
		B[i].x = 1.0f / (rand() % 200);
		B[i].y = 1.0f / (rand() % 200);
		B[i].z = 1.0f / (rand() % 200);
	}
	// Process Data
	//CPUVectorAdd(A, B, C, WORKINGSET);

	//Allocate Memory on GPU
	Vector* gpuA = nullptr;
	Vector* gpuB = nullptr;
	Vector* gpuC = nullptr;
	cudaMalloc(&gpuA, sizeof(Vector) * WORKINGSET);
	cudaMalloc(&gpuB, sizeof(Vector) * WORKINGSET);
	cudaMalloc(&gpuC, sizeof(Vector) * WORKINGSET);
	assert(gpuA && gpuB && gpuC && "Memory allocation failed");



	//Copy Data to GPU (CPU-> GPU)
	cudaMemcpy(gpuA, A, sizeof(Vector) * WORKINGSET, cudaMemcpyHostToDevice);
	cudaMemcpy(gpuB, B, sizeof(Vector) * WORKINGSET, cudaMemcpyHostToDevice);

	//Process Data on GPU
	const size_t BLOCK_SIZE = 256;
	const size_t blockCount = WORKINGSET / BLOCK_SIZE;
	GPUVectorArrAdd << <blockCount, BLOCK_SIZE >> > (gpuA, gpuB, gpuC);

	//Fetch Result from GPU (GPU -> CPU)
	cudaMemcpy(C, gpuC, sizeof(Vector) * WORKINGSET, cudaMemcpyDeviceToHost);


	//Print Some Values from result
	cout << "Print First 10 Results" << endl;
	for (int i = 0; i < 10; i++) {
		cout << "Result of X: " << C[i].x << " Y: " << C[i].y << " Z: " << C[i].z << endl;
	}

	//Free Memory
	cudaFree(gpuA);
	cudaFree(gpuB);
	cudaFree(gpuC);
	free(A);
	free(B);
	free(C);

}
