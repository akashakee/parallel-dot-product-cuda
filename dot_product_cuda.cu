#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 100000000

__global__ void dotProductKernel(double *A, double *B, double *partial, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;

    while (tid < n)
    {
        sum += A[tid] * B[tid];
        tid += blockDim.x * gridDim.x;
    }

    partial[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

double dotProductOMP(double *A, double *B, int n)
{
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for(int i=0;i<n;i++)
    {
        sum += A[i] * B[i];
    }

    return sum;
}

double dotProductCUDA(double *A, double *B, int n)
{
    double *d_A;
    double *d_B;
    double *d_partial;
    double *h_partial;

    int threads = 256;
    int blocks = 256;

    int size = n * sizeof(double);
    int partialSize = threads * blocks * sizeof(double);

    h_partial = (double*)malloc(partialSize);

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_partial, partialSize);

    cudaMemcpy(d_A,A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,size,cudaMemcpyHostToDevice);

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    dotProductKernel<<<blocks,threads>>>(d_A,d_B,d_partial,n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernelTime;
    cudaEventElapsedTime(&kernelTime,start,stop);

    printf("Kernel Execution Time: %f ms\n",kernelTime);

    cudaMemcpy(h_partial,d_partial,partialSize,cudaMemcpyDeviceToHost);

    double sum=0.0;

    for(int i=0;i<threads*blocks;i++)
    {
        sum+=h_partial[i];
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_partial);
    free(h_partial);

    return sum;
}

int main()
{
    double *A;
    double *B;

    A=(double*)malloc(N*sizeof(double));
    B=(double*)malloc(N*sizeof(double));

    for(int i=0;i<N;i++)
    {
        A[i]=rand()/(double)RAND_MAX;
        B[i]=rand()/(double)RAND_MAX;
    }

    clock_t start_cpu,end_cpu;
    start_cpu=clock();

    double c=dotProductOMP(A,B,N);

    end_cpu=clock();
    double cpuTime=((double)(end_cpu-start_cpu))/CLOCKS_PER_SEC;

    clock_t start_gpu,end_gpu;
    start_gpu=clock();

    double d=dotProductCUDA(A,B,N);

    end_gpu=clock();
    double gpuTime=((double)(end_gpu-start_gpu))/CLOCKS_PER_SEC;

    printf("OpenMP Result: %f\n",c);
    printf("CUDA Result: %f\n",d);

    printf("CPU Time: %f seconds\n",cpuTime);
    printf("GPU Time: %f seconds\n",gpuTime);

    printf("Speedup: %f\n",cpuTime/gpuTime);

    free(A);
    free(B);

    return 0;
}