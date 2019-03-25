#ifndef polar_cuda_cu
#define polar_cuda_cu

#include "polar_cuda.h"

__global__ void cuda_Kernel_LaunchTest()
{
    printf("CUDA: Test Kernel launch [X: %d Y: %d]\n", blockIdx.x, blockIdx.y);
}

__device__ f32 cuda_Series(f32 x)
{
    f32 y = 0;
    
    f32 C[5] = {0.0000024609388329975758276f, -0.00019698112762578435577f, 0.0083298294559966612167f, -0.16666236485125293496f, 0.99999788400553332261f};

    
    y = C[0];
    y = y * x + C[1];
    y = y * x + C[2];
    y = y * x + C[3];
    y = y * x + C[4];

    return y;
}

__device__ f32 cuda_MiniMax(f32 x)
{
    if(x < -PI32)
    {
        x += TWO_PI32;
    }

    else if (x > PI32)
    {
        x -= TWO_PI32;   
    }

    return x * cuda_Series(x * x);
}


__global__ void cuda_Kernel_Sine(f32 *Result, f32 X)
{
    Result[0] = cuda_MiniMax(X);
}
  
//Get GPU
i32 cuda_DeviceGet(CUDA_DEVICE *GPU, i32 ID)
{
    //TODO: Implement gpuGetMaxGflopsDeviceId if no ID is supplied
    GPU->ID = ID;
    if(ID == -1)
    {
        // GPU->ID = gpuGetMaxGflopsDeviceId();
    }

    //Get current device info
    checkCudaErrors(cudaGetDevice(&GPU->ID));
    checkCudaErrors(cudaGetDeviceProperties(&GPU->Properties, GPU->ID));

    cuda_Kernel_LaunchTest<<<1, 1>>>();
    cudaDeviceSynchronize();

    return 0;
}

f32 cuda_Sine(f32 X, u32 Block)
{
    f32 *Result;
    cudaMallocManaged(&Result, Block * sizeof(f32));

    cuda_Kernel_Sine<<<1, Block>>>(Result, X);
    cudaDeviceSynchronize();

    f32 Value = Result[0];

    cudaFree(Result);

    return Value;
}


#endif