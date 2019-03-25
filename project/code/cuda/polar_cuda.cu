#ifndef polar_cuda_cu
#define polar_cuda_cu

#include "polar_cuda.h"

__global__ void cuda_Kernel_LaunchTest()
{
    printf("CUDA: Test Kernel launch [X: %d Y: %d]\n", blockIdx.x, blockIdx.y);
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


#endif