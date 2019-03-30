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

__global__ void cuda_Kernel_MultiplyTest(i32 GridWidth, i32 GridHeight, f32 *DataA, f32 *DataB, f32 *Result)
{
    //Create ID for every thread in the block
    i32 ThreadID = (blockIdx.y * GridHeight * GridWidth) + blockIdx.x * GridWidth + threadIdx.x;
	Result[ThreadID] = sqrt(DataA[ThreadID] * DataB[ThreadID] / 12.34567) * sin(DataA[ThreadID]);
}

//MaxSize must be integer multiple of GridWidth * GridHeight
void cuda_Multiply(i32 GridWidth, i32 GridHeight, u64 MaxSize)
{
	f32 *HostDataA       = (f32 *) malloc(sizeof(f32) * MaxSize);
	f32 *HostDataB       = (f32 *) malloc(sizeof(f32) * MaxSize);
	f32 *HostResult      = (f32 *) malloc(sizeof(f32) * MaxSize);
    
    f32 *DeviceDataA;
	f32 *DeviceDataB;
	f32 *DeviceResult;
    cudaMalloc(&DeviceDataA, (sizeof(f32) * MaxSize));
    cudaMalloc(&DeviceDataB, (sizeof(f32) * MaxSize));
    cudaMalloc(&DeviceResult, (sizeof(f32) * MaxSize));

    //Fill with random floats
	for(u64 i = 0; i < MaxSize; ++i)
	{
		HostDataA[i] = ((f32) pcg32_random() / ((f32) UINT32_MAX));
		HostDataB[i] = ((f32) pcg32_random() / ((f32) UINT32_MAX));
	}

	for(i32 Size = MaxSize; Size > (GridWidth * GridHeight); (Size /= 2))
	{
        i32 BlockGridWidth = GridWidth;
        i32 BlockGridHeight = ((Size / GridHeight) / BlockGridWidth);

        //Copy to device
        cudaMemcpy(DeviceDataA, HostDataA, (sizeof(f32) * Size), cudaMemcpyHostToDevice);
        cudaMemcpy(DeviceDataB, HostDataB, (sizeof(f32) * Size), cudaMemcpyHostToDevice);

        //Run kernel
        dim3 BlockGrid(BlockGridWidth, BlockGridHeight);
        dim3 ThreadBlock(GridHeight, 1);
        cuda_Kernel_MultiplyTest<<<BlockGrid, ThreadBlock>>>(GridWidth, GridHeight, DeviceDataA, DeviceDataB, DeviceResult);

        // Copy to host
        cudaMemcpy(HostResult, DeviceResult, (sizeof(f32) * Size), cudaMemcpyDeviceToHost);
    }

    cudaFree(DeviceDataA);
    cudaFree(DeviceDataB);
    cudaFree(DeviceResult);

    free(HostDataA);
    free(HostDataB);
    free(HostResult);
}

#endif