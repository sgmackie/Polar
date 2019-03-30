#ifndef polar_cuda_h
#define polar_cuda_h

#include "../polar.h"
#include "../../external/pcg_basic.h"

//CUDA runtime 
#include <cuda_runtime.h>

//CUDA SDK error handling code
#include "polar_cuda_error.h"

typedef struct CUDA_DEVICE
{
    i32 ID;
    cudaDeviceProp Properties;
} CUDA_DEVICE;

//Use the CUDA helper functions to find the best device on the system (usually one with the highest FLOPs rating)
i32 cuda_DeviceGet(CUDA_DEVICE *GPU, i32 ID);

f32 cuda_Sine(f32 X, u32 Block);

void cuda_Multiply(i32 GridWidth, i32 GridHeight, u64 MaxSize);

#endif



