#ifndef polar_cuda_h
#define polar_cuda_h


//CUDA runtime 
#include <cuda_runtime.h>
#include <vector_types.h>

//CUDA SDK error handling code
#include "cuda/polar_cuda_error.h"

typedef struct CUDA_DEVICE
{
    i32 ID;
    cudaDeviceProp Properties;
} CUDA_DEVICE;

//Use the CUDA helper functions to find the best device on the system (usually one with the highest FLOPs rating)
i32 cuda_DeviceGet(CUDA_DEVICE *GPU, i32 ID);

void cuda_DevicePrint(CUDA_DEVICE *GPU);

f32 cuda_Sine(f32 X, u32 Block);

void cuda_Multiply(i32 GridWidth, i32 GridHeight, u64 MaxSize);

typedef struct CUDA_SINE
{
	float mag_c0;
	float mag_c1;
    float phase_c0;
    float phase_c1;
    float phase_c2;
} CUDA_SINE;

void cuda_PhaseCalc();

// void cuda_SineArray(u32 SampleCount, u32 SampleRate, u32 ThreadsPerPartial, POLAR_OSCILLATOR *Oscillator, f32 *HostResult);

#endif



