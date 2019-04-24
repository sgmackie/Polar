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


typedef struct OSCILLATOR
{
    //Flags
    typedef enum TYPE
    {
        SQUARE      = 1 << 0,
        SINE        = 1 << 1,
        TRIANGLE    = 1 << 2,
        SAWTOOTH    = 1 << 3,
    } TYPE;

    //Data
    i32         Flag;
    f64         Phasor;
    f64         PhaseIncrement;
    f64         SizeOverSampleRate;
    f64         Frequency;

    //Functions
    void Init(i32 Type, u32 SampleRate, f64 InputFrequency, f64 Limit = TWO_PI32)
    {
        Flag |= Type;
        Phasor = 0;
        PhaseIncrement = 0;
        SizeOverSampleRate = Limit / SampleRate;
        Frequency = InputFrequency;
    }


} OSCILLATOR;


void cuda_SineArray(u32 SampleCount, u32 SampleRate, u32 ThreadsPerPartial, f32 *HostResult);

#endif



