#ifndef polar_cuda_h
#define polar_cuda_h

//CUDA runtime 
#if CUDA
#include <cuda_runtime.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>
#endif

//CUDA SDK error handling code
#include "cuda/polar_cuda_error.h"
#include "cuda/helper_timer.h"

// Same max values as polar.h
#define MAX_SOURCES             256
#define MAX_VOICES              128
#define MAX_VOICES_PER_SOURCE   8
#define MAX_STRING_LENGTH       128
#define MAX_BUFFER_SIZE         2048
#define MAX_BUBBLE_COUNT        256
#define MAX_PARTIALS            16
#define MAX_BREAKPOINTS         64
#define MAX_WAV_SIZE            ((48000 * 600) * 2)
#define MAX_GRAINS              512 * 4
#define MAX_GRAIN_LENGTH        4800
#define MAX_GRAIN_PLAYLIST      6

// CUDA device constants
#define GPU_WARP_SIZE           32
#define GPU_SM_COUNT            13
#define GPU_THREADS             1024

#if CUDA
typedef struct CUDA_DEVICE
{
    i32 ID;
    cudaDeviceProp Properties;
} CUDA_DEVICE;

//Use the CUDA helper functions to find the best device on the system (usually one with the highest FLOPs rating)
i32 cuda_DeviceGet(CUDA_DEVICE *GPU, i32 ID);

void cuda_DevicePrint(CUDA_DEVICE *GPU);
#endif

// Sine wave partial function
void PhasorProcess(f32 *Buffer, size_t Frames, f32 *Phases, f32 Frequency, f32 Amplitude, f64 SizeOverSampleRate, f32 &LastPhaseValue);

// Bubbles structs
typedef struct CMP_BUBBLES_MODEL
{
    // Data
    f64 Frequency;
    f64 FrequencyBase;
    f64 Damping;
    f64 RiseFactor;

    // Filter
    f64 R2;
    f64 R2CosTheta;
    f64 R; 
    f64 C;
    f64 Y1;
    f64 Y2;

    size_t RiseCounter;
    f32 Amplitude;
    u16 IsRising;
    u16 IsSilent;

    void Init()
    {
        Frequency = 100;
        FrequencyBase = 100;
        Damping = 1.0;
        Amplitude = 1.0;
        RiseFactor = 0;
        RiseCounter = 0;
        IsRising = true;
        IsSilent = true;

        // Filter
        R2CosTheta = 0;
        R2 = 0;
        R = 0;
        C = 0;
        Y1 = 0;
        Y2 = 0;
    }

} CMP_BUBBLES_MODEL;


typedef struct CMP_BUBBLES_PULSE 
{
    f32 Density;
    f32 DensityBaseline;
    f32 Threshold;
    f32 Scale;
    f32 OneOverControlRate;
    u32 RandomSeed;

    void Init(f32 ControlRate, u32 Seed) 
    {
        Density             = 40;
        DensityBaseline     = 0.0f;
        Threshold           = 0.0f;
        Scale               = 0.0f;
        RandomSeed          = Seed;
        OneOverControlRate  = 1.0f / ControlRate;
    }

} CMP_BUBBLES_PULSE;


typedef struct CMP_BUBBLES_GENERATOR
{
    CMP_BUBBLES_MODEL Model;
    CMP_BUBBLES_PULSE Pulse;

} CMP_BUBBLES_GENERATOR;


typedef struct TPL_BUBBLES
{
    // Data
    f64 *Radii;
    f64 *Lambda;  
    f64 RadiusMaximum; // In mm
    f64 BubblesPerSec;
    f64 RiseCutoff;
    f64 LambdaSum;
    CMP_BUBBLES_GENERATOR *Generators;  
    u32 Count;

    void Init(f64 SampleRate, u32 BubbleCount = 1, f64 InputBubblesPerSec = 100, f64 InputRadius = 10, f64 InputRiseCutoff = 0.5);
    void Destroy();
    void CreateFromPool(MEMORY_POOL *GeneratorPool, MEMORY_POOL *RadiiPool, MEMORY_POOL *LambdaPool);
    void FreeFromPool(MEMORY_POOL *GeneratorPool, MEMORY_POOL *RadiiPool, MEMORY_POOL *LambdaPool);

} TPL_BUBBLES;

typedef struct CUDA_BUBBLES
{
    CMP_BUBBLES_GENERATOR *DeviceGenerators;
    f64 *DeviceLambda;
    f64 *DeviceRadii;
    f32 *DevicePulseBuffer;
    f32 *DeviceMixBuffer;
#if CUDA
    curandState *DeviceRNGStates;
#endif
} CUDA_BUBBLES;


// Bubble function prototypes
void cuda_BubblesCreate(CUDA_BUBBLES *GPU);
void cuda_BubblesDestroy(CUDA_BUBBLES *GPU);

// Computation
void cuda_BubblesComputeModel(CUDA_BUBBLES *GPU, TPL_BUBBLES *Bubbles, f64 SampleRate, size_t SamplesToWrite);
void cuda_BubblesComputeEvents(CUDA_BUBBLES *GPU, TPL_BUBBLES *Bubbles);

// Render
void cuda_BubblesUpdate(CUDA_BUBBLES *GPU, TPL_BUBBLES *Bubbles, f32 SampleRate, u32 Seed, f32 *Output, size_t BufferCount);
void cuda_BubblesUpdateV2(CUDA_BUBBLES *GPU, TPL_BUBBLES *Bubbles, f32 SampleRate, u32 Seed, f32 *Output, size_t BufferCount);

#endif



