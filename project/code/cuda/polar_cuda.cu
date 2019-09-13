#ifndef polar_cuda_cu
#define polar_cuda_cu

#include <stdint.h>
#include <stdio.h>
#include "../co/types.h"
#include "../co/core_internal.h"
#define PI32 3.14159265358979323846
#define TWO_PI32 (2.0 * PI32)
#include "../polar_cuda.h"

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
    Result[0] = __sinf(X);
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

void cuda_DevicePrint(CUDA_DEVICE *GPU)
{
    //Name
    printf("CUDA: Name:                          \t%s\n",     GPU->Properties.name);
	printf("CUDA: Major revision number:         \t%d\n",     GPU->Properties.major);
	printf("CUDA: Minor revision number:         \t%d\n",     GPU->Properties.minor);
    
    //Processors
    printf("CUDA: Number of multiprocessors:     \t%d\n",     GPU->Properties.multiProcessorCount);
    printf("CUDA: Clock rate:                    \t%d\n",     GPU->Properties.clockRate);

    //Memory
    printf("CUDA: Total global memory:           \t%zu\n",    GPU->Properties.totalGlobalMem);
    printf("CUDA: Total constant memory:         \t%zu\n",    GPU->Properties.totalConstMem);
	printf("CUDA: Total shared memory per block: \t%zu\n",    GPU->Properties.sharedMemPerBlock);
	printf("CUDA: Total registers per block:     \t%d\n",     GPU->Properties.regsPerBlock);
	printf("CUDA: Warp size:                     \t%d\n",     GPU->Properties.warpSize);
	printf("CUDA: Maximum memory pitch:          \t%zu\n",    GPU->Properties.memPitch);
	printf("CUDA: Texture alignment:             \t%zu\n",    GPU->Properties.textureAlignment);
    
    //Threads
    printf("CUDA: Maximum threads per block:     \t%d\n",     GPU->Properties.maxThreadsPerBlock);
    for(i32 i = 0; i < 3; ++i) 
    {
		printf("CUDA: Maximum dimension %d of block:   \t%d\n", i,  GPU->Properties.maxThreadsDim[i]);
	}
    for(i32 i = 0; i < 3; ++i) 
    {
		printf("CUDA: Maximum dimension %d of grid:    \t%d\n", i,  GPU->Properties.maxGridSize[i]);
    }
        
    //Various
	printf("CUDA: Concurrent copy and execution: \t%s\n",     (GPU->Properties.deviceOverlap ? "Yes" : "No"));
	printf("CUDA: Kernel execution timeout:      \t%s\n",     (GPU->Properties.kernelExecTimeoutEnabled ? "Yes" : "No"));
}

// __global__ void CalculateSine(f32 *Buffer, f32 *Phases, f32 Amplitude)
// {
// 	int ThreadIndex = (blockIdx.x * blockDim.x + threadIdx.x);
//     Buffer[ThreadIndex] = (__sinf(Phases[ThreadIndex]) * Amplitude);
// }

__global__ void CalculateSine(f32 *Buffer, size_t Count, f32 *Phases, f32 Amplitude)
{
    __shared__ f32 Sample;

    int Min = threadIdx.x + blockDim.x * blockIdx.x;
    int Stride = blockDim.x * gridDim.x;

    if(threadIdx.x == 0) 
    {
        // ! Not actually valid
        Sample = __sinf(Phases[0]) * Amplitude;
    }
    __syncthreads();

    for(size_t i = Min; i < Count; i += Stride) 
    {
        Buffer[i] = Sample;
    }
}


void PhasorProcess(f32 *Buffer, size_t Frames, f32 *Phases, f32 Frequency, f32 Amplitude, f64 SizeOverSampleRate, f32 &LastPhaseValue)
{
    //!Check this
    size_t KernelFrames = Frames;

	//Calculate phase increments
	f32 PhaseIncrement = (Frequency * SizeOverSampleRate);
	f32 CurrentPhase = LastPhaseValue + PhaseIncrement;
	for(size_t i = 0; i < KernelFrames; ++i)
	{
		Phases[i] = CurrentPhase;
		CurrentPhase += PhaseIncrement;
        
        //Wrap
        while(CurrentPhase >= TWO_PI32)
        {
            CurrentPhase -= TWO_PI32;
        }
        while(CurrentPhase < 0)
        {
            CurrentPhase += TWO_PI32;
        }          
    }
    
    //Allocate device buffers
    f32 *DeviceBuffer;
	f32 *DevicePhases;
	size_t DeviceSize = (sizeof(f32) * KernelFrames);
	cudaMalloc((void **) &DeviceBuffer, DeviceSize);
	cudaMalloc((void **) &DevicePhases, DeviceSize);
	cudaMemcpy(DeviceBuffer, Buffer, DeviceSize, cudaMemcpyHostToDevice);
    cudaMemcpy(DevicePhases, Phases, DeviceSize, cudaMemcpyHostToDevice);
    
    //!Check core count
	// size_t BlockSize = 4;
	// size_t BlockCount = (KernelFrames / BlockSize + (KernelFrames % BlockSize == 0 ? 0 : 1));


    const int warpSize = 32;
    const int maxGridSize = 104; // this is 8 blocks per MP for a Telsa C2050
    int warpCount = (KernelFrames / warpSize) + (((KernelFrames % warpSize) == 0) ? 0 : 1);
    int warpPerBlock = max(1, min(4, warpCount));
    int threadCount = warpSize * warpPerBlock;
    int blockCount = min( maxGridSize, max(1, warpCount/warpPerBlock) );

    
    dim3 GridDim  = dim3(blockCount, 1, 1);
    dim3 BlockDim = dim3(threadCount, 1, 1);


    //Kernel
    //!Works, now investigate having an oscillator per thread and summing 
	CalculateSine<<<GridDim, BlockDim>>>(DeviceBuffer, KernelFrames, DevicePhases, Amplitude);

    //Copy back to host
    cudaMemcpy(Buffer, DeviceBuffer, DeviceSize, cudaMemcpyDeviceToHost);
	cudaFree(DeviceBuffer);
	cudaFree(DevicePhases);

	//Save last phase value
	LastPhaseValue = Phases[KernelFrames - 1];

    return;
}


// #if __CUDA_ARCH__ < 600
// __device__ double atomicAddF64(double* address, double val)
// {
//     unsigned long long int* address_as_ull =
//                               (unsigned long long int*)address;
//     unsigned long long int old = *address_as_ull, assumed;

//     do {
//         assumed = old;
//         old = atomicCAS(address_as_ull, assumed,
//                         __double_as_longlong(val +
//                                __longlong_as_double(assumed)));

//     // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
//     } while (assumed != old);

//     return __longlong_as_double(old);
// }
// #endif

// __global__ void cuda_kernel_LambdaAdd(f64 *Lambda, f64 Curve, f64 *LambdaSum, int Count)
// {
//     //Holds intermediates in shared memory reduction
//     __syncthreads();
//     __shared__ f64 SharedBuffer[WARP_SIZE];
//     int GlobalIndex = blockIdx.x * blockDim.x + threadIdx.x;
//     int ThreadLane = threadIdx.x % WARP_SIZE;
//     f64 Temporary;
//     while(GlobalIndex < Count)
//     {   
//         Lambda[GlobalIndex] /= Curve;
        

//     	// All threads in a block of 1024 take an element
//         Temporary = Lambda[GlobalIndex];
//         // All warps in this block (32) compute the sum of all
//         // threads in their warp
//         for(int delta = WARP_SIZE/2; delta > 0; delta /= 2)
//         {
//             Temporary+= __shfl_xor_sync(0xffffffff, Temporary, delta);
//         }
//         // Write all 32 of these partial sums to shared memory
//         if(ThreadLane == 0)
//         {
//             SharedBuffer[threadIdx.x / WARP_SIZE] = Temporary;
//         }
//         __syncthreads();
//         // Add the remaining 32 partial sums using a single warp
//         if(threadIdx.x < WARP_SIZE) 
//         {
//             Temporary = SharedBuffer[threadIdx.x];
//             for(int delta = WARP_SIZE / 2; delta > 0; delta /= 2)
//             {  
//                 Temporary += __shfl_xor_sync(0xffffffff, Temporary, delta);
//             }
//         }
//         // Add this block's sum to the total sum
//         if(threadIdx.x == 0)
//         {
//             atomicAddF64(LambdaSum, Temporary);
//         }
//         // Jump ahead 1024 * #SMs to the next region of numbers to sum
//         GlobalIndex += blockDim.x * gridDim.x;
//         __syncthreads();
//     } 
// }


__device__ void cuda_device_Coefficients(CMP_BUBBLES_MODEL *Model, f64 SampleRate)
{
    f64 R               = (f64) exp(-Model->Damping / SampleRate);
    Model->R2           = R * R;
    Model->R2CosTheta   = (f64) (2 * cos(TWO_PI32 * Model->Frequency / SampleRate) * R);
    Model->C            = (f64) sin((TWO_PI32 * Model->Frequency / SampleRate) * R);
    Model->R            = Model->C * Model->Amplitude;    
}


__global__ void cuda_kernel_ComputeModel(CMP_BUBBLES_GENERATOR *Generators, f64 *Radii, f64 RadiusMaximum, f64 AmplitudeExponent, f64 SampleRate, size_t SamplesToWrite, int Count)
{
    // Grid-stride loop
    int Start           = blockIdx.x * blockDim.x + threadIdx.x;
    int ThreadsPerGrid  = blockDim.x * gridDim.x;

    for(int i = Start; i < Count; i += ThreadsPerGrid) 
    {
        // Calculate frequency, amplitude and damping according to bubble radius
        f64 Frequency                       = (f64) 3 / Radii[i];
        f64 Amplitude                       = (f64) (pow(Radii[i] / (RadiusMaximum / 1000), AmplitudeExponent));
        f64 Damping                         = (f64) (Frequency * (0.043 + sqrt(Frequency) / 721));

        // Assign and find rising frequency factor
        Generators[i].Model.Frequency       = Frequency;
        Generators[i].Model.FrequencyBase   = Frequency;
        Generators[i].Model.Amplitude       = Amplitude;
        Generators[i].Model.Damping         = Damping;
        Generators[i].Model.RiseFactor      = ((SampleRate / SamplesToWrite) / (Generators[i].Model.RiseAmplitude * Damping));

        // Compute Coefficients
        cuda_device_Coefficients(&Generators[i].Model, SampleRate);  
    }        
}



void cuda_BubblesComputeModel(CUDA_BUBBLES *GPU, TPL_BUBBLES *Bubbles, f64 SampleRate, size_t SamplesToWrite)
{
    // Locals
    size_t Count            = Bubbles->Count;    

    // Find radius ranges
    f64 LogMinimum                  = log(Bubbles->RadiusMinimum / 1000);
    f64 LogMaximum                  = log(Bubbles->RadiusMaximum / 1000);
    f64 LogSize                     = (LogMaximum - LogMinimum) / (Bubbles->Count - 1);

    // Reset sum
    Bubbles->LambdaSum = 0;
    f64 Curve = 0;

    // Calculate radii and lambda values
    for(size_t i = 0; i < Bubbles->Count; ++i)
    {
        // Calculate radius from minimum size
        Bubbles->Radii[i] = exp(LogMinimum + i * LogSize);

        // Calculate lambda values and sum together
        Bubbles->Lambda[i] = 1.0 / pow((1000 * Bubbles->Radii[i] / Bubbles->RadiusMinimum), Bubbles->LambdaExponent);

        // Save max value
        if(Bubbles->Lambda[i] > Curve) 
        {
            Curve = Bubbles->Lambda[i];
        }

        // Sum
        Bubbles->LambdaSum += Bubbles->Lambda[i];        
    }

    // Divive by maxium value
    for(size_t i = 0; i < Bubbles->Count; ++i)
    {
        Bubbles->Lambda[i] /= Curve;
    }

    // Copy host to device memory
    cudaMemcpy(GPU->DeviceGenerators, Bubbles->Generators, (sizeof(CMP_BUBBLES_GENERATOR) * Count), cudaMemcpyHostToDevice);
    cudaMemcpy(GPU->DeviceRadii, Bubbles->Radii, (sizeof(f64) * Count), cudaMemcpyHostToDevice);

    // Launch kernel
    cuda_kernel_ComputeModel<<<GPU_WARP_SIZE * GPU_SM_COUNT, 1024>>>(GPU->DeviceGenerators, GPU->DeviceRadii, Bubbles->RadiusMaximum, Bubbles->AmplitudeExponent, SampleRate, SamplesToWrite, Count);
    // cuda_kernel_ComputeModel<<<1, 1>>>(DeviceGenerators, DeviceRadii, Bubbles->RadiusMaximum, Bubbles->AmplitudeExponent, SampleRate, SamplesToWrite, Count);

    // Copy back to host mmemory
    cudaMemcpy(Bubbles->Generators, GPU->DeviceGenerators, (sizeof(CMP_BUBBLES_GENERATOR) * Count), cudaMemcpyDeviceToHost);
    cudaMemcpy(Bubbles->Radii, GPU->DeviceRadii, (sizeof(f64) * Count), cudaMemcpyDeviceToHost);  
}



__global__ void cuda_kernel_ComputeEvents(CMP_BUBBLES_GENERATOR *Generators, f64 *Lambda, f64 BubblesPerSec, f64 ProbabilityExponent, f64 Average, int Count)
{

    // Grid-stride loop
    int Start           = blockIdx.x * blockDim.x + threadIdx.x;
    int ThreadsPerGrid  = blockDim.x * gridDim.x;

    for(int i = Start; i < Count; i += ThreadsPerGrid) 
    {
        f64 Mean = Average * (BubblesPerSec * Lambda[i]);
        Generators[i].Pulse.Density = Mean * ProbabilityExponent;        
    }    
}


void cuda_BubblesComputeEvents(CUDA_BUBBLES *GPU, TPL_BUBBLES *Bubbles) 
{
    // Get the average from the lamba summing
    f64 Average = Bubbles->LambdaSum;
    if(Average == 0)
    {
        Average = 1;
    }

    // Locals
    size_t Count            = Bubbles->Count;
    f64 BubblesPerSec       = Bubbles->BubblesPerSec;
    f64 ProbabilityExponent = Bubbles->ProbabilityExponent;

    // Copy host to device memory
    cudaMemcpy(GPU->DeviceGenerators, Bubbles->Generators, (sizeof(CMP_BUBBLES_GENERATOR) * Count), cudaMemcpyHostToDevice);
    cudaMemcpy(GPU->DeviceLambda, Bubbles->Lambda, (sizeof(f64) * Count), cudaMemcpyHostToDevice);

    // Launch kernel
    cuda_kernel_ComputeEvents<<<GPU_WARP_SIZE * GPU_SM_COUNT, 1024>>>(GPU->DeviceGenerators, GPU->DeviceLambda, BubblesPerSec, ProbabilityExponent, Average, Count);
    // cuda_kernel_ComputeEvents<<<1, 1>>>(DeviceGenerators, DeviceLambda, BubblesPerSec, ProbabilityExponent, Average, Count);

    // Copy back to host mmemory
    cudaMemcpy(Bubbles->Generators, GPU->DeviceGenerators, (sizeof(CMP_BUBBLES_GENERATOR) * Count), cudaMemcpyDeviceToHost);
    cudaMemcpy(Bubbles->Lambda, GPU->DeviceLambda, (sizeof(f64) * Count), cudaMemcpyDeviceToHost);
}


__device__ f32 cuda_device_PulseSample(CMP_BUBBLES_PULSE *Pulse, f32 RNG)
{
    f32 Result          = 0;

    // Density has shifted - recompute event threshold
    if(Pulse->Density != Pulse->DensityBaseline) 
    {
        Pulse->Threshold         = Pulse->Density * Pulse->OneOverControlRate;
        Pulse->Scale             = (Pulse->Threshold > 0.0 ? 2.0 / Pulse->Threshold : 0.0);
        Pulse->DensityBaseline   = Pulse->Density;
    }

    // Seed RNG
    f32 RandomValue         = RNG;

    // Multiply result
    Result = Pulse->Amplitude * (RandomValue < Pulse->Threshold ? RandomValue * Pulse->Scale - 1.0 : 0.0);

    return Result;    
}


__global__ void cuda_kernel_Pulse(CMP_BUBBLES_GENERATOR *Generators, curandState *DeviceRNGStates, f32 *MixBuffer, u32 Seed, f32 MasterAmplitude, f32 SampleRate, f32 RiseCutoff, size_t i, int BufferCount)
{
    // Grid-stride loop
    int Start           = blockIdx.x * blockDim.x + threadIdx.x;
    int ThreadsPerGrid  = blockDim.x * gridDim.x;

    // Locals
    __shared__ f32 PulseBuffer[MAX_BUFFER_SIZE];
    f32 Sample = 0.0f;
    f32 Random = 0.0f;
    f32 Impulse = 0;
    f32 Threshold = 0.00000001;

    // RNG initialise
    curandState *State = DeviceRNGStates + Start;
    curand_init(Seed, Start, 0, State);

    for(int k = Start; k < BufferCount; k += ThreadsPerGrid) 
    {
        Random = curand_uniform(State);
        Sample = cuda_device_PulseSample(&Generators[i].Pulse, Random);
        Sample *= MasterAmplitude;
        PulseBuffer[k] = Sample;
        
        if((Impulse = abs(PulseBuffer[k])) >= Threshold) 
        {
            // Start rising from the base frequency
            Generators[i].Model.IsSilent = false;
            Generators[i].Model.RiseCounter = 0;
            Generators[i].Model.Frequency = Generators[i].Model.FrequencyBase;
            cuda_device_Coefficients(&Generators[i].Model, SampleRate);

            if(Impulse > RiseCutoff) 
            {
                Generators[i].Model.IsRising = true;
            } 
            else 
            {
                Generators[i].Model.IsRising = false;
            }
        }        
        
        MixBuffer[k] = PulseBuffer[k];
        // __syncthreads();
        // atomicAdd(&MixBuffer[k], PulseBuffer[k]);
    }

}

__global__ void cuda_kernel_Silence(CMP_BUBBLES_GENERATOR *Generators, f32 SampleRate, int BubbleCount)
{
    // Grid-stride loop
    int Start           = blockIdx.x * blockDim.x + threadIdx.x;
    int ThreadsPerGrid  = blockDim.x * gridDim.x;

    // Locals
    f32 Threshold = 0.00000001;

    for(int i = Start; i < BubbleCount; i += ThreadsPerGrid) 
    {
        if(Generators[i].Model.IsRising) 
        {
            Generators[i].Model.Frequency = 
            (f64) (Generators[i].Model.FrequencyBase * (1.0 + (++Generators[i].Model.RiseCounter) / 
            Generators[i].Model.RiseFactor));
            cuda_device_Coefficients(&Generators[i].Model, SampleRate);
        }

        // Final check if silent using the last filter chache values
        if(Generators[i].Model.IsSilent) 
        {
            if(abs(Generators[i].Model.Y1) >= Threshold || abs(Generators[i].Model.Y2) >= Threshold) 
            {
                Generators[i].Model.IsSilent = false;
            }
        }  
    }
}    


__global__ void cuda_kernel_Model(CMP_BUBBLES_GENERATOR *Generators, f32 *MixBuffer, size_t i, int BufferCount)
{
    // Grid-stride loop
    int Start           = blockIdx.x * blockDim.x + threadIdx.x;
    int ThreadsPerGrid  = blockDim.x * gridDim.x;

    // Locals
    f32 Sample = 0.0f;
    f32 TempY1 = 0.0f;
    f32 TempY2 = 0.0f;
    f32 Pulse = 0;

    TempY1 = Generators[i].Model.Y1;
    TempY2 = Generators[i].Model.Y2;   

    for(int k = Start; k < BufferCount; k += ThreadsPerGrid) 
    {
        Pulse = MixBuffer[k];
        Sample      = (Generators[i].Model.R2CosTheta * TempY1 - Generators[i].Model.R2 * TempY2 + Generators[i].Model.R * Pulse);
        TempY2          = TempY1;
        TempY1          = Sample;
        
        MixBuffer[k]    += Sample;
        // reduceOutputs(MixBuffer, i, k, BufferCount, Sample);

        // __syncthreads();
        // atomicAdd(&MixBuffer[k], PulseBuffer[k]);
    }     
    
    Generators[i].Model.Y1 = TempY1;
    Generators[i].Model.Y2 = TempY2;    
}

void cuda_BubblesUpdate(CUDA_BUBBLES *GPU, TPL_BUBBLES *Bubbles, f32 SampleRate, u32 Seed, f32 *Output, size_t BufferCount)
{    
    // Copy host to device memory
    cudaMemcpy(GPU->DeviceGenerators, Bubbles->Generators, (sizeof(CMP_BUBBLES_GENERATOR) * Bubbles->Count), cudaMemcpyHostToDevice);
    cudaMemcpy(GPU->DeviceMixBuffer, Output, (sizeof(f32) * BufferCount), cudaMemcpyHostToDevice);
    
    // Launch kernels
    for(size_t i = 0; i < Bubbles->Count; ++i)
    {
        cuda_kernel_Pulse<<<1, 1>>>(GPU->DeviceGenerators, GPU->DeviceRNGStates, GPU->DeviceMixBuffer, Seed, Bubbles->Amplitude, SampleRate, Bubbles->RiseCutoff, i, BufferCount);
        cuda_kernel_Model<<<1, 1>>>(GPU->DeviceGenerators, GPU->DeviceMixBuffer, i, BufferCount);     
    }

    // Silence check delayed
    cuda_kernel_Silence<<<32 * GPU_SM_COUNT, 1024>>>(GPU->DeviceGenerators, SampleRate, Bubbles->Count);

    // Copy back to host mmemory
    cudaMemcpy(Output, GPU->DeviceMixBuffer, (sizeof(f32) * BufferCount), cudaMemcpyDeviceToHost);
    cudaMemcpy(Bubbles->Generators, GPU->DeviceGenerators, (sizeof(CMP_BUBBLES_GENERATOR) * Bubbles->Count), cudaMemcpyDeviceToHost);
}

void cuda_BubblesCreate(CUDA_BUBBLES *GPU)
{
    cudaMalloc((void **) &GPU->DeviceGenerators, (sizeof(CMP_BUBBLES_GENERATOR) * MAX_BUBBLE_COUNT));
    cudaMalloc((void **) &GPU->DeviceLambda, (sizeof(f64) * MAX_BUBBLE_COUNT));
    cudaMalloc((void **) &GPU->DeviceRadii, (sizeof(f64) * MAX_BUBBLE_COUNT));
    cudaMalloc((void **) &GPU->DeviceMixBuffer, (sizeof(f32) * MAX_BUFFER_SIZE));
    cudaMalloc((void **) &GPU->DeviceRNGStates, (sizeof(curandState) * GPU_WARP_SIZE * GPU_SM_COUNT * GPU_THREADS));
}

void cuda_BubblesDestroy(CUDA_BUBBLES *GPU)
{
    cudaFree(GPU->DeviceGenerators);
    cudaFree(GPU->DeviceLambda);
    cudaFree(GPU->DeviceRadii);
    cudaFree(GPU->DeviceMixBuffer);
    cudaFree(GPU->DeviceRNGStates);
}


#endif