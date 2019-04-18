#ifndef polar_cuda_cu
#define polar_cuda_cu

#include "../core/core.h"
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

// f32 cuda_Sine(f32 X, u32 Block)
// {
//     f32 *Result;
//     cudaMallocManaged(&Result, Block * sizeof(f32));

//     cuda_Kernel_Sine<<<1, Block>>>(Result, X);
//     cudaDeviceSynchronize();

//     f32 Value = Result[0];

//     cudaFree(Result);

//     return Value;
// }

// __global__ void cuda_Kernel_MultiplyTest(i32 GridWidth, i32 GridHeight, f32 *DataA, f32 *DataB, f32 *Result)
// {
//     //Create ID for every thread in the block
//     i32 ThreadID = (blockIdx.y * GridHeight * GridWidth) + blockIdx.x * GridWidth + threadIdx.x;
// 	Result[ThreadID] = sqrt(DataA[ThreadID] * DataB[ThreadID] / 12.34567) * sin(DataA[ThreadID]);
// }

// //MaxSize must be integer multiple of GridWidth * GridHeight
// void cuda_Multiply(i32 GridWidth, i32 GridHeight, u64 MaxSize)
// {
// 	f32 *HostDataA       = (f32 *) malloc(sizeof(f32) * MaxSize);
// 	f32 *HostDataB       = (f32 *) malloc(sizeof(f32) * MaxSize);
// 	f32 *HostResult      = (f32 *) malloc(sizeof(f32) * MaxSize);
    
//     f32 *DeviceDataA;
// 	f32 *DeviceDataB;
// 	f32 *DeviceResult;
//     cudaMalloc(&DeviceDataA, (sizeof(f32) * MaxSize));
//     cudaMalloc(&DeviceDataB, (sizeof(f32) * MaxSize));
//     cudaMalloc(&DeviceResult, (sizeof(f32) * MaxSize));

//     //Fill with random floats
// 	for(u64 i = 0; i < MaxSize; ++i)
// 	{
// 		HostDataA[i] = ((f32) pcg32_random() / ((f32) UINT32_MAX));
// 		HostDataB[i] = ((f32) pcg32_random() / ((f32) UINT32_MAX));
// 	}

// 	for(i32 Size = MaxSize; Size > (GridWidth * GridHeight); (Size /= 2))
// 	{
//         i32 BlockGridWidth = GridWidth;
//         i32 BlockGridHeight = ((Size / GridHeight) / BlockGridWidth);

//         //Copy to device
//         cudaMemcpy(DeviceDataA, HostDataA, (sizeof(f32) * Size), cudaMemcpyHostToDevice);
//         cudaMemcpy(DeviceDataB, HostDataB, (sizeof(f32) * Size), cudaMemcpyHostToDevice);

//         //Run kernel
//         dim3 BlockGrid(BlockGridWidth, BlockGridHeight);
//         dim3 ThreadBlock(GridHeight, 1);
//         cuda_Kernel_MultiplyTest<<<BlockGrid, ThreadBlock>>>(GridWidth, GridHeight, DeviceDataA, DeviceDataB, DeviceResult);

//         // Copy to host
//         cudaMemcpy(HostResult, DeviceResult, (sizeof(f32) * Size), cudaMemcpyDeviceToHost);
//     }

//     cudaFree(DeviceDataA);
//     cudaFree(DeviceDataB);
//     cudaFree(DeviceResult);

//     free(HostDataA);
//     free(HostDataB);
//     free(HostResult);
// }

// __device__ __host__ float phaseAtIdx(CUDA_SINE *Sine, unsigned idx) 
// {
//     return Sine->phase_c0 + idx * (Sine->phase_c1 + (idx * Sine->phase_c2));
// }

// __device__ __host__ float magAtIdx(CUDA_SINE *Sine, unsigned idx)
// {
//     return Sine->mag_c0 + (idx * Sine->mag_c1);
// }

// __device__ __host__ float valueAtIdx(CUDA_SINE *Sine, unsigned idx)
// {
//     return magAtIdx(Sine, idx) * sinf(phaseAtIdx(Sine, idx));
// }

// __device__ __host__ void newFrequencyAndDepth(CUDA_SINE *Sine, u32 BlockSize, u32 SampleRate, float startFreq, float endFreq, float startDepth, float endDepth) 
// {
//     float InverseBlockSize  = (1.0f / BlockSize);
//     float InverseSampleRate = (1.0f / SampleRate);


//     // compute phase function coefficients
//     // first, carry over the phase from the end of the previous buffer.
//     Sine->phase_c0 = phaseAtIdx(Sine, BlockSize);
//     // initial slope is w0
//     Sine->phase_c1 = startFreq * InverseSampleRate;
//     float endW = endFreq * InverseSampleRate;
//     // phase'(BlockSize) = endW
//     // phase_c1 + 2*t*phase_c2 = endW
//     // phase_c2 = (endW - phase_c1) / (2*BlockSize)
//     Sine->phase_c2 = (endW - Sine->phase_c1) * 0.5f * InverseBlockSize;
//     // compute magnitude function coefficients
//     Sine->mag_c0 = startDepth;
//     float deltaDepth = endDepth - startDepth;
//     Sine->mag_c1 = deltaDepth * InverseBlockSize;
// }

// __global__ void evaluateSynthVoiceBlockKernel(CUDA_SINE *Sine, unsigned samplesPerThread)
// {
//     unsigned threadIdWithinPartial = blockIdx.x;

//     for(unsigned sampleIdx = (threadIdWithinPartial * samplesPerThread); sampleIdx < ((threadIdWithinPartial + 1) * samplesPerThread); ++sampleIdx)
//     {
//         float sinusoid = valueAtIdx(Sine, sampleIdx);
//         printf("%f\t%u\n", sinusoid, sampleIdx);
//     }
// }


// void cuda_PhaseCalc()
// {
//     CUDA_SINE Sine = {};
//     newFrequencyAndDepth(&Sine, 4096, 48000, 440, 880, 1.f, 1.f);

// 	// printf("%f\n", Sine.mag_c0);
// 	// printf("%f\n", Sine.mag_c1);
//     // printf("%f\n", Sine.phase_c0);
//     // printf("%f\n", Sine.phase_c1);
//     // printf("%f\n", Sine.phase_c2);

//     unsigned NumPartials = 32;
//     unsigned threadsPerPartial = 512;
//     unsigned samplesPerThread = 4096 / threadsPerPartial;
    
//     evaluateSynthVoiceBlockKernel<<<threadsPerPartial, NumPartials>>>(&Sine, samplesPerThread);
//     checkCudaErrors(cudaGetLastError());
    
//     cudaDeviceSynchronize();
// }

// __device__ f32 polar_Device_PhaseWrap(f32 &Phase, f64 Size)
// {    
//     while(Phase >= Size)
//     {
//         Phase -= Size;
//     }

//     while(Phase < 0)
//     {
//         Phase += Size;
//     }

//     return Phase;
// }


// __global__ void cuda_Kernel_SineArray(POLAR_OSCILLATOR Oscillator, u32 SampleCount, u32 SampleRate, f32 *DeviceResult)
// {
//     //1D thread index
//     i32 ThreadID = (blockIdx.x * blockDim.x + threadIdx.x);
//     i32 SampleIndex = ThreadID % SampleCount;
    
//     f32 PhaseIndex = Oscillator.Frequency.Current / SampleRate;
//     f32 CurrentPhase = SampleIndex * PhaseIndex;


//     atomicAdd(&DeviceResult[SampleIndex], __sinf((TWO_PI32 * CurrentPhase)));
    
    
//     // DeviceResult[SampleIndex] = __sinf((TWO_PI32 / Oscillator.Frequency.Current) * (f32) SampleIndex);
//     // DeviceResult[ThreadID] = __sinf(Oscillator.PhaseCurrent * (f32) ThreadID);

//     // Oscillator.PhaseIncrement = Oscillator.TwoPiOverSampleRate * Oscillator.Frequency.Current; 
//     // Oscillator.PhaseCurrent += Oscillator.PhaseIncrement; //Increase phase by the calculated cycle increment
    
//     // polar_Device_PhaseWrap(Oscillator.PhaseCurrent, TWO_PI32);

// }

// void cuda_SineArray(u32 SampleCount, u32 SampleRate, u32 ThreadsPerPartial, POLAR_OSCILLATOR *Oscillator, f32 *HostResult)
// {
// 	f32 *DeviceResult;
//     cudaMalloc(&DeviceResult, (sizeof(f32) * SampleCount));

//     POLAR_OSCILLATOR DeviceOscillator = {};
//     DeviceOscillator.Waveform                   = Oscillator->Waveform;
//     DeviceOscillator.TwoPiOverSampleRate        = Oscillator->TwoPiOverSampleRate;
//     DeviceOscillator.PhaseCurrent               = Oscillator->PhaseCurrent;
//     DeviceOscillator.PhaseIncrement             = Oscillator->PhaseIncrement;
//     DeviceOscillator.Frequency                  = Oscillator->Frequency;

//     dim3 ThreadCount(ThreadsPerPartial, 1);
//     dim3 BlockCount((SampleCount / ThreadsPerPartial), 1);

    
//     cuda_Kernel_SineArray<<<BlockCount, ThreadCount>>>(DeviceOscillator, SampleCount, SampleRate, DeviceResult);
//     cudaDeviceSynchronize(); //Wait for kernel to end

//     cudaMemcpy(HostResult, DeviceResult, (sizeof(f32) * SampleCount), cudaMemcpyDeviceToHost);

//     cudaFree(DeviceResult);
// }



#endif