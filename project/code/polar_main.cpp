//TODO: Get .wav playback working before copying over Polar_main and POLAR objects

//CRT
#include <stdlib.h>
#include <Windows.h>

//Type defines
#include "misc/includes/win32_types.h"

//Debug
#include "library/debug/debug_macros.h"

//Includes
//Libraries
#include "library/dsp/dsp_wave.h"

//Polar
#include "polar_platform.cpp"

global bool GlobalRunning = true;

typedef struct POLAR_DATA
{
	WASAPI_DATA *WASAPI;
	WASAPI_BUFFER Buffer;
	i8 Channels;
	i32 SampleRate;
} POLAR_DATA;

int main(int argc, char *argv[])
{
	POLAR_DATA PolarEngine = {};

	PolarEngine.WASAPI = polar_WASAPI_Create(PolarEngine.Buffer);
	PolarEngine.Channels = PolarEngine.WASAPI->OutputWaveFormat->Format.nChannels;
	PolarEngine.SampleRate = PolarEngine.WASAPI->OutputWaveFormat->Format.nSamplesPerSec;

	OSCILLATOR *Osc = dsp_wave_CreateOscillator();
	dsp_wave_InitOscillator(Osc, SINE, PolarEngine.SampleRate);
	Osc->FrequencyCurrent = 880;

//TODO: How to flip this from 0 to 1 instead of using a defined name?
#if WIN32_METRICS
	LARGE_INTEGER PerformanceCounterFrequencyResult;
	QueryPerformanceFrequency(&PerformanceCounterFrequencyResult);
	i64 PerformanceCounterFrequency = PerformanceCounterFrequencyResult.QuadPart;
	
	LARGE_INTEGER LastCounter;
	QueryPerformanceCounter(&LastCounter);
	u64 LastCycleCount = __rdtsc();
#endif

	//TODO: Create audio callback function
	while(GlobalRunning)
	{
		polar_WASAPI_Render(PolarEngine.WASAPI, PolarEngine.Buffer, Osc);
#if WIN32_METRICS
        LARGE_INTEGER EndCounter;
        QueryPerformanceCounter(&EndCounter);
                
        u64 EndCycleCount = __rdtsc();

        i64 CounterElapsed = EndCounter.QuadPart - LastCounter.QuadPart;
        u64 CyclesElapsed = EndCycleCount - LastCycleCount;
        f32 MSPerFrame = (f32) (((1000.0f * (f32) CounterElapsed) / (f32) PerformanceCounterFrequency));
        f32 FramesPerSecond = (f32) PerformanceCounterFrequency / (f32) CounterElapsed;
        f32 MegaHzCyclesPerFrame = (f32) (CyclesElapsed / (1000.0f * 1000.0f));

        char MetricsBuffer[256];
        sprintf(MetricsBuffer, "WASAPI: %0.2f ms/frame\t %0.2f FPS\t %0.2f cycles(MHz)/frame\n", MSPerFrame, FramesPerSecond, MegaHzCyclesPerFrame);
        OutputDebugString(MetricsBuffer);

        LastCounter = EndCounter;
        LastCycleCount = EndCycleCount;	
#endif		
	}

	dsp_wave_DestroyOscillator(Osc);
	polar_WASAPI_Destroy(PolarEngine.WASAPI);

	return 0;
}