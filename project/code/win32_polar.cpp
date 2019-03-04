//Perfomance defines (will change stack allocation sizes for things like max sources per container)
#define MAX_STRING_LENGTH 64
#define MAX_CHANNELS 4
#define MAX_CONTAINERS 4
#define MAX_SOURCES 128
#define MAX_BREAKPOINTS 64
#define MAX_ENVELOPES 4
#define DEFAULT_AMPLITUDE 0.8
#define DEFAULT_SAMPLERATE 48000

#include "polar.h"
#include "../external/external_code.h"
#include "win32_polar.h"

global_scope char AssetPath[MAX_STRING_LENGTH] = {"../../data/"};
global_scope i64 GlobalPerformanceCounterFrequency;

#include "polar.cpp"

WASAPI_DATA *win32_WASAPI_Create(MEMORY_ARENA *Arena, i32 &FramesAvailable)
{
    WASAPI_DATA *Result = 0;
    Result = (WASAPI_DATA *) memory_arena_Push(Arena, Result, (sizeof (WASAPI_DATA)));

	Result->HR = CoInitializeEx(0, COINIT_MULTITHREADED);
	HR_TO_RETURN(Result->HR, "Failed to initialise COM", nullptr);

    Result->RenderEvent = CreateEvent(0, 0, 0, 0);
	if(!Result->RenderEvent)
	{
		HR_TO_RETURN(Result->HR, "Failed to create event", nullptr);
	}

    Result->HR = CoCreateInstance(__uuidof(MMDeviceEnumerator), NULL, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator), (void **) &Result->DeviceEnumerator);
    HR_TO_RETURN(Result->HR, "Failed to create device COM", nullptr);

    Result->HR = Result->DeviceEnumerator->GetDefaultAudioEndpoint(eRender, eConsole, &Result->AudioDevice);
    HR_TO_RETURN(Result->HR, "Failed to get default audio endpoint", nullptr);

	Result->HR = Result->AudioDevice->Activate(__uuidof(IAudioClient), CLSCTX_ALL, NULL, (void**) &Result->AudioClient);
	HR_TO_RETURN(Result->HR, "Failed to activate audio endpoint", nullptr);

	Result->HR = Result->AudioClient->GetDevicePeriod(&Result->DevicePeriod, &Result->DevicePeriodMin);
	HR_TO_RETURN(Result->HR, "Failed to get device period for callback buffer", nullptr);

    Result->HR = Result->AudioClient->GetMixFormat(&Result->DeviceWaveFormat);
    HR_TO_RETURN(Result->HR, "Failed to get default wave format for audio client", nullptr);

	Result->HR = Result->AudioClient->Initialize(AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_EVENTCALLBACK, Result->DevicePeriodMin, 0, Result->DeviceWaveFormat, NULL);
    HR_TO_RETURN(Result->HR, "Failed to initialise audio client", nullptr);

	Result->HR = Result->AudioClient->SetEventHandle(Result->RenderEvent);
	HR_TO_RETURN(Result->HR, "Failed to set rendering event", nullptr);

    Result->HR = Result->AudioClient->GetBufferSize(&Result->OutputBufferFrames);
	HR_TO_RETURN(Result->HR, "Failed to get maximum read buffer size for audio client", nullptr);

    Result->HR = Result->AudioClient->GetCurrentPadding(&Result->PaddingFrames);
	HR_TO_RETURN(Result->HR, "Failed to get buffer padding size", nullptr);

	Result->HR = Result->AudioClient->GetService(__uuidof(IAudioRenderClient), (void**) &Result->AudioRenderClient);
	HR_TO_RETURN(Result->HR, "Failed to assign client to render client", nullptr);	

	Result->HR = Result->AudioClient->Reset();
	HR_TO_RETURN(Result->HR, "Failed to reset audio client before playback", nullptr);

	Result->HR = Result->AudioClient->Start();
	HR_TO_RETURN(Result->HR, "Failed to start audio client", nullptr);

    FramesAvailable = ((Result->OutputBufferFrames - Result->PaddingFrames) * Result->DeviceWaveFormat->nChannels);

    return Result;
}

void win32_WASAPI_Destroy(MEMORY_ARENA *Arena, WASAPI_DATA *WASAPI)
{
	WASAPI->AudioRenderClient->Release();
	WASAPI->AudioClient->Reset();
	WASAPI->AudioClient->Stop();
	WASAPI->AudioClient->Release();
	WASAPI->AudioDevice->Release();
	CloseHandle(WASAPI->RenderEvent);
	WASAPI->RenderEvent = 0;

	CoUninitialize();

    memory_arena_Reset(Arena);
    memory_arena_Pull(Arena);
}

void win32_WASAPI_Callback(WASAPI_DATA *WASAPI, POLAR_ENGINE Engine, POLAR_MIXER *Mixer, POLAR_RINGBUFFER *CallbackBuffer)
{
    WaitForSingleObject(WASAPI->RenderEvent, INFINITE);

    if(polar_ringbuffer_WriteCheck(CallbackBuffer))
    {
	    WASAPI->HR = WASAPI->AudioClient->GetCurrentPadding(&WASAPI->PaddingFrames);
	    HR_TO_RETURN(WASAPI->HR, "Couldn't get current padding", NONE);

	    Engine.BufferFrames = (WASAPI->OutputBufferFrames - WASAPI->PaddingFrames);

        Callback(Engine, Mixer, polar_ringbuffer_WriteData(CallbackBuffer));
        polar_ringbuffer_WriteFinish(CallbackBuffer);
    }

    if(polar_ringbuffer_ReadCheck(CallbackBuffer))
    {
        BYTE *BYTEBuffer = 0;

        WASAPI->HR = WASAPI->AudioRenderClient->GetBuffer((Engine.BufferFrames / Engine.Channels), &BYTEBuffer);
	    HR_TO_RETURN(WASAPI->HR, "Couldn't get WASAPI buffer", NONE);

        memcpy(BYTEBuffer, polar_ringbuffer_ReadData(CallbackBuffer), ((sizeof (* polar_ringbuffer_ReadData(CallbackBuffer))) * (Engine.BufferFrames)));

        WASAPI->HR = WASAPI->AudioRenderClient->ReleaseBuffer((Engine.BufferFrames / Engine.Channels), 0);
        HR_TO_RETURN(WASAPI->HR, "Couldn't release WASAPI buffer", NONE);

        WASAPI->FramesWritten = (Engine.BufferFrames / Engine.Channels);

        polar_ringbuffer_ReadFinish(CallbackBuffer);
    }
}


LARGE_INTEGER win32_WallClock()
{    
    LARGE_INTEGER Result;
    QueryPerformanceCounter(&Result);
    return Result;
}

f32 win32_SecondsElapsed(LARGE_INTEGER Start, LARGE_INTEGER End)
{
    f32 Result = ((f32) (End.QuadPart - Start.QuadPart) / (f32) GlobalPerformanceCounterFrequency);
    return Result;
}

int main()
{
    //Allocate memory
    MEMORY_ARENA *EngineArena = memory_arena_Create(Kilobytes(500));
    MEMORY_ARENA *SourceArena = memory_arena_Create(Megabytes(100));

    //Start timings
    LARGE_INTEGER PerformanceCounterFrequencyResult;
    QueryPerformanceFrequency(&PerformanceCounterFrequencyResult);
    GlobalPerformanceCounterFrequency = PerformanceCounterFrequencyResult.QuadPart;

    //Request 1ms period for timing functions
    UINT SchedulerPeriodInMS = 1;
    bool IsSleepGranular = (timeBeginPeriod(SchedulerPeriodInMS) == TIMERR_NOERROR);

    //Define engine update rate
    f32 EngineUpdateRate = 60;
    f32 TargetSecondsPerFrame = 1.0f / (f32) EngineUpdateRate;

    //Fill out engine properties
    POLAR_ENGINE Engine = {};
    i32 FramesAvailable = 0;
    WASAPI_DATA *WASAPI = win32_WASAPI_Create(EngineArena, FramesAvailable);
    Engine.BufferFrames = FramesAvailable;
    Engine.Channels = WASAPI->DeviceWaveFormat->nChannels;
    Engine.SampleRate = WASAPI->DeviceWaveFormat->nSamplesPerSec;

    //Define the callback and channel summing functions
    Callback = &polar_render_Callback;
    switch(Engine.Channels)
    {
        case 2:
        {
            Summing = &polar_render_SumStereo;
            break;
        }
        default:
        {
            Summing = &polar_render_SumStereo;
            break;
        }
    }

    //PCG Random Setup
    i32 Rounds = 5;
    pcg32_srandom(time(NULL) ^ (intptr_t) &printf, (intptr_t) &Rounds);

    //Create ringbuffer with a specified block count (default is 3)
    POLAR_RINGBUFFER *CallbackBuffer = polar_ringbuffer_Create(EngineArena, Engine.BufferFrames, 3);

    //Create mixer object that holds all submixes and their containers
    POLAR_MIXER *MasterOutput = polar_mixer_Create(SourceArena, -1);
    
    //Sine sources
    polar_mixer_SubmixCreate(SourceArena, MasterOutput, 0, "SM_SineChordMix", -1);
    polar_mixer_ContainerCreate(MasterOutput, "SM_SineChordMix", "CO_ChordContainer", -1);
    polar_source_CreateFromFile(SourceArena, MasterOutput, Engine, "asset_lists/Source_Import.txt");

    //File sources
    polar_mixer_SubmixCreate(SourceArena, MasterOutput, 0, "SM_FileMix", -1);
    polar_mixer_ContainerCreate(MasterOutput, "SM_FileMix", "CO_FileContainer", -1);
    polar_source_Create(SourceArena, MasterOutput, Engine, "CO_FileContainer", "SO_WPN_Phasor", Stereo, SO_FILE, "audio/wpn_phasor.wav");
    polar_source_Create(SourceArena, MasterOutput, Engine, "CO_FileContainer", "SO_AMB_Forest_01", Stereo, SO_FILE, "audio/amb_river.wav");
    polar_source_Create(SourceArena, MasterOutput, Engine, "CO_FileContainer", "SO_Whiterun", Stereo, SO_FILE, "audio/Whiterun48.wav");

    //OSC setup
    UdpSocket OSCSocket = polar_OSC_StartServer(4795);

    //Silent first loop
    printf("Polar: Pre-roll silence\n");
    MasterOutput->Amplitude = DB(-99);
    for(u32 i = 0; i < 60; ++i)
    {
        win32_WASAPI_Callback(WASAPI, Engine, MasterOutput, CallbackBuffer);
    }

    //Start timings
    LARGE_INTEGER LastCounter = win32_WallClock();
    LARGE_INTEGER FlipWallClock = win32_WallClock();
    u64 LastCycleCount = __rdtsc();
    
    //Loop
    printf("Polar: Playback\n");
    MasterOutput->Amplitude = DB(-6);
    for(u32 i = 0; i < 2000; ++i)
    {
        //!liblo is a vanilla C OSC library that uses malloc - switch to that and change allocations for arena
        polar_OSC_UpdateMessages(OSCSocket, 1);

        polar_source_UpdatePlaying(MasterOutput);

        if(i == 100)
        {
            f32 StackPositions[MAX_CHANNELS] = {0.0};

            // polar_source_Play(MasterOutput, "SO_SineChord_Segment_A", 9, StackPositions, FX_DRY, EN_BREAKPOINT, "breakpoints/breaks2.txt");
            // polar_source_Play(MasterOutput, "SO_SineChord_Segment_B", 9, StackPositions, FX_DRY, EN_BREAKPOINT, "breakpoints/breaks2.txt");
            // polar_source_Play(MasterOutput, "SO_SineChord_Segment_C", 9, StackPositions, FX_DRY, EN_BREAKPOINT, "breakpoints/breaks2.txt");
            // polar_source_Play(MasterOutput, "SO_SineChord_Segment_D", 9, StackPositions, FX_DRY, EN_BREAKPOINT, "breakpoints/breaks2.txt");

            polar_source_Play(MasterOutput, "SO_Whiterun", 8, StackPositions, FX_DRY, EN_NONE, AMP(-1));
        }

        //Callback
        win32_WASAPI_Callback(WASAPI, Engine, MasterOutput, CallbackBuffer);
        printf("WASAPI: Frames written:\t%d\n", WASAPI->FramesWritten);

        //End performance timings
        FlipWallClock = win32_WallClock();
        u64 EndCycleCount = __rdtsc();
        LastCycleCount = EndCycleCount;

        //Check rendering work elapsed and sleep if time remaining
        LARGE_INTEGER WorkCounter = win32_WallClock();
        f32 WorkSecondsElapsed = win32_SecondsElapsed(LastCounter, WorkCounter);
        f32 SecondsElapsedForFrame = WorkSecondsElapsed;

        //If the rendering finished under the target seconds, then sleep until the next update
        if(SecondsElapsedForFrame < TargetSecondsPerFrame)
        {                        
            if(IsSleepGranular)
            {
                //!Sleep adds another 1ms delay, casuing WASAPI buffer size to misalign (from 480 to 528 samples)
                // DWORD SleepTimeInMS = (DWORD)(1000.0f * (TargetSecondsPerFrame - SecondsElapsedForFrame));
            
                // if(SleepTimeInMS > 0)
                // {
                //     Sleep(SleepTimeInMS);
                // }
            }
        }

        else
        {
            //!Missed frame rate!
            f32 Difference = (SecondsElapsedForFrame - TargetSecondsPerFrame);
            printf("Polar\tERROR: Missed frame rate!\tDifference: %f\t[Current: %f, Target: %f]\n", Difference, SecondsElapsedForFrame, TargetSecondsPerFrame);
        } 

        //Prepare timers before next loop
        LARGE_INTEGER EndCounter = win32_WallClock();
        LastCounter = EndCounter;
    }

    polar_mixer_Destroy(EngineArena, MasterOutput);

    polar_ringbuffer_Destroy(EngineArena, CallbackBuffer);
    win32_WASAPI_Destroy(EngineArena, WASAPI);

    memory_arena_Destroy(EngineArena);
    memory_arena_Destroy(SourceArena);

    return 0;
}