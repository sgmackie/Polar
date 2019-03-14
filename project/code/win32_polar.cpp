#define MonitorRefreshHz 60
#define MAX_STRING_LENGTH 64
#define MAX_CHANNELS 4
#define MAX_CONTAINERS 4
#define MAX_SOURCES 32
#define MAX_BREAKPOINTS 1024
#define MAX_ENVELOPES 4

#define DEFAULT_SAMPLERATE 48000
#define DEFAULT_CHANNELS 2
#define DEFAULT_LATENCY_FRAMES 2
#define DEFAULT_AMPLITUDE 0.8

#include "polar.h"
#include "../external/external_code.h"
#include "win32_polar.h"

global_scope char AssetPath[MAX_STRING_LENGTH] = {"../../data/"};
global_scope i64 GlobalPerformanceCounterFrequency;

#include "polar.cpp"

WASAPI_DATA *win32_WASAPI_Create(MEMORY_ARENA *Arena, u32 SampleRate, u32 BufferSize)
{
    WASAPI_DATA *Result = 0;
    Result = (WASAPI_DATA *) memory_arena_Push(Arena, Result, (sizeof (WASAPI_DATA)));

    Result->HR = CoInitializeEx(0, COINIT_SPEED_OVER_MEMORY);
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

    WAVEFORMATEXTENSIBLE *MixFormat;
	Result->HR = Result->AudioClient->GetMixFormat((WAVEFORMATEX **) &MixFormat);
	HR_TO_RETURN(Result->HR, "Failed to activate audio endpoint", nullptr);

    //Create output format
    Result->DeviceWaveFormat = (WAVEFORMATEXTENSIBLE *) memory_arena_Push(Arena, Result->DeviceWaveFormat, (sizeof (Result->DeviceWaveFormat)));
    Result->DeviceWaveFormat->Format.cbSize = sizeof(WAVEFORMATEXTENSIBLE);
    Result->DeviceWaveFormat->Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE;
    Result->DeviceWaveFormat->Format.wBitsPerSample = 16;
    Result->DeviceWaveFormat->Format.nChannels = 2;
    Result->DeviceWaveFormat->Format.nSamplesPerSec = (DWORD) SampleRate;
    Result->DeviceWaveFormat->Format.nBlockAlign = (WORD) (Result->DeviceWaveFormat->Format.nChannels * Result->DeviceWaveFormat->Format.wBitsPerSample / 8);
    Result->DeviceWaveFormat->Format.nAvgBytesPerSec = Result->DeviceWaveFormat->Format.nSamplesPerSec * Result->DeviceWaveFormat->Format.nBlockAlign;
    Result->DeviceWaveFormat->Samples.wValidBitsPerSample = 16;
    Result->DeviceWaveFormat->dwChannelMask = KSAUDIO_SPEAKER_STEREO;
    Result->DeviceWaveFormat->SubFormat = KSDATAFORMAT_SUBTYPE_PCM;

    //If the current device sample rate doesn't equal the output, than set WASAPI to autoconvert
    DWORD Flags = 0;
    if(MixFormat->Format.nSamplesPerSec != Result->DeviceWaveFormat->Format.nSamplesPerSec)
    {
        printf("WASAPI: Sample rate does not equal the requested rate, resampling\t Result: %lu\t Requested: %lu\n", MixFormat->Format.nSamplesPerSec, Result->DeviceWaveFormat->Format.nSamplesPerSec);
        Flags = AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY;
    }

    //Free reference formati
    CoTaskMemFree(MixFormat);

    //Buffer size in 100 nano second units
    REFERENCE_TIME BufferDuration = 10000000ULL * BufferSize / Result->DeviceWaveFormat->Format.nSamplesPerSec;
	Result->HR = Result->AudioClient->Initialize(AUDCLNT_SHAREMODE_SHARED, Flags, BufferDuration, 0, &Result->DeviceWaveFormat->Format, NULL);
    HR_TO_RETURN(Result->HR, "Failed to initialise audio client", nullptr);

	Result->HR = Result->AudioClient->GetService(__uuidof(IAudioRenderClient), (void**) &Result->AudioRenderClient);
	HR_TO_RETURN(Result->HR, "Failed to assign client to render client", nullptr);

    Result->HR = Result->AudioClient->GetBufferSize(&Result->OutputBufferFrames);
	HR_TO_RETURN(Result->HR, "Failed to get maximum read buffer size for audio client", nullptr);

	Result->HR = Result->AudioClient->Reset();
	HR_TO_RETURN(Result->HR, "Failed to reset audio client before playback", nullptr);

	Result->HR = Result->AudioClient->Start();
	HR_TO_RETURN(Result->HR, "Failed to start audio client", nullptr);

    if(Result->OutputBufferFrames != BufferSize)
    {
        printf("WASAPI: WASAPI buffer size does not equal requested size!\t Result: %u\t Requested: %u\n", Result->OutputBufferFrames, BufferSize);
    }

    return Result;
}


void win32_WASAPI_Destroy(MEMORY_ARENA *Arena, WASAPI_DATA *WASAPI)
{
	WASAPI->AudioRenderClient->Release();
	WASAPI->AudioClient->Reset();
	WASAPI->AudioClient->Stop();
	WASAPI->AudioClient->Release();
	WASAPI->AudioDevice->Release();

	CoUninitialize();

    memory_arena_Reset(Arena);
    memory_arena_Pull(Arena);
}


void win32_WASAPI_Callback(WASAPI_DATA *WASAPI, u32 SampleCount, u32 Channels, i16 *OutputBuffer)
{
    BYTE* BYTEBuffer;
    
    if(SUCCEEDED(WASAPI->AudioRenderClient->GetBuffer((UINT32) SampleCount, &BYTEBuffer)))
    {
        //memcopy the output buffer * output channels into BYTEs for WASAPI to read
        size_t CopySize = ((sizeof(* OutputBuffer) * SampleCount) * Channels);
        memcpy(BYTEBuffer, OutputBuffer, CopySize);

        WASAPI->AudioRenderClient->ReleaseBuffer((UINT32) SampleCount, 0);
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
    MEMORY_ARENA *EngineArena = memory_arena_Create(Kilobytes(100));
    MEMORY_ARENA *SourceArena = memory_arena_Create(Megabytes(100));

    //Start timings
    LARGE_INTEGER PerformanceCounterFrequencyResult;
    QueryPerformanceFrequency(&PerformanceCounterFrequencyResult);
    GlobalPerformanceCounterFrequency = PerformanceCounterFrequencyResult.QuadPart;

    //Request 1ms period for timing functions
    UINT SchedulerPeriodInMS = 1;
    bool IsSleepGranular = (timeBeginPeriod(SchedulerPeriodInMS) == TIMERR_NOERROR);

    //Define engine update rate
    POLAR_ENGINE Engine = {};
    Engine.UpdateRate = MonitorRefreshHz / 2;
    f32 TargetSecondsPerFrame = 1.0f / (f32) Engine.UpdateRate;

    //PCG Random Setup
    i32 Rounds = 5;
    pcg32_srandom(time(NULL) ^ (intptr_t) &printf, (intptr_t) &Rounds);

    //Start WASAPI
    WASAPI_DATA *WASAPI = win32_WASAPI_Create(EngineArena, DEFAULT_SAMPLERATE, DEFAULT_SAMPLERATE);
    
    //Fill out engine properties
    Engine.NoiseFloor = AMP(-50);
    Engine.SampleRate = WASAPI->DeviceWaveFormat->Format.nSamplesPerSec;
    Engine.Channels = WASAPI->DeviceWaveFormat->Format.nChannels;
    Engine.BytesPerSample = sizeof(i16) * Engine.Channels;
    Engine.BufferSize = WASAPI->OutputBufferFrames;
    Engine.LatencySamples = DEFAULT_LATENCY_FRAMES * (Engine.SampleRate / Engine.UpdateRate);

    //Buffer size:
    //The max buffer size is 1 second worth of samples
    //LatencySamples determines how many samples to render at a given frame delay (default is 2)
    //The sample count to write for each callback is the LatencySamples - any padding from the audio device

    //Create ringbuffer with a specified block count (default is 3)
    POLAR_RINGBUFFER *CallbackBuffer = polar_ringbuffer_Create(EngineArena, Engine.BufferSize, DEFAULT_LATENCY_FRAMES);
    
    //Create a temporary mixing buffer 
    POLAR_BUFFER *MixBuffer = 0;
    MixBuffer = (POLAR_BUFFER *) memory_arena_Push(EngineArena, MixBuffer, (sizeof(POLAR_BUFFER)));
    MixBuffer->SampleCount = Engine.BufferSize;
    MixBuffer->Data = (f32 *) memory_arena_Push(EngineArena, MixBuffer, MixBuffer->SampleCount);

    if(WASAPI && CallbackBuffer && MixBuffer)
    {
        //OSC setup
        UdpSocket OSCSocket = polar_OSC_StartServer(4795);

        //Create mixer object that holds all submixes and their containers
        POLAR_MIXER *MasterOutput = polar_mixer_Create(SourceArena, -1);

        //Assign a listener to the mixer
        polar_listener_Create(MasterOutput, "LN_Player");

        //Sine sources
        polar_mixer_SubmixCreate(SourceArena, MasterOutput, 0, "SM_Trumpet", -1);
        polar_mixer_ContainerCreate(MasterOutput, "SM_Trumpet", "CO_Trumpet14", -1);
        polar_source_Create(SourceArena, MasterOutput, Engine, "CO_Trumpet14", "SO_Trumpet14_Partial_01", Mono, SO_OSCILLATOR, WV_SINE, 704.00);
        polar_source_Create(SourceArena, MasterOutput, Engine, "CO_Trumpet14", "SO_Trumpet14_Partial_02", Mono, SO_OSCILLATOR, WV_SINE, 704.00);
        polar_source_Create(SourceArena, MasterOutput, Engine, "CO_Trumpet14", "SO_Trumpet14_Partial_03", Mono, SO_OSCILLATOR, WV_SINE, 704.00);
        polar_source_Create(SourceArena, MasterOutput, Engine, "CO_Trumpet14", "SO_Trumpet14_Partial_04", Mono, SO_OSCILLATOR, WV_SINE, 704.00);
        polar_source_Create(SourceArena, MasterOutput, Engine, "CO_Trumpet14", "SO_Trumpet14_Partial_05", Mono, SO_OSCILLATOR, WV_SINE, 704.00);
        polar_source_Create(SourceArena, MasterOutput, Engine, "CO_Trumpet14", "SO_Trumpet14_Partial_06", Mono, SO_OSCILLATOR, WV_SINE, 704.00);
        polar_source_Create(SourceArena, MasterOutput, Engine, "CO_Trumpet14", "SO_Trumpet14_Partial_07", Mono, SO_OSCILLATOR, WV_SINE, 704.00);
        polar_source_Create(SourceArena, MasterOutput, Engine, "CO_Trumpet14", "SO_Trumpet14_Partial_08", Mono, SO_OSCILLATOR, WV_SINE, 704.00);
        polar_source_Create(SourceArena, MasterOutput, Engine, "CO_Trumpet14", "SO_Trumpet14_Partial_09", Mono, SO_OSCILLATOR, WV_SINE, 704.00);
        polar_source_Create(SourceArena, MasterOutput, Engine, "CO_Trumpet14", "SO_Trumpet14_Partial_10", Mono, SO_OSCILLATOR, WV_SINE, 704.00);
        polar_source_Create(SourceArena, MasterOutput, Engine, "CO_Trumpet14", "SO_Trumpet14_Partial_11", Mono, SO_OSCILLATOR, WV_SINE, 704.00);
        polar_source_Create(SourceArena, MasterOutput, Engine, "CO_Trumpet14", "SO_Trumpet14_Partial_12", Mono, SO_OSCILLATOR, WV_SINE, 704.00);
        polar_source_Create(SourceArena, MasterOutput, Engine, "CO_Trumpet14", "SO_Trumpet14_Partial_13", Mono, SO_OSCILLATOR, WV_SINE, 704.00);

        //File sources
        polar_mixer_SubmixCreate(SourceArena, MasterOutput, 0, "SM_FileMix", -1);
        polar_mixer_ContainerCreate(MasterOutput, "SM_FileMix", "CO_FileContainer", -1);
        polar_source_Create(SourceArena, MasterOutput, Engine, "CO_FileContainer", "SO_Whiterun", Stereo, SO_FILE, "audio/Whiterun48.wav");
        polar_source_Create(SourceArena, MasterOutput, Engine, "CO_FileContainer", "SO_Orbifold", Stereo, SO_FILE, "audio/LGOrbifold48.wav");

        //Start timings
        LARGE_INTEGER LastCounter = win32_WallClock();
        LARGE_INTEGER FlipWallClock = win32_WallClock();
        u64 LastCycleCount = __rdtsc();


        i64 i = 0;

        //Loop
        f64 GlobalTime = 0;
        bool GlobalRunning = true;
        MasterOutput->Amplitude = DB(-1);
        printf("Polar: Playback\n");
        while(GlobalRunning)
        {
            ++i;
            //Update
            //Get current time for update functions
            GlobalTime = polar_WallTime();
            
            //Get OSC messages from Unreal
            //!Uses std::vector for message allocation: replace with arena to be realtime safe
            polar_OSC_UpdateMessages(MasterOutput, GlobalTime, OSCSocket, 1);

            //Update the amplitudes, durations etc of all playing sources
            polar_source_UpdatePlaying(MasterOutput, GlobalTime, Engine.NoiseFloor);
            
            if(i == 10)
            {
                f32 StackPositions[MAX_CHANNELS] = {0.0};
                // polar_source_Play(MasterOutput, "SO_Orbifold", GlobalTime, 0, StackPositions, FX_DRY, EN_NONE, AMP(-12));

                // polar_source_Play(MasterOutput, "SO_Trumpet14_Partial_01", GlobalTime, 7, StackPositions, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial1.txt");
                // polar_source_Play(MasterOutput, "SO_Trumpet14_Partial_02", GlobalTime, 7, StackPositions, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial2.txt");
                // polar_source_Play(MasterOutput, "SO_Trumpet14_Partial_03", GlobalTime, 7, StackPositions, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial3.txt");
                // polar_source_Play(MasterOutput, "SO_Trumpet14_Partial_04", GlobalTime, 7, StackPositions, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial4.txt");
                // polar_source_Play(MasterOutput, "SO_Trumpet14_Partial_05", GlobalTime, 7, StackPositions, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial5.txt");
                // polar_source_Play(MasterOutput, "SO_Trumpet14_Partial_06", GlobalTime, 7, StackPositions, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial6.txt");
                // polar_source_Play(MasterOutput, "SO_Trumpet14_Partial_07", GlobalTime, 7, StackPositions, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial7.txt");
                // polar_source_Play(MasterOutput, "SO_Trumpet14_Partial_08", GlobalTime, 7, StackPositions, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial8.txt");
                // polar_source_Play(MasterOutput, "SO_Trumpet14_Partial_09", GlobalTime, 7, StackPositions, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial9.txt");
                // polar_source_Play(MasterOutput, "SO_Trumpet14_Partial_10", GlobalTime, 7, StackPositions, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial10.txt");
                // polar_source_Play(MasterOutput, "SO_Trumpet14_Partial_11", GlobalTime, 7, StackPositions, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial11.txt");
                // polar_source_Play(MasterOutput, "SO_Trumpet14_Partial_12", GlobalTime, 7, StackPositions, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial12.txt");
                // polar_source_Play(MasterOutput, "SO_Trumpet14_Partial_13", GlobalTime, 7, StackPositions, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial13.txt");
            }

            //Render
            //Write data
            if(polar_ringbuffer_WriteCheck(CallbackBuffer))
            {
                i32 SamplesToWrite = 0;
                i32 MaxSampleCount = 0;

                //Get current padding of the audio device and determine samples to write for this callback
                if(SUCCEEDED(WASAPI->AudioClient->GetCurrentPadding(&WASAPI->PaddingFrames)))
                {
                    MaxSampleCount = (i32) (Engine.BufferSize - WASAPI->PaddingFrames);
                    SamplesToWrite = (i32) (Engine.LatencySamples - WASAPI->PaddingFrames);
                    
                    if(SamplesToWrite < 0)
                    {
                        UINT32 DeviceSampleCount = 0;
                        if(SUCCEEDED(WASAPI->AudioClient->GetBufferSize(&DeviceSampleCount)))
                        {
                            SamplesToWrite = DeviceSampleCount;
                            printf("WASAPI: Failed to set SamplesToWrite!\n");
                        }
                    }

                    Assert(SamplesToWrite <= MaxSampleCount);
                    MixBuffer->SampleCount = SamplesToWrite;
                }

                //Render sources
                polar_render_Callback(&Engine, MasterOutput, MixBuffer, polar_ringbuffer_WriteData(CallbackBuffer));

                //Update ringbuffer addresses
                polar_ringbuffer_WriteFinish(CallbackBuffer);
            }

            //Read data
            if(polar_ringbuffer_ReadCheck(CallbackBuffer))
            {
                //Fill WASAPI BYTE buffer
                win32_WASAPI_Callback(WASAPI, MixBuffer->SampleCount, Engine.Channels, polar_ringbuffer_ReadData(CallbackBuffer));
                // printf("Polar: Samples written: %u\n", MixBuffer->SampleCount);

                //Update ringbuffer addresses
                polar_ringbuffer_ReadFinish(CallbackBuffer);
            }

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
                    DWORD SleepTimeInMS = (DWORD)(1000.0f * (TargetSecondsPerFrame - SecondsElapsedForFrame));

                    if(SleepTimeInMS > 0)
                    {
                        Sleep(SleepTimeInMS);
                        // printf("Sleep\n");
                    }
                }

                f32 TestSecondsElapsedForFrame = win32_SecondsElapsed(LastCounter, win32_WallClock());
                while(SecondsElapsedForFrame < TargetSecondsPerFrame)
                {                            
                    SecondsElapsedForFrame = win32_SecondsElapsed(LastCounter, win32_WallClock());
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
    }

    else
    {
    }

    polar_ringbuffer_Destroy(EngineArena, CallbackBuffer);
    win32_WASAPI_Destroy(EngineArena, WASAPI);

    memory_arena_Destroy(EngineArena);
    memory_arena_Destroy(SourceArena);

    return 0;
}