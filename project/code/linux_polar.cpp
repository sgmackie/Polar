#define DEFAULT_SAMPLERATE 48000
#define DEFAULT_CHANNELS 2
#define DEFAULT_LATENCY_FRAMES 2
#define DEFAULT_AMPLITUDE 0.8

#define MONITOR_HZ 120

#include "polar.h"
#include "../external/external_code.h"
#include "cuda/polar_cuda.h"
#include "linux_polar.h"

global_scope char AssetPath[MAX_STRING_LENGTH] = {"../../data/"};

#include "polar.cpp"

//ALSA setup
ALSA_DATA *linux_ALSA_Create(MEMORY_ARENA *Arena, u32 SampleRate, u32 BufferSize)
{
    ALSA_DATA *Result = 0;
    Result = (ALSA_DATA *) memory_arena_Push(Arena, Result, (sizeof (ALSA_DATA)));
    Result->SampleRate = SampleRate;
    Result->ALSAResample = 1;
    Result->Channels = 2;
    Result->Frames = BufferSize;

    //Error handling code passed to snd_strerror()
    Result->ALSAError = 0;

    Result->ALSAError = snd_pcm_open(&Result->Device, "default", SND_PCM_STREAM_PLAYBACK, SND_PCM_NONBLOCK | SND_PCM_ASYNC);   
    ERR_TO_RETURN(Result->ALSAError, "Failed to open default audio device", nullptr);

    Result->HardwareParameters = (snd_pcm_hw_params_t *) memory_arena_Push(Arena, Result->HardwareParameters, (sizeof (Result->HardwareParameters)));

    Result->ALSAError = snd_pcm_hw_params_any(Result->Device, Result->HardwareParameters);
    ERR_TO_RETURN(Result->ALSAError, "Failed to initialise hardware parameters", nullptr);

    Result->ALSAError = snd_pcm_hw_params_set_access(Result->Device, Result->HardwareParameters, SND_PCM_ACCESS_RW_INTERLEAVED);
    ERR_TO_RETURN(Result->ALSAError, "Failed to set PCM read and write access", nullptr);

    Result->ALSAError = snd_pcm_hw_params_set_format(Result->Device, Result->HardwareParameters, SND_PCM_FORMAT_S16);
    ERR_TO_RETURN(Result->ALSAError, "Failed to set PCM output format", nullptr);

    Result->ALSAError = snd_pcm_hw_params_set_rate(Result->Device, Result->HardwareParameters, Result->SampleRate, 0);
    ERR_TO_RETURN(Result->ALSAError, "Failed to set sample rate", nullptr);

    Result->ALSAError = snd_pcm_hw_params_set_rate_resample(Result->Device, Result->HardwareParameters, Result->ALSAResample);
    ERR_TO_RETURN(Result->ALSAError, "Failed to set resampling", nullptr);

    Result->ALSAError = snd_pcm_hw_params_set_channels(Result->Device, Result->HardwareParameters, Result->Channels);
    ERR_TO_RETURN(Result->ALSAError, "Failed to set channels", nullptr);

    Result->ALSAError = snd_pcm_hw_params(Result->Device, Result->HardwareParameters);
    ERR_TO_RETURN(Result->ALSAError, "Failed to set period", nullptr);

    Result->ALSAError = snd_pcm_prepare(Result->Device);
    ERR_TO_RETURN(Result->ALSAError, "Failed to start PCM device", nullptr);

    return Result;
}


//ALSA destroy
void linux_ALSA_Destroy(MEMORY_ARENA *Arena, ALSA_DATA *ALSA)
{
    snd_pcm_close(ALSA->Device);
    memory_arena_Reset(Arena);
    memory_arena_Pull(Arena);
}

void linux_ALSA_Callback(ALSA_DATA *ALSA, u32 SampleCount, u32 Channels, i16 *OutputBuffer)
{
    ALSA->FramesWritten = snd_pcm_writei(ALSA->Device, OutputBuffer, SampleCount);
    if(ALSA->FramesWritten < 0)
    {
        ALSA->FramesWritten = snd_pcm_recover(ALSA->Device, ALSA->FramesWritten, 0);
    }
    if(ALSA->FramesWritten < 0) 
    {
        // ERR_TO_RETURN(ALSA->FramesWritten, "ALSA: Failed to write any output frames! snd_pcm_writei()", NONE);
    }
    if(ALSA->FramesWritten > 0 && ALSA->FramesWritten < (SampleCount / Channels))
    {
        // printf("ALSA: Short write!\tExpected %i, wrote %li\n",  (SampleCount / Channels), ALSA->FramesWritten);
    }
}

timespec linux_WallClock()
{
    timespec Result;
    clock_gettime(CLOCK_MONOTONIC, &Result);
    
    return Result;
}

f32 linux_SecondsElapsed(timespec Start, timespec End)
{
    f32 Result = ((f32) (End.tv_sec - Start.tv_sec) + ((f32) (End.tv_nsec - Start.tv_nsec) * 1e-9f));
    return Result;
}



int main(int argc, char *argv[])
{
    //Allocate memory
    MEMORY_ARENA *EngineArena = memory_arena_Create(Kilobytes(100));
    MEMORY_ARENA *SourceArena = memory_arena_Create(Megabytes(100));

#if CUDA
    //Get CUDA Device
    CUDA_DEVICE GPU = {};
    cuda_DeviceGet(&GPU, 0);

    printf("%f\n", cuda_Sine(0.63787, 1));

#endif

    if(EngineArena && SourceArena)
    {

        //Define engine update rate
        POLAR_ENGINE Engine = {};
        Engine.UpdateRate = MONITOR_HZ / 2;
        f32 TargetSecondsPerFrame = 1.0f / (f32) Engine.UpdateRate;

        //PCG Random Setup
        i32 Rounds = 5;
        pcg32_srandom(time(NULL) ^ (intptr_t) &printf, (intptr_t) &Rounds);

        //Start ALSA
        ALSA_DATA *ALSA = linux_ALSA_Create(EngineArena, DEFAULT_SAMPLERATE, DEFAULT_SAMPLERATE);

        //Fill out engine properties
        Engine.NoiseFloor = AMP(-50);
        Engine.SampleRate = ALSA->SampleRate;
        Engine.Channels = ALSA->Channels;
        Engine.BytesPerSample = sizeof(i16) * Engine.Channels;
        Engine.BufferSize = ALSA->Frames;
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

        if(ALSA && CallbackBuffer && MixBuffer)
        {
            //OSC setup
            UdpSocket OSCSocket = polar_OSC_StartServer(4795);

            //Create mixer object that holds all submixes and their containers
            POLAR_MIXER *Master = polar_mixer_Create(SourceArena, -1);

            //Assign a listener to the mixer
            polar_listener_Create(Master, "LN_Player");

            //Sine sources
            polar_mixer_SubmixCreate(SourceArena, Master, 0, "SM_Trumpet", -1);
            polar_mixer_ContainerCreate(Master, "SM_Trumpet", "CO_Trumpet14", AMP(-10));
            polar_source_Create(SourceArena, Master, Engine, Hash("CO_Trumpet14"), Hash("SO_Trumpet14_Partial_01"), Mono, SO_OSCILLATOR, WV_SINE, 0);
            polar_source_Create(SourceArena, Master, Engine, Hash("CO_Trumpet14"), Hash("SO_Trumpet14_Partial_02"), Mono, SO_OSCILLATOR, WV_SINE, 0);
            polar_source_Create(SourceArena, Master, Engine, Hash("CO_Trumpet14"), Hash("SO_Trumpet14_Partial_03"), Mono, SO_OSCILLATOR, WV_SINE, 0);
            polar_source_Create(SourceArena, Master, Engine, Hash("CO_Trumpet14"), Hash("SO_Trumpet14_Partial_04"), Mono, SO_OSCILLATOR, WV_SINE, 0);
            polar_source_Create(SourceArena, Master, Engine, Hash("CO_Trumpet14"), Hash("SO_Trumpet14_Partial_05"), Mono, SO_OSCILLATOR, WV_SINE, 0);
            polar_source_Create(SourceArena, Master, Engine, Hash("CO_Trumpet14"), Hash("SO_Trumpet14_Partial_06"), Mono, SO_OSCILLATOR, WV_SINE, 0);
            polar_source_Create(SourceArena, Master, Engine, Hash("CO_Trumpet14"), Hash("SO_Trumpet14_Partial_07"), Mono, SO_OSCILLATOR, WV_SINE, 0);
            polar_source_Create(SourceArena, Master, Engine, Hash("CO_Trumpet14"), Hash("SO_Trumpet14_Partial_08"), Mono, SO_OSCILLATOR, WV_SINE, 0);
            polar_source_Create(SourceArena, Master, Engine, Hash("CO_Trumpet14"), Hash("SO_Trumpet14_Partial_09"), Mono, SO_OSCILLATOR, WV_SINE, 0);
            polar_source_Create(SourceArena, Master, Engine, Hash("CO_Trumpet14"), Hash("SO_Trumpet14_Partial_10"), Mono, SO_OSCILLATOR, WV_SINE, 0);
            polar_source_Create(SourceArena, Master, Engine, Hash("CO_Trumpet14"), Hash("SO_Trumpet14_Partial_11"), Mono, SO_OSCILLATOR, WV_SINE, 0);
            polar_source_Create(SourceArena, Master, Engine, Hash("CO_Trumpet14"), Hash("SO_Trumpet14_Partial_12"), Mono, SO_OSCILLATOR, WV_SINE, 0);
            polar_source_Create(SourceArena, Master, Engine, Hash("CO_Trumpet14"), Hash("SO_Trumpet14_Partial_13"), Mono, SO_OSCILLATOR, WV_SINE, 0);

            //File sources
            polar_mixer_SubmixCreate(SourceArena, Master, 0, "SM_FileMix", -1);
            polar_mixer_ContainerCreate(Master, "SM_FileMix", "CO_FileContainer", AMP(-1));
            polar_source_Create(SourceArena, Master, Engine, Hash("CO_FileContainer"), Hash("SO_Whiterun"), Stereo, SO_FILE, "audio/Whiterun48.wav");
            polar_source_Create(SourceArena, Master, Engine, Hash("CO_FileContainer"), Hash("SO_Orbifold"), Stereo, SO_FILE, "audio/LGOrbifold48.wav");

            //Start timings
            timespec LastCounter = linux_WallClock();
            timespec FlipWallClock = linux_WallClock();

            i64 i = 0;

            //Loop
            f64 GlobalTime = 0;
            bool GlobalRunning = true;
            printf("Polar: Playback\n");
            while(GlobalRunning)
            {
                ++i;

                //Updates
                //Calculate size of callback sample block
                i32 MaxSampleCount = (i32) (Engine.BufferSize);
                i32 SamplesToWrite = (i32) (Engine.LatencySamples);

                //Round the samples to write to the next power of 2
                MaxSampleCount = UpperPowerOf2(MaxSampleCount);
                SamplesToWrite = UpperPowerOf2(SamplesToWrite);

                if(SamplesToWrite < 0)
                {
                    snd_pcm_hw_params_get_buffer_size(ALSA->HardwareParameters, (u64 *) &SamplesToWrite);
                }

                Assert(SamplesToWrite <= MaxSampleCount);
                MixBuffer->SampleCount = SamplesToWrite;
                snd_pcm_hw_params_set_buffer_size(ALSA->Device, ALSA->HardwareParameters, SamplesToWrite);

                //Check the minimum update period for per-sample stepping states
                f64 MinPeriod = ((f64) SamplesToWrite / (f64) Engine.SampleRate);

                //Get current time for update functions
                GlobalTime = polar_WallTime();

                //Get OSC messages from Unreal
                //!Uses std::vector for message allocation: replace with arena to be realtime safe
                polar_OSC_UpdateMessages(Master, GlobalTime, OSCSocket, 1);

                //Update the amplitudes, durations etc of all playing sources
                polar_source_UpdatePlaying(Master, GlobalTime, MinPeriod, Engine.NoiseFloor);

                if(i == 10)
                {
                    f32 StackPositions[MAX_CHANNELS] = {0.0};
                    // polar_container_Play(Master, Hash("CO_FileContainer"), 0, StackPositions, FX_DRY, EN_NONE, AMP(-1));

                    polar_source_Play(Master, Hash("SO_Whiterun"), 0, StackPositions, FX_DRY, EN_NONE, AMP(-1));

                    // polar_source_Play(Master, Hash("SO_Trumpet14_Partial_01"), 1, StackPositions, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial1.txt");
                    // polar_source_Play(Master, Hash("SO_Trumpet14_Partial_02"), 1, StackPositions, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial2.txt");
                    // polar_source_Play(Master, Hash("SO_Trumpet14_Partial_03"), 1, StackPositions, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial3.txt");
                    // polar_source_Play(Master, Hash("SO_Trumpet14_Partial_04"), 1, StackPositions, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial4.txt");
                    // polar_source_Play(Master, Hash("SO_Trumpet14_Partial_05"), 1, StackPositions, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial5.txt");
                    // polar_source_Play(Master, Hash("SO_Trumpet14_Partial_06"), 1, StackPositions, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial6.txt");
                    // polar_source_Play(Master, Hash("SO_Trumpet14_Partial_07"), 1, StackPositions, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial7.txt");
                    // polar_source_Play(Master, Hash("SO_Trumpet14_Partial_08"), 1, StackPositions, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial8.txt");
                    // polar_source_Play(Master, Hash("SO_Trumpet14_Partial_09"), 1, StackPositions, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial9.txt");
                    // polar_source_Play(Master, Hash("SO_Trumpet14_Partial_10"), 1, StackPositions, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial10.txt");
                    // polar_source_Play(Master, Hash("SO_Trumpet14_Partial_11"), 1, StackPositions, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial11.txt");
                    // polar_source_Play(Master, Hash("SO_Trumpet14_Partial_12"), 1, StackPositions, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial12.txt");
                    // polar_source_Play(Master, Hash("SO_Trumpet14_Partial_13"), 1, StackPositions, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial13.txt");
                }


                //Render
                //Write data
                if(polar_ringbuffer_WriteCheck(CallbackBuffer))
                {
                    //Render sources
                    polar_render_Callback(&Engine, Master, MixBuffer, polar_ringbuffer_WriteData(CallbackBuffer));

                    //Update ringbuffer addresses
                    polar_ringbuffer_WriteFinish(CallbackBuffer);
                }

                //Read data
                if(polar_ringbuffer_ReadCheck(CallbackBuffer))
                {
                    //ALSA pcm_write call
                    linux_ALSA_Callback(ALSA, MixBuffer->SampleCount, Engine.Channels, polar_ringbuffer_ReadData(CallbackBuffer));

                    //Update ringbuffer addresses
                    polar_ringbuffer_ReadFinish(CallbackBuffer);
                }

                //End performance timings
                FlipWallClock = linux_WallClock();

                //Check rendering work elapsed and sleep if time remaining
                timespec WorkCounter = linux_WallClock();
                f32 WorkSecondsElapsed = linux_SecondsElapsed(LastCounter, WorkCounter);
                f32 SecondsElapsedForFrame = WorkSecondsElapsed;

                //If the rendering finished under the target seconds, then sleep until the next update
                if(SecondsElapsedForFrame < TargetSecondsPerFrame)
                {
                    f32 SleepTimeInMS = (1000.0f * (TargetSecondsPerFrame - SecondsElapsedForFrame));
                    u64 SleepTimeInNS = (u32) SleepTimeInMS * 1000000;
                    timespec SleepTimer = {};
                    SleepTimer.tv_nsec = SleepTimeInNS;

                    if(SleepTimeInMS > 0)
                    {
                        // printf("Polar: Sleeping for %fms\n", (f32) SleepTimeInMS);
                        nanosleep(&SleepTimer, 0);
                    }

                    f32 TestSecondsElapsedForFrame = linux_SecondsElapsed(LastCounter, linux_WallClock());
                    while(SecondsElapsedForFrame < TargetSecondsPerFrame)
                    {                            
                        SecondsElapsedForFrame = linux_SecondsElapsed(LastCounter, linux_WallClock());
                    }
                }

                else
                {
                    //!Missed frame rate!
                    f32 Difference = (SecondsElapsedForFrame - TargetSecondsPerFrame);
                    printf("Polar\tERROR: Missed frame rate!\tDifference: %f\t[Current: %f, Target: %f]\n", Difference, SecondsElapsedForFrame, TargetSecondsPerFrame);
                } 

                //Prepare timers before next loop
                timespec EndCounter = linux_WallClock();
                LastCounter = EndCounter;
            }
        }

        else
        {
        }

        polar_ringbuffer_Destroy(EngineArena, CallbackBuffer);
        linux_ALSA_Destroy(EngineArena, ALSA);
    }

    else
    {
    }

    memory_arena_Destroy(EngineArena);
    memory_arena_Destroy(SourceArena);

    return 0;
}