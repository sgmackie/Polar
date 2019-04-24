
#include "polar.h"

#define DEFAULT_WIDTH 1280
#define DEFAULT_HEIGHT 720
#define DEFAULT_HZ 120

#define DEFAULT_SAMPLERATE 48000
#define DEFAULT_CHANNELS 2
#define DEFAULT_AMPLITUDE 0.8
#define DEFAULT_LATENCY_FRAMES 2

//!On Windows can write to 4 latency frames @120Hz, Linux needs 2 or will underrun (short write)!
//Latency frames determines update rate - 4 @ 120HZ = 30FPS

#if MICROPROFILE
#define MICROPROFILE_MAX_FRAME_HISTORY (2<<10)
#include "../external/microprofile/microprofile.h"
#include "../external/microprofile/microprofile_html.h"
#include "../external/microprofile/microprofile.cpp"
#endif

#include <sys/mman.h>
#include <sys/time.h>


//Globals
static f64                      GlobalTime = 0;
static u32                      GlobalSamplesWritten = 0;
static bool                     GlobalRunning = false;
static i64                      GlobalPerformanceCounterFrequency = 0;
static bool                     GlobalUseCUDA = false;


f64 core_WallTime()
{
#ifdef _WIN32
    LARGE_INTEGER time,freq;
    if (!QueryPerformanceFrequency(&freq)){
        //  Handle error
        return 0;
    }
    if (!QueryPerformanceCounter(&time)){
        //  Handle error
        return 0;
    }
    return (double)time.QuadPart / freq.QuadPart;

#else
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
#endif
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

//Includes
#include <alsa/asoundlib.h>
#include <alsa/pcm.h>

//Macros
#define NONE    //Blank space for returning nothing in void functions

//ALSA Error code print and return
#define ERR_TO_RETURN(Result, Text, Type)				                    \
	if(Result < 0)								                            \
	{												                        \
		printf(Text "\t[%s]\n", snd_strerror(Result));   	                \
		return Type;								                        \
	}

typedef struct ALSA
{
    //Data
    i32 ALSAError;

    snd_pcm_t *Device;
    snd_pcm_hw_params_t *HardwareParameters;
    snd_pcm_sw_params_t *SoftwareParameters;
    snd_pcm_sframes_t FramesWritten;

    u32 SampleRate;
    u8 ALSAResample;
    u16 Channels;
    u32 LatencyInMS;
    u32 Frames;

    u32 Periods;
    snd_pcm_uframes_t PeriodSizeMin;
    snd_pcm_uframes_t PeriodSizeMax;
    snd_pcm_uframes_t BufferSizeMin;
    snd_pcm_uframes_t BufferSizeMax;

    //Functions
    void Init()
    {
        i32 ALSAError = 0;

        snd_pcm_t *Device = 0;
        snd_pcm_hw_params_t *HardwareParameters = 0;
        snd_pcm_sw_params_t *SoftwareParameters = 0;
        snd_pcm_sframes_t FramesWritten = 0;

        u32 SampleRate = 0;
        u8 ALSAResample = 0;
        u16 Channels = 0;
        u32 LatencyInMS = 0;
        u32 Frames = 0;

        u32 Periods = 0;
        snd_pcm_uframes_t PeriodSizeMin = 0;
        snd_pcm_uframes_t PeriodSizeMax = 0;
        snd_pcm_uframes_t BufferSizeMin = 0;
        snd_pcm_uframes_t BufferSizeMax = 0;
    }

    void Create(MEMORY_ARENA *Arena, u32 InputSampleRate, u32 InputChannelCount, u32 InputBitRate, size_t InputBufferSize)
    {
        Init();

        SampleRate = InputSampleRate;
        ALSAResample = 1;
        Channels = InputChannelCount;
        Frames = InputBufferSize;

        //Error handling code passed to snd_strerror()
        ALSAError = 0;

        ALSAError = snd_pcm_open(&Device, "default", SND_PCM_STREAM_PLAYBACK, SND_PCM_NONBLOCK | SND_PCM_ASYNC);   
        ERR_TO_RETURN(ALSAError, "Failed to open default audio device", NONE);

        //!LEAKING MEMORY
        snd_config_update_free_global();

        HardwareParameters = (snd_pcm_hw_params_t *) Arena->Alloc(sizeof(HardwareParameters), MEMORY_ARENA_ALIGNMENT);

        ALSAError = snd_pcm_hw_params_any(Device, HardwareParameters);
        ERR_TO_RETURN(ALSAError, "Failed to initialise hardware parameters", NONE);

        ALSAError = snd_pcm_hw_params_set_access(Device, HardwareParameters, SND_PCM_ACCESS_RW_INTERLEAVED);
        ERR_TO_RETURN(ALSAError, "Failed to set PCM read and write access", NONE);

        ALSAError = snd_pcm_hw_params_set_format(Device, HardwareParameters, SND_PCM_FORMAT_S16);
        ERR_TO_RETURN(ALSAError, "Failed to set PCM output format", NONE);

        ALSAError = snd_pcm_hw_params_set_rate(Device, HardwareParameters, SampleRate, 0);
        ERR_TO_RETURN(ALSAError, "Failed to set sample rate", NONE);

        ALSAError = snd_pcm_hw_params_set_rate_resample(Device, HardwareParameters, ALSAResample);
        ERR_TO_RETURN(ALSAError, "Failed to set resampling", NONE);

        ALSAError = snd_pcm_hw_params_set_channels(Device, HardwareParameters, Channels);
        ERR_TO_RETURN(ALSAError, "Failed to set channels", NONE);

        ALSAError = snd_pcm_hw_params(Device, HardwareParameters);
        ERR_TO_RETURN(ALSAError, "Failed to set period", NONE);

        ALSAError = snd_pcm_prepare(Device);
        ERR_TO_RETURN(ALSAError, "Failed to start PCM device", NONE);
    }

    //ALSA destroy
    void Destoy()
    {
        snd_pcm_close(Device);
        snd_config_update_free_global();
        Init();
    }


} ALSA;

void linux_ALSA_Callback(ALSA *ALSA, u32 SampleCount, u32 Channels, i16 *OutputBuffer)
{
    ALSA->FramesWritten = snd_pcm_writei(ALSA->Device, OutputBuffer, SampleCount);
    if(ALSA->FramesWritten < 0)
    {
        ALSA->FramesWritten = snd_pcm_recover(ALSA->Device, ALSA->FramesWritten, 0);
    }
    if(ALSA->FramesWritten < 0) 
    {
        ERR_TO_RETURN(ALSA->FramesWritten, "ALSA: Failed to write any output frames! snd_pcm_writei()", NONE);
    }
    if(ALSA->FramesWritten > 0 && ALSA->FramesWritten < (SampleCount / Channels))
    {
        printf("ALSA: Short write!\tExpected %i, wrote %li\n",  (SampleCount / Channels), ALSA->FramesWritten);
    }
}

int main()
{
    //Create logging function
#if LOGGER_ERROR    
    if(core_CreateLogger("logs.txt", LOG_ERROR, false))
#else
    if(core_CreateLogger("logs.txt", LOG_TRACE, false))
#endif
    {
        Info("linux: File logger created succesfully");
    }
    else
    {
        printf("linux: Failed to create logger!\n");
    }

    //Allocate memory arenas from virtual pages
    MEMORY_ARENA EngineArena = {};
    MEMORY_ARENA SourceArena = {};
    void *EngineBlock = mmap(0, Kilobytes(500), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    void *SourceBlock = mmap(0, Megabytes(100), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    EngineArena.Init(EngineBlock, Kilobytes(500));    
    SourceArena.Init(SourceBlock, Megabytes(100));
    Assert(EngineBlock && SourceBlock, "linux: Failed to create memory arenas!");

    //Create memory pools for component memory
    MEMORY_POOL SourcePoolNames = {};
    MEMORY_POOL SourcePoolBuffers = {};
    MEMORY_POOL SourcePoolBreakpoints = {};
    SourcePoolNames.Init(SourceArena.Alloc(Megabytes(10), MEMORY_ARENA_ALIGNMENT), Megabytes(10), (sizeof(char) * MAX_STRING_LENGTH), MEMORY_POOL_ALIGNMENT);
    SourcePoolBuffers.Init(SourceArena.Alloc(Megabytes(10), MEMORY_ARENA_ALIGNMENT), Megabytes(10), (sizeof(f32) * MAX_BUFFER_SIZE), MEMORY_POOL_ALIGNMENT);
    SourcePoolBreakpoints.Init(SourceArena.Alloc(Megabytes(10), MEMORY_ARENA_ALIGNMENT), Megabytes(10), (sizeof(CMP_BREAKPOINT_POINT) * MAX_BREAKPOINTS), MEMORY_POOL_ALIGNMENT);
    Assert(SourcePoolNames.Data && SourcePoolBuffers.Data && SourcePoolBreakpoints.Data, "linux: Failed to create source memory pools!");

    //Create memory pool for mixers
    MEMORY_POOL MixerPool = {};
    MEMORY_POOL MixerIntermediatePool = {};
    MixerPool.Init(SourceArena.Alloc(Megabytes(10), MEMORY_ARENA_ALIGNMENT), Megabytes(10), sizeof(SYS_MIX), MEMORY_POOL_ALIGNMENT);
    MixerIntermediatePool.Init(SourceArena.Alloc(Megabytes(10), MEMORY_ARENA_ALIGNMENT), Megabytes(10), (sizeof(f32) * MAX_BUFFER_SIZE), MEMORY_POOL_ALIGNMENT);
    Assert(MixerPool.Data && MixerIntermediatePool.Data, "linux: Failed to create mixer memory pools!");

    if(EngineBlock)
    {
        //Define engine update rate
        POLAR_ENGINE Engine = {};
        Engine.UpdateRate = (DEFAULT_HZ / DEFAULT_LATENCY_FRAMES);
        f32 TargetSecondsPerFrame = 1.0f / (f32) Engine.UpdateRate;        

#if MICROPROFILE
        MicroProfileOnThreadCreate("Main");
        MicroProfileSetEnableAllGroups(true);
        MicroProfileSetForceMetaCounters(true);
        MicroProfileStartAutoFlip(Engine.UpdateRate);
        Info("Microprofiler: Started profiler with autoflip");
#endif

        //PCG Random Setup
        i32 Rounds = 5;
        pcg32_srandom(time(NULL) ^ (intptr_t) &printf, (intptr_t) &Rounds);

        //Start ALSA
        ALSA ALSA = {};
        ALSA.Create(&EngineArena, DEFAULT_SAMPLERATE, 2, 16, DEFAULT_SAMPLERATE);

        //Fill out engine properties
        Engine.NoiseFloor           = DB(-120);
        Engine.Format.SampleRate    = ALSA.SampleRate;
        Engine.Format.Channels      = ALSA.Channels;
        Engine.BytesPerSample       = sizeof(i16) * Engine.Format.Channels;
        Engine.BufferFrames         = ALSA.Frames;
        Engine.LatencyFrames        = DEFAULT_LATENCY_FRAMES * (Engine.Format.SampleRate / Engine.UpdateRate);

        //Buffer size:
        //The max buffer size is 1 second worth of samples
        //LatencySamples determines how many samples to render at a given frame delay (default is 2)
        //The sample count to write for each callback is the LatencySamples - any padding from the audio D3Device
        
        //Create ringbuffer with a specified block count (default is 3)
        Engine.CallbackBuffer.Create(&EngineArena, sizeof(i16), 4096, 3);
        
        //Create a temporary mixing buffer 
        Engine.MixBuffer.CreateFromArena(&EngineArena, sizeof(f32), Engine.BufferFrames);
        Assert(Engine.CallbackBuffer.Data && Engine.MixBuffer.Data, "linux: Failed to create mix and callback buffers!");
        
        //Create systems
        SYS_FADE FadeSystem = {};
        SYS_ENVELOPE_BREAKPOINT BreakpointSystem = {};
        SYS_ENVELOPE_ADSR ADSRSystem = {};
        SYS_PLAY PlaySystem = {};
        SYS_WAV WavSystem   = {};
        FadeSystem.Create(&SourceArena, MAX_SOURCES);
        BreakpointSystem.Create(&SourceArena, MAX_SOURCES);
        ADSRSystem.Create(&SourceArena, MAX_SOURCES);
        PlaySystem.Create(&SourceArena, MAX_SOURCES);
        WavSystem.Create(&SourceArena, MAX_SOURCES);

        //Create oscillator module and subsystems
        MDL_OSCILLATOR OscillatorModule = {};
        OscillatorModule.Sine.Create(&SourceArena, MAX_SOURCES);
        OscillatorModule.Square.Create(&SourceArena, MAX_SOURCES);

        //Create noise module and subsystems
        MDL_NOISE NoiseModule = {};
        NoiseModule.White.Create(&SourceArena, MAX_SOURCES);
        NoiseModule.Brown.Create(&SourceArena, MAX_SOURCES);

        //Create mixer - a pool of mix systems
        POLAR_MIXER GlobalMixer = {};
        GlobalMixer.Mixes = (SYS_MIX **) SourceArena.Alloc((sizeof(SYS_MIX **) * 256), MEMORY_ARENA_ALIGNMENT);
        GlobalMixer.Mixes[GlobalMixer.Count] = (SYS_MIX *) MixerPool.Alloc();
        GlobalMixer.Mixes[GlobalMixer.Count]->Create(&SourceArena, MAX_SOURCES);
        ++GlobalMixer.Count;

        //Create entities
        ENTITY_SOURCES SoundSources = {};
        SoundSources.Create(&SourceArena, MAX_SOURCES);

        //Start timings
        timespec LastCounter = linux_WallClock();
        timespec FlipWallClock = linux_WallClock();

        //Loop
        i64 i = 0;
        GlobalTime = 0;
        GlobalRunning = true;
        Info("Polar: Playback\n");
        while(GlobalRunning)
        {
            //Updates
            ++i;

            //Process incoming mouse/keyboard messages, check for QUIT command
            if(!GlobalRunning) break;

            //Calculate size of callback sample block
            i32 MaxSampleCount = (i32) (Engine.BufferFrames);
            i32 SamplesToWrite = (i32) (Engine.LatencyFrames);

            //Round the samples to write to the next power of 2
            MaxSampleCount = UpperPowerOf2(MaxSampleCount);
            SamplesToWrite = UpperPowerOf2(SamplesToWrite);

            if(SamplesToWrite < 0)
            {
                snd_pcm_hw_params_get_buffer_size(ALSA.HardwareParameters, (u64 *) &SamplesToWrite);
            }

            // Assert(SamplesToWrite <= MaxSampleCount);
            Engine.MixBuffer.Count = SamplesToWrite;
            snd_pcm_hw_params_set_buffer_size(ALSA.Device, ALSA.HardwareParameters, SamplesToWrite);

            //Check the minimum update period for per-sample stepping states
            f64 MinPeriod = ((f64) SamplesToWrite / (f64) Engine.Format.SampleRate);

            //Get current time for update functions
            GlobalTime = core_WallTime();

            // if(i == 25)
            // {
            //     ID_SOURCE ID = SoundSources.AddByHash(FastHash("sine1"));
            //     size_t Index = SoundSources.RetrieveIndex(ID);

            //     //Add to playback system - set format and allocate buffer
            //     SoundSources.Playbacks[Index].Buffer.CreateFromPool(&SourcePoolBuffers, MAX_BUFFER_SIZE);
            //     SoundSources.Playbacks[Index].Format.Init(DEFAULT_SAMPLERATE, DEFAULT_CHANNELS);
            //     SoundSources.Flags[Index] |= ENTITY_SOURCES::PLAYBACK;
            //     PlaySystem.Add(ID);

            //     //Add to fade system
            //     SoundSources.Flags[Index] |= ENTITY_SOURCES::ADSR;
            //     ADSRSystem.Add(ID);
            //     ADSRSystem.Edit(&SoundSources, ID, SoundSources.Playbacks[Index].Format.SampleRate, 0.9, 4.0, 1.0, 0.7, 5.0);

            //     //!Fix file loading
            //     //Add to breakpoint system
            //     // SoundSources.Breakpoints[Index].CreateFromPool(&SourcePoolBreakpoints, MAX_BUFFER_SIZE);
            //     // SoundSources.Flags[Index] |= ENTITY_SOURCES::BREAKPOINT;
            //     // BreakpointSystem.Add(ID);
            //     // BreakpointSystem.CreateFromFile(&SoundSources, ID, "data/testpoints.csv");

            //     //Add to fade system
            //     SoundSources.Amplitudes[Index].Init(0.1);
            //     SoundSources.Flags[Index] |= ENTITY_SOURCES::AMPLITUDE;
            //     FadeSystem.Add(ID);

            //     //Add to oscillator system
            //     SoundSources.Oscillators[Index].Init(CMP_OSCILLATOR::SINE, SoundSources.Playbacks[Index].Format.SampleRate, 261.63);
            //     SoundSources.Flags[Index] |= ENTITY_SOURCES::OSCILLATOR;
            //     OscillatorModule.Sine.Add(ID);

            //     //Create modulator
            //     SoundSources.Modulators[Index].Init(CMP_MODULATOR::TYPE::LFO_OSCILLATOR, CMP_MODULATOR::ASSIGNMENT::FREQUENCY);
            //     SoundSources.Flags[Index] |= ENTITY_SOURCES::MODULATOR;
            //     if(SoundSources.Modulators[Index].Flag & CMP_MODULATOR::TYPE::LFO_OSCILLATOR)
            //     {
            //         SoundSources.Modulators[Index].Oscillator.Init(CMP_OSCILLATOR::SINE, SoundSources.Playbacks[Index].Format.SampleRate, 40);
            //     }

            //     //Play
            //     PlaySystem.Start(&SoundSources, ID, 10.0, 30);
            //     GlobalMixer.Mixes[0]->Add(ID);
            // }

            if(i == 10)
            {
                ID_SOURCE ID = SoundSources.AddByHash(FastHash("sine1"));
                size_t Index = SoundSources.RetrieveIndex(ID);

                //Add to playback system - set format and allocate buffer
                SoundSources.Playbacks[Index].Buffer.CreateFromPool(&SourcePoolBuffers, MAX_BUFFER_SIZE);
                SoundSources.Playbacks[Index].Format.Init(DEFAULT_SAMPLERATE, DEFAULT_CHANNELS);
                SoundSources.Flags[Index] |= ENTITY_SOURCES::PLAYBACK;
                PlaySystem.Add(ID);

                //Add to fade system
                SoundSources.Flags[Index] |= ENTITY_SOURCES::ADSR;
                ADSRSystem.Add(ID);
                ADSRSystem.Edit(&SoundSources, ID, SoundSources.Playbacks[Index].Format.SampleRate, 0.9, 4.0, 1.0, 0.7, 5.0);

                //Add to fade system
                SoundSources.Amplitudes[Index].Init(0.5);
                SoundSources.Flags[Index] |= ENTITY_SOURCES::AMPLITUDE;
                FadeSystem.Add(ID);

                //Add to noise system
                SoundSources.Noises[Index].Init(CMP_NOISE::BROWN);
                SoundSources.Flags[Index] |= ENTITY_SOURCES::NOISE;
                NoiseModule.Brown.Add(ID);

                //Create modulator
                SoundSources.Modulators[Index].Init(CMP_MODULATOR::TYPE::LFO_OSCILLATOR, CMP_MODULATOR::ASSIGNMENT::FREQUENCY);
                SoundSources.Flags[Index] |= ENTITY_SOURCES::MODULATOR;
                if(SoundSources.Modulators[Index].Flag & CMP_MODULATOR::TYPE::LFO_OSCILLATOR)
                {
                    SoundSources.Modulators[Index].Oscillator.Init(CMP_OSCILLATOR::SINE, SoundSources.Playbacks[Index].Format.SampleRate, 40);
                }

                //Play
                PlaySystem.Start(&SoundSources, ID, 10.0, 30);
                GlobalMixer.Mixes[0]->Add(ID);
            }

            if(i == 200)
            {
                GlobalRunning = false;
            }

            //Update & Render
            //Write data
            if(Engine.CallbackBuffer.CanWrite())
            {
                //Update systems
                //Sample counts
                PlaySystem.Update(&SoundSources, GlobalTime, GlobalSamplesWritten, Engine.MixBuffer.Count);
                
                //Source types
                //Oscillators
                OscillatorModule.Sine.Update(&SoundSources, Engine.MixBuffer.Count);
                OscillatorModule.Square.Update(&SoundSources, Engine.MixBuffer.Count);
                
                //Noise generators
                NoiseModule.White.Update(&SoundSources, Engine.MixBuffer.Count);
                NoiseModule.Brown.Update(&SoundSources, Engine.MixBuffer.Count);
                
                //Files
                WavSystem.Update(&SoundSources, 1.0, Engine.MixBuffer.Count);
                
                //Amplitudes
                BreakpointSystem.Update(&SoundSources, &FadeSystem, GlobalTime);
                ADSRSystem.Update(&SoundSources, Engine.MixBuffer.Count);
                FadeSystem.Update(&SoundSources, GlobalTime);
                
                //Clear mixer to 0
                f32 *MixBuffer = Engine.MixBuffer.Write();
                memset(MixBuffer, 0, (sizeof(f32) * Engine.MixBuffer.Count));

                //Loop through all active mixes
                for(size_t MixerIndex = 0; MixerIndex < GlobalMixer.Count; ++MixerIndex)
                {
                    //Allocate a temporary buffer from the pool
                    f32 *IntermediateBuffer = (f32 *) MixerIntermediatePool.Alloc();
                    memset(IntermediateBuffer, 0, (sizeof(f32) * Engine.MixBuffer.Count));

                    //Render all sources in a mix the temporary buffer
                    GlobalMixer.Mixes[MixerIndex]->Update(&SoundSources, IntermediateBuffer, Engine.MixBuffer.Count);

                    //Add temporary pool to global mixbuffer
                    for(size_t i = 0; i < Engine.MixBuffer.Count; ++i)
                    {
                        MixBuffer[i] += IntermediateBuffer[i];
                    }

                    //Free temporary buffer
                    MixerIntermediatePool.Free(IntermediateBuffer);
                }

                //Callback - convert f32 global mixing buffer to i16 output for WASAPI
                GlobalSamplesWritten = polar_render_Callback(&Engine);

                //Update ringbuffer addresses
                Engine.CallbackBuffer.FinishWrite();
            }

            //Read data
            if(Engine.CallbackBuffer.CanRead())
            {
                //Fill WASAPI BYTE buffer
                linux_ALSA_Callback(&ALSA, Engine.MixBuffer.Count, Engine.Format.Channels, Engine.CallbackBuffer.Read());

                //Update ringbuffer addresses
                Engine.CallbackBuffer.FinishRead();
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
                    nanosleep(&SleepTimer, 0);
                }

                f32 TestSecondsElapsedForFrame = linux_SecondsElapsed(LastCounter, linux_WallClock());
                while(SecondsElapsedForFrame < TargetSecondsPerFrame)
                {                            
                    SecondsElapsedForFrame = linux_SecondsElapsed(LastCounter, linux_WallClock());
#if MICROPROFILE                    
                    MicroProfileTick();
#endif
                }
            }
            else
            {
                //!Missed frame rate!
                f32 Difference = (SecondsElapsedForFrame - TargetSecondsPerFrame);
                Fatal("win32: Missed frame rate!\tDifference: %f\t[Current: %f, Target: %f]", Difference, SecondsElapsedForFrame, TargetSecondsPerFrame);
            } 

            //Prepare timers before next loop
            timespec EndCounter = linux_WallClock();
            LastCounter = EndCounter; 
        }

        Engine.MixBuffer.Destroy();
        Engine.CallbackBuffer.Destroy();
    }
    else
    {
        Fatal("linux: Failed to create window!");
    }

    //Free pools
    SourcePoolBuffers.FreeAll();
    SourcePoolNames.FreeAll();
    SourcePoolBreakpoints.FreeAll();
    MixerPool.FreeAll();

    //Free arenas
    SourceArena.FreeAll();
    EngineArena.FreeAll();
    //!Page fault errors here?
    munmap(EngineBlock, Kilobytes(500));
    munmap(SourceBlock, Megabytes(100));

#if MICROPROFILE
    MicroProfileStopAutoFlip();
    Info("Microprofiler: Results @ localhost:%d", MicroProfileWebServerPort());
    MicroProfileShutdown();
#endif

    //Destroy logging function - close file
    core_DestroyLogger();
}