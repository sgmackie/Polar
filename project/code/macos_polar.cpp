
#include "polar.h"

#define DEFAULT_WIDTH 1280
#define DEFAULT_HEIGHT 720
#define DEFAULT_HZ 60

#define DEFAULT_SAMPLERATE 48000
#define DEFAULT_CHANNELS 2
#define DEFAULT_AMPLITUDE 0.8
#define DEFAULT_LATENCY_FRAMES 4

//Latency frames determines update rate - 4 @ 120HZ = 30FPS

#if MICROPROFILE
#define MICROPROFILE_MAX_FRAME_HISTORY (2<<10)
#include "../external/microprofile/microprofile.h"
#include "../external/microprofile/microprofile_html.h"
#include "../external/microprofile/microprofile.cpp"
#endif



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

#elif __linux__
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
#endif
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
        Info("macOS: File logger created succesfully");
    }
    else
    {
        printf("macOS: Failed to create logger!\n");
    }

    //Allocate memory arenas from virtual pages
    MEMORY_ARENA EngineArena = {};
    MEMORY_ARENA SourceArena = {};
    void *EngineBlock = VirtualAlloc(0, Kilobytes(500), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    void *SourceBlock = VirtualAlloc(0, Megabytes(100), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    EngineArena.Init(EngineBlock, Kilobytes(500));    
    SourceArena.Init(SourceBlock, Megabytes(100));
    Assert(EngineBlock && SourceBlock, "macOS: Failed to create memory arenas!");

    //Create memory pools for component memory
    MEMORY_POOL SourcePoolNames = {};
    MEMORY_POOL SourcePoolBuffers = {};
    MEMORY_POOL SourcePoolBreakpoints = {};
    SourcePoolNames.Init(SourceArena.Alloc(Megabytes(10), MEMORY_ARENA_ALIGNMENT), Megabytes(10), (sizeof(char) * MAX_STRING_LENGTH), MEMORY_POOL_ALIGNMENT);
    SourcePoolBuffers.Init(SourceArena.Alloc(Megabytes(10), MEMORY_ARENA_ALIGNMENT), Megabytes(10), (sizeof(f32) * MAX_BUFFER_SIZE), MEMORY_POOL_ALIGNMENT);
    SourcePoolBreakpoints.Init(SourceArena.Alloc(Megabytes(10), MEMORY_ARENA_ALIGNMENT), Megabytes(10), (sizeof(CMP_BREAKPOINT_POINT) * MAX_BREAKPOINTS), MEMORY_POOL_ALIGNMENT);
    Assert(SourcePoolNames.Data && SourcePoolBuffers.Data && SourcePoolBreakpoints.Data, "macOS: Failed to create source memory pools!");

    //Create memory pool for mixers
    MEMORY_POOL MixerPool = {};
    MEMORY_POOL MixerIntermediatePool = {};
    MixerPool.Init(SourceArena.Alloc(Megabytes(10), MEMORY_ARENA_ALIGNMENT), Megabytes(10), sizeof(SYS_MIX), MEMORY_POOL_ALIGNMENT);
    MixerIntermediatePool.Init(SourceArena.Alloc(Megabytes(10), MEMORY_ARENA_ALIGNMENT), Megabytes(10), (sizeof(f32) * MAX_BUFFER_SIZE), MEMORY_POOL_ALIGNMENT);
    Assert(MixerPool.Data && MixerIntermediatePool.Data, "macOS: Failed to create mixer memory pools!");

    if(EngineBlock)
    {
        //Define engine update rate
        POLAR_ENGINE Engine = {};
        Engine.UpdateRate = (MonitorRefresh / DEFAULT_LATENCY_FRAMES);
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

        //Buffer size:
        //The max buffer size is 1 second worth of samples
        //LatencySamples determines how many samples to render at a given frame delay (default is 2)
        //The sample count to write for each callback is the LatencySamples - any padding from the audio D3Device
        
        //Create ringbuffer with a specified block count (default is 3)
        Engine.CallbackBuffer.Create(&EngineArena, sizeof(i16), 4096, 3);
        
        //Create a temporary mixing buffer 
        Engine.MixBuffer.CreateFromArena(&EngineArena, sizeof(f32), Engine.BufferFrames);
        Assert(Engine.CallbackBuffer.Data && Engine.MixBuffer.Data, "macOS: Failed to create mix and callback buffers!");
        
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
        OscillatorModule.Noise.Create(&SourceArena, MAX_SOURCES);

        //Create mixer - a pool of mix systems
        POLAR_MIXER GlobalMixer = {};
        GlobalMixer.Mixes = (SYS_MIX **) SourceArena.Alloc((sizeof(SYS_MIX **) * 256), MEMORY_ARENA_ALIGNMENT);
        GlobalMixer.Mixes[GlobalMixer.Count] = (SYS_MIX *) MixerPool.Alloc();
        GlobalMixer.Mixes[GlobalMixer.Count]->Create(&SourceArena, MAX_SOURCES);
        ++GlobalMixer.Count;

        //Create entities
        ENTITY_SOURCES SoundSources = {};
        SoundSources.Create(&SourceArena, MAX_SOURCES);

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
            i32 SamplesToWrite = 0;
            i32 MaxSampleCount = 0;


            //Check the minimum update period for per-sample stepping states
            f64 MinPeriod = ((f64) SamplesToWrite / (f64) Engine.Format.SampleRate);

            //Get current time for update functions
            GlobalTime = core_WallTime();

            if(i == 25)
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

                //Add to breakpoint system
                SoundSources.Breakpoints[Index].CreateFromPool(&SourcePoolBreakpoints, MAX_BUFFER_SIZE);
                SoundSources.Flags[Index] |= ENTITY_SOURCES::BREAKPOINT;
                BreakpointSystem.Add(ID);
                BreakpointSystem.CreateFromFile(&SoundSources, ID, "data/testpoints.csv");

                //Add to fade system
                SoundSources.Amplitudes[Index].Init(0.1);
                SoundSources.Flags[Index] |= ENTITY_SOURCES::AMPLITUDE;
                FadeSystem.Add(ID);

                //Add to oscillator system
                SoundSources.Oscillators[Index].Init(CMP_OSCILLATOR::SINE, SoundSources.Playbacks[Index].Format.SampleRate, 261.63);
                SoundSources.Flags[Index] |= ENTITY_SOURCES::OSCILLATOR;
                OscillatorModule.Sine.Add(ID);

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

            //Update & Render
            //Write data
            if(Engine.CallbackBuffer.CanWrite())
            {
                //Update systems
                //Sample counts
                PlaySystem.Update(&SoundSources, GlobalTime, GlobalSamplesWritten, Engine.MixBuffer.Count);
                
                //Source types
                OscillatorModule.Sine.Update(&SoundSources, Engine.MixBuffer.Count);
                OscillatorModule.Square.Update(&SoundSources, Engine.MixBuffer.Count);
                OscillatorModule.Noise.Update(&SoundSources, Engine.MixBuffer.Count);
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

                //Update ringbuffer addresses
                Engine.CallbackBuffer.FinishRead();
            }






            //If the rendering finished under the target seconds, then sleep until the next update
            if(SecondsElapsedForFrame < TargetSecondsPerFrame)
            {                        
                if(IsSleepGranular)
                {
                    if(SleepTimeInMS > 0)
                    {
                        Sleep(SleepTimeInMS);
                    }
                }

                f32 TestSecondsElapsedForFrame = win32_SecondsElapsed(LastCounter, win32_WallClock());
                while(SecondsElapsedForFrame < TargetSecondsPerFrame)
                {                            
#if MICROPROFILE                    
                    MicroProfileTick();
#endif
                }
            }

            else
            {
                //!Missed frame rate!
                f32 Difference = (SecondsElapsedForFrame - TargetSecondsPerFrame);
                Fatal("macOS: Missed frame rate!\tDifference: %f\t[Current: %f, Target: %f]", Difference, SecondsElapsedForFrame, TargetSecondsPerFrame);
            }     
        }

        Engine.MixBuffer.Destroy();
        Engine.CallbackBuffer.Destroy();
        WASAPI.Destroy();
    }
    else
    {
        Fatal("macOS: Failed to create window!");
    }

    //Free pools
    SourcePoolBuffers.FreeAll();
    SourcePoolNames.FreeAll();
    SourcePoolBreakpoints.FreeAll();
    MixerPool.FreeAll();

    //Free arenas
    SourceArena.FreeAll();
    EngineArena.FreeAll();
    VirtualFree(SourceBlock, 0, MEM_RELEASE);
    VirtualFree(EngineBlock, 0, MEM_RELEASE);

#if MICROPROFILE
    MicroProfileStopAutoFlip();
    Info("Microprofiler: Results @ localhost:%d", MicroProfileWebServerPort());
    MicroProfileShutdown();
#endif

    //Destroy logging function - close file
    core_DestroyLogger();
}