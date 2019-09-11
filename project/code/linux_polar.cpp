
// sudo /usr/local/cuda/bin/nvprof ./linux_polar

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

//IMGUI implementation - OpenGL
#include <GL/glew.h>    // Initialize with glewInit()
#include "../external/imgui/linux/imgui_impl_glfw.cpp"
#include "../external/imgui/linux/imgui_impl_opengl3.cpp"
#include <GLFW/glfw3.h>

//ALSA
#include <alsa/asoundlib.h>
#include <alsa/pcm.h>

//Globals
static f64                      GlobalTime = 0;
static u32                      GlobalSamplesWritten = 0;
static bool                     GlobalRunning = false;
static i64                      GlobalPerformanceCounterFrequency = 0;
static bool                     GlobalUseCUDA = false;
static LISTENER                 GlobalListener = {};
static bool                     GlobalFireEvent = false;
static u32                      GlobalEventCount = 0;

static bool                     GlobalEditEvent = false;
static ID_VOICE                 GlobalBubbleVoice = 0;
static i32                      GlobalBubbleCount = 4;
static f32                      GlobalBubblesPerSec = 100;
static f32                      GlobalRadius = 10;
static f32                      GlobalAmplitude = 0.9;
static f32                      GlobalProbability = 1.0;
static f32                      GlobalRiseCutoff = 0.5;

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

    void Create(MEMORY_ALLOCATOR *Allocator, u32 InputSampleRate, u32 InputChannelCount, u32 InputBitRate, size_t InputBufferSize)
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

        HardwareParameters = (snd_pcm_hw_params_t *) Allocator->Alloc(sizeof(HardwareParameters), HEAP_TAG_PLATFORM);

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

static void glfw_error_callback(int error, const char* description)
{
    Fatal("Glfw Error %d: %s\n", error, description);
}

int main(int argc, char** argv)
{
    //Create logging function
#if LOGGER_PROFILE
    if(core_CreateLogger("logs.txt", LOG_FATAL, false))
#elif LOGGER_ERROR    
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

#if CUDA
    //Get CUDA Device
    i32 DeviceCount = 0;
    GlobalUseCUDA = (cudaGetDeviceCount(&DeviceCount) == cudaSuccess && DeviceCount != 0);
    
    if(GlobalUseCUDA)
    {
        CUDA_DEVICE GPU = {};
        cuda_DeviceGet(&GPU, 0);
        // cuda_DevicePrint(&GPU);
    }
#endif

    // Setup window
    glfwSetErrorCallback(glfw_error_callback);
    if(!glfwInit())
    {
        return 1;
    }
    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only    

    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(DEFAULT_WIDTH, DEFAULT_HEIGHT, "Polar", NULL, NULL);
    if(window == 0)
    {
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Initialize OpenGL loader
    bool err = glewInit() != GLEW_OK;
    if(err)
    {
        return 1;
    }

    // Create allocators
    // Platform allocator for getting the fixed blocks
    MEMORY_ALLOCATOR LinuxAllocator = {};
    LinuxAllocator.Create(MEMORY_ALLOCATOR_VIRTUAL, "Linux Virtual Allocator", 0, Gigabytes(1));
    void *EngineBlock = LinuxAllocator.Alloc(Megabytes(100));
    void *SourceBlock = LinuxAllocator.Alloc(Megabytes(900));

    // General heap allocator used for systems
    MEMORY_ALLOCATOR EngineHeap = {};
    EngineHeap.Create(MEMORY_ALLOCATOR_HEAP, "Engine Heap Allocator", MEMORY_TAGGED_HEAP_FIXED_SIZE, Megabytes(2), EngineBlock, Megabytes(100));

    // Allocate fixed arena for source memory (pools, large data allocations)
    MEMORY_ARENA SourceArena = {};
    SourceArena.CreateFixedSize("Source Fixed Arena", MEMORY_ARENA_FIXED_SIZE, SourceBlock, Megabytes(900));

    // Create debug profiler info
#if CORE_PROFILE
    // Expanding debug arena
    MEMORY_ALLOCATOR DebugArena = {};
    DebugArena.Create(MEMORY_ALLOCATOR_ARENA, "Debug Expanding Arena", MEMORY_ARENA_NORMAL);

    // Allocate
    debug_state *GlobalDebugState = 0;
    GlobalDebugTable = 0;
    GlobalDebugTable = (debug_table *) DebugArena.Alloc(sizeof(debug_table));
    GlobalDebugState = (debug_state *) DebugArena.Alloc(sizeof(debug_state));
	DEBUGSetEventRecording(true);
    DEBUGInit(GlobalDebugState);
#endif

    //Create memory pools for voice memory
    POLAR_POOL Pools = {};
    Pools.Names.CreateFixedSize("Source Names Pool", MEMORY_POOL_FIXED_SIZE, (sizeof(char) * MAX_STRING_LENGTH), SourceArena.Push(Megabytes(10)), Megabytes(10));
    Pools.Buffers.CreateFixedSize("Source Buffer Pool", MEMORY_POOL_FIXED_SIZE, (sizeof(f32) * MAX_BUFFER_SIZE), SourceArena.Push(Megabytes(10)), Megabytes(10));
    Pools.Partials.CreateFixedSize("Source Partials Pool", MEMORY_POOL_FIXED_SIZE, (sizeof(CMP_CUDA_PHASOR) * MAX_PARTIALS), SourceArena.Push(Megabytes(10)), Megabytes(10));
    Pools.Phases.CreateFixedSize("Source Phases Pool", MEMORY_POOL_FIXED_SIZE, (sizeof(f32) * MAX_BUFFER_SIZE), SourceArena.Push(Megabytes(10)), Megabytes(10));
    Pools.Breakpoints.CreateFixedSize("Source Breakpoints Pool", MEMORY_POOL_FIXED_SIZE, (sizeof(CMP_BREAKPOINT_POINT) * MAX_BREAKPOINTS), SourceArena.Push(Megabytes(10)), Megabytes(10));
    Pools.WAVs.CreateFixedSize("Source WAVs Pool", MEMORY_POOL_FIXED_SIZE, (sizeof(f32) * MAX_WAV_SIZE), SourceArena.Push(Megabytes(10)), Megabytes(10));

    Pools.Bubble.Generators.CreateFixedSize("Source Bubble Generators Pool", MEMORY_POOL_FIXED_SIZE, (sizeof(CMP_BUBBLES_MODEL) * MAX_BUBBLE_COUNT), SourceArena.Push(Megabytes(10)), Megabytes(10));
    Pools.Bubble.Radii.CreateFixedSize("Source Bubble Radii Pool", MEMORY_POOL_FIXED_SIZE, (sizeof(f64) * MAX_BUBBLE_COUNT), SourceArena.Push(Megabytes(10)), Megabytes(10));
    Pools.Bubble.Lambda.CreateFixedSize("Source Bubble Lambda Pool", MEMORY_POOL_FIXED_SIZE, (sizeof(f64) * MAX_BUBBLE_COUNT), SourceArena.Push(Megabytes(10)), Megabytes(10));

    if(EngineBlock)
    {
        //Define engine update rate
        POLAR_ENGINE Engine = {};
        Engine.Init((u32) core_WallTime(), (DEFAULT_HZ / DEFAULT_LATENCY_FRAMES));  

        //Create GUI context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;
		io.DisplaySize.x = (f32) DEFAULT_WIDTH;
		io.DisplaySize.y = (f32) DEFAULT_HEIGHT;

        //Set GUI style
        ImGui::StyleColorsDark();

        //Set GUI state
        ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

        // GUI bindings
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init(glsl_version);

        //Start ALSA
        ALSA ALSA = {};
        ALSA.Create(&EngineHeap, DEFAULT_SAMPLERATE, 2, 16, DEFAULT_SAMPLERATE);

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
        Engine.CallbackBuffer.Create(&EngineHeap, sizeof(i16), 4096, 3);
        
        //OSC setup
        //!Replace with vanilla C version
        // UdpSocket OSCSocket = polar_OSC_StartServer(4795);

        //Create systems
        Engine.VoiceSystem.Create(&EngineHeap, MAX_SOURCES);
        Engine.Systems.Play.Create(&EngineHeap, MAX_SOURCES);
        Engine.Systems.Position.Create(&EngineHeap, MAX_SOURCES);
        Engine.Systems.Mix.Create(&EngineHeap, MAX_SOURCES);
        Engine.Systems.Fade.Create(&EngineHeap, MAX_SOURCES);
        Engine.Systems.Crossfade.Create(&EngineHeap, MAX_SOURCES);
        Engine.Systems.Breakpoint.Create(&EngineHeap, MAX_SOURCES);
        Engine.Systems.ADSR.Create(&EngineHeap, MAX_SOURCES);
        Engine.Systems.Filter.Create(&EngineHeap, MAX_SOURCES);

        Engine.Systems.Oscillator.Sine.Create(&EngineHeap, MAX_SOURCES);
        Engine.Systems.Oscillator.Square.Create(&EngineHeap, MAX_SOURCES);
        Engine.Systems.Oscillator.Triangle.Create(&EngineHeap, MAX_SOURCES);
        Engine.Systems.Oscillator.Sawtooth.Create(&EngineHeap, MAX_SOURCES);
        Engine.Systems.Noise.White.Create(&EngineHeap, MAX_SOURCES);
        Engine.Systems.Noise.Brown.Create(&EngineHeap, MAX_SOURCES);

        Engine.Systems.WAV.Create(&EngineHeap, MAX_SOURCES);

        Engine.Systems.Cuda.Create(&EngineHeap, MAX_SOURCES);
        Engine.Systems.Bubbles.Create(&EngineHeap, MAX_SOURCES);

        Engine.Systems.FFT.Create(&EngineHeap, MAX_SOURCES);
        Engine.Systems.Grain.Create(&EngineHeap, MAX_SOURCES);

        //Create entities
        ENTITY_SOURCES SoundSources = {};
        SoundSources.Create(&EngineHeap, MAX_SOURCES);

        //Create voices
        ENTITY_VOICES SoundVoices = {};
        SoundVoices.Create(&EngineHeap, MAX_VOICES);

        //Start timings
        timespec LastCounter = linux_WallClock();
        timespec FlipWallClock = linux_WallClock();

        u32 ExpectedFramesPerUpdate = 1;
        f32 TargetSecondsPerFrame = ExpectedFramesPerUpdate / (f32) Engine.UpdateRate;        

        //!Had to make these permanent allocations for now as memset after a sleep causes memory overwrites (SystemVoiceCount reset to 0, effecting other structs)
        f32 *MixerChannel0 = (f32 *) EngineHeap.Alloc((sizeof(f32) * MAX_BUFFER_SIZE), HEAP_TAG_MIXER);
        f32 *MixerChannel1 = (f32 *) EngineHeap.Alloc((sizeof(f32) * MAX_BUFFER_SIZE), HEAP_TAG_MIXER);

        f32 *SineTemp = (f32 *) EngineHeap.Alloc((sizeof(f32) * MAX_BUFFER_SIZE), HEAP_TAG_MIXER);
        f32 *PulseTemp = (f32 *) EngineHeap.Alloc((sizeof(f32) * MAX_BUFFER_SIZE), HEAP_TAG_MIXER);

        //Loop
        GlobalTime = 0;
        GlobalRunning = true;
        Info("Polar: Playback\n");
        while(GlobalRunning && !glfwWindowShouldClose(window))
        {
            glfwPollEvents();
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
            snd_pcm_hw_params_set_buffer_size(ALSA.Device, ALSA.HardwareParameters, SamplesToWrite);

            //Check the minimum update period for per-sample stepping states
            f64 MinPeriod = ((f64) SamplesToWrite / (f64) Engine.Format.SampleRate);

            //Get current time for update functions
            GlobalTime = core_WallTime();

            if(GlobalFireEvent == true)
            {
                GlobalFireEvent = false;
                GlobalEditEvent = true;
                //Create source
                HANDLE_SOURCE Source = SoundSources.AddByHash(FastHash("SO_Bubbles"));

                //Add to playback system - set format then add to the voice system to spawn future voices 
                SoundSources.Formats[Source.Index].Init(DEFAULT_SAMPLERATE, DEFAULT_CHANNELS);
                SoundSources.Flags[Source.Index] |= ENTITY_SOURCES::PLAYBACK;
                Engine.VoiceSystem.Add(Source.ID);

                //Add to distance system
                SoundSources.Distances[Source.Index].Init();
                SoundSources.Flags[Source.Index] |= ENTITY_SOURCES::POSITION;
                SoundSources.Positions[Source.Index].Init();

                //Add to fade system
                SoundSources.Flags[Source.Index] |= ENTITY_SOURCES::AMPLITUDE;
                SoundSources.Amplitudes[Source.Index].Init(0.9);

                //Add to pan system
                SoundSources.Flags[Source.Index] |= ENTITY_SOURCES::PAN;
                SoundSources.Pans[Source.Index].Init(CMP_PAN::MODE::STEREO, 0.0);           

                // SoundSources.Flags[Source.Index] |= ENTITY_SOURCES::CUDA_SINE;
                SoundSources.Flags[Source.Index] |= ENTITY_SOURCES::CUDA_BUBBLE;
                SoundSources.Bubbles[Source.Index].Init(SoundSources.Formats[Source.Index].SampleRate, GlobalBubbleCount, GlobalBubblesPerSec, GlobalRadius, GlobalAmplitude, GlobalProbability, GlobalRiseCutoff);

                //Play
                GlobalBubbleVoice = Engine.VoiceSystem.Spawn(&SoundSources, &SoundVoices, Source.ID, Engine.Format.SampleRate, RandomU32(&Engine.RNG), false, &Engine.Systems, &Pools);
                Engine.Systems.Bubbles.ComputeBubbles(&SoundVoices, GlobalBubbleVoice, Engine.Format.SampleRate, SamplesToWrite, &Engine.RNG);
                Engine.Systems.Bubbles.ComputeEvents(&SoundVoices, GlobalBubbleVoice, Engine.Format.SampleRate);

                Engine.Systems.Play.Start(&SoundVoices, GlobalBubbleVoice, 5.0, -1);
            }

            if(GlobalEditEvent)
            {             
                Engine.Systems.Bubbles.Edit(&SoundVoices, GlobalBubbleVoice, Engine.Format.SampleRate, SamplesToWrite, &Engine.RNG, GlobalBubbleCount, GlobalBubblesPerSec, GlobalRadius, GlobalAmplitude, GlobalProbability, GlobalRiseCutoff);
            }

            //Update & Render
            //Write data
            if(Engine.CallbackBuffer.CanWrite())
            {
                //Update systems
                BEGIN_BLOCK("Systems Update");
                //Check for voices that need to be spawned as a result of the play system update, add any to the mixer system
                Engine.VoiceSystem.Update(&SoundSources, &SoundVoices, &Engine.Systems, &Pools);

                //Sample counts
                Engine.Systems.Play.Update(&SoundVoices, GlobalTime, GlobalSamplesWritten, SamplesToWrite);
                
                //Source types
                //Oscillators
                Engine.Systems.Oscillator.Sine.Update(&SoundVoices, SamplesToWrite);
                Engine.Systems.Oscillator.Square.Update(&SoundVoices, SamplesToWrite);
                Engine.Systems.Oscillator.Triangle.Update(&SoundVoices, SamplesToWrite);
                Engine.Systems.Oscillator.Sawtooth.Update(&SoundVoices, SamplesToWrite);
                
                //Noise generators
                Engine.Systems.Noise.White.Update(&SoundVoices, &Engine.RNG, SamplesToWrite);
                Engine.Systems.Noise.Brown.Update(&SoundVoices, &Engine.RNG, SamplesToWrite);           
                

                Engine.Systems.Cuda.Update(&SoundVoices, SineTemp, SamplesToWrite);
                Engine.Systems.Bubbles.Update(&SoundVoices, Engine.Format.SampleRate, &Engine.RNG, PulseTemp, SamplesToWrite);

                //Files
                Engine.Systems.WAV.Update(&SoundVoices, 1.0, SamplesToWrite);

                // Engine.Systems.Grain.Update(&SoundVoices, SamplesToWrite, GrainParameter);

                //Filters
                Engine.Systems.Filter.Update(&SoundVoices, SamplesToWrite);

                //Amplitudes
                Engine.Systems.Breakpoint.Update(&SoundVoices, &Engine.Systems.Fade, GlobalTime);
                Engine.Systems.ADSR.Update(&SoundVoices, SamplesToWrite);
                
                // Engine.Systems.Parameter.Update(&SoundVoices, GlobalTime);

                //World positions
                Engine.Systems.Position.Update(&SoundVoices, &GlobalListener, Engine.NoiseFloor);


                Engine.Systems.Crossfade.Update(&SoundVoices, SamplesToWrite);
                Engine.Systems.Fade.Update(&SoundVoices, GlobalTime);

                END_BLOCK();
                BEGIN_BLOCK("Mixing");

                //Clear mixer channels to 0
                memset(MixerChannel0, 0, (sizeof(f32) * SamplesToWrite));
                memset(MixerChannel1, 0, (sizeof(f32) * SamplesToWrite));

                //Render all sources in a mix the temporary buffer
                GlobalSamplesWritten = Engine.Systems.Mix.Update(&SoundSources, &SoundVoices, MixerChannel0, MixerChannel1, SamplesToWrite);

                //Int16 conversion
                //Copy over mixer channels
                f32 *FloatChannel0 = MixerChannel0;
                f32 *FloatChannel1 = MixerChannel1;

                //Get callback buffer
                int16 *ConvertedSamples = Engine.CallbackBuffer.Write();
                memset(ConvertedSamples, 0, (sizeof(i16) * SamplesToWrite));

                for(size_t SampleIndex = 0; SampleIndex < SamplesToWrite; ++SampleIndex)
                {
                    //Channel 1
                    f32 FloatSample     = FloatChannel0[SampleIndex];
                    i16 IntSample       = FloatToInt16(FloatSample);     
                    *ConvertedSamples++ = IntSample;

                    //Channel 2
                    FloatSample         = FloatChannel1[SampleIndex];
                    IntSample           = FloatToInt16(FloatSample);     
                    *ConvertedSamples++ = IntSample;        
                }

                //Update ringbuffer addresses
                Engine.CallbackBuffer.FinishWrite();

                END_BLOCK();
            }

            //Read data
            BEGIN_BLOCK("OS Device Callback");
            if(Engine.CallbackBuffer.CanRead())
            {
                //Fill WASAPI BYTE buffer
                linux_ALSA_Callback(&ALSA, SamplesToWrite, Engine.Format.Channels, Engine.CallbackBuffer.Read());

                //Update ringbuffer addresses
                Engine.CallbackBuffer.FinishRead();
            }

            END_BLOCK();
            BEGIN_BLOCK("GUI Render");

            //Start GUI frame
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            
            // Any application code here
            ImGui::Begin("Debug");
#if CORE_PROFILE            
            ImGui::LabelText("Total Frame Count", "%d", GlobalDebugState->TotalFrameCount);
            ImGui::LabelText("Target Frame Time", "%f", TargetSecondsPerFrame);
            ImGui::LabelText("Last Frame Time", "%f", GlobalDebugState->Frames[GlobalDebugState->ViewingFrameOrdinal].WallSecondsElapsed);
            ImGui::LabelText("Difference", "%f", (GlobalDebugState->Frames[GlobalDebugState->ViewingFrameOrdinal].WallSecondsElapsed - TargetSecondsPerFrame));
            if(GlobalDebugState->TotalFrameCount > 1)
            {
                u64 MinDuration = ((u64)-1);
                u64 MaxDuration = 0;
                u64 CurDuration = 0;
                u64 TotalDuration = 0;
                u32 TotalCount = 5;                
                u64 Duration = 0;
                for(size_t i = 0; i < TotalCount; i += 2)
                {
                    Duration += GlobalDebugState->StoredEvents[i].ClockDuration;
                    if(MinDuration > Duration)
                    {
                        MinDuration = Duration;
                    }
                    
                    if(MaxDuration < Duration)
                    {
                        MaxDuration = Duration;
                    }

                    TotalDuration += Duration;
                }

                u32 MinKilocycles = (u32) (MinDuration / 1000);
                u32 MaxKilocycles = (u32) (MaxDuration / 1000);
                u32 AvgKilocycles = (u32) SafeRatio0((f32) TotalDuration, (1000.0f * (f32) TotalCount));                

                ImGui::LabelText("Breakdown", "Min: %u | Max: %u | Avg: %u", MinKilocycles, MaxKilocycles, AvgKilocycles);
                if(GlobalEventCount > 0)
                {
                    for(size_t ei = 0; ei < GlobalEventCount - 1; ei += 2)
                    {
                        ImGui::LabelText(GlobalDebugState->StoredEvents[ei].Name, "%u", (u32) GlobalDebugState->StoredEvents[ei].ClockDuration);
                    }
                }
            }
#endif            
            ImGui::End();
            ImGui::Begin("Bubbles");
            if(ImGui::Button("Play"))
            {
                GlobalFireEvent = true;
            }
            ImGui::SliderInt("Count", &GlobalBubbleCount, 1, MAX_BUBBLE_COUNT);
            ImGui::SliderFloat("BPS", &GlobalBubblesPerSec, 1, 10000);
            ImGui::SliderFloat("Maximum Radius", &GlobalRadius, 0.0001, 20);
            ImGui::SliderFloat("Probability", &GlobalProbability, 0.5, 5.0);
            ImGui::SliderFloat("Rise Threshold", &GlobalRiseCutoff, 0.1, 0.9);
            ImGui::SliderFloat("Amplitude", &GlobalAmplitude, 0.0000001, 1.0);
            ImGui::End();

            // Render
            ImGui::Render();
            int display_w, display_h;
            glfwGetFramebufferSize(window, &display_w, &display_h);
            glViewport(0, 0, display_w, display_h);
            glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(window);     

            END_BLOCK();
            BEGIN_BLOCK("Sleep");

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
                }
            }
            else
            {
                //!Missed frame rate!
                f32 Difference = (SecondsElapsedForFrame - TargetSecondsPerFrame);
                Fatal("linux: Missed frame rate!\tDifference: %f\t[Current: %f, Target: %f]", Difference, SecondsElapsedForFrame, TargetSecondsPerFrame);
            } 

            END_BLOCK();

            // Record frame time
            FRAME_MARKER(SecondsElapsedForFrame);

#if CORE_PROFILE
            GlobalDebugTable->CurrentEventArrayIndex = !GlobalDebugTable->CurrentEventArrayIndex;
            u64 ArrayIndex_EventIndex = AtomicExchangeU64(&GlobalDebugTable->EventArrayIndex_EventIndex, (u64)GlobalDebugTable->CurrentEventArrayIndex << 32);
            u32 EventArrayIndex = ArrayIndex_EventIndex >> 32;
            Assert(EventArrayIndex <= 1, "");
            u32 EventCount = ArrayIndex_EventIndex & 0xFFFFFFFF;

            DEBUGStart(GlobalDebugState);
            CollateDebugRecords(GlobalDebugState, EventCount, GlobalDebugTable->Events[EventArrayIndex]);
            DEBUGEnd(GlobalDebugState);
#endif

            //Prepare timers before next loop
            timespec EndCounter = linux_WallClock();
            LastCounter = EndCounter; 
        }

        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        glfwDestroyWindow(window);
        glfwTerminate();

        Engine.CallbackBuffer.Destroy();
    }
    else
    {
        Fatal("linux: Failed to create window!");
    }

    //Free pools
    Pools.Names.Destroy();
    Pools.Buffers.Destroy();
    Pools.Partials.Destroy();
    Pools.Phases.Destroy();
    Pools.WAVs.Destroy();
    Pools.Bubble.Generators.Destroy();
    Pools.Bubble.Radii.Destroy();
    Pools.Bubble.Lambda.Destroy();
    Pools.Breakpoints.Destroy();

    //Free arenas
    SourceArena.Destroy();
    EngineHeap.Free(0, HEAP_TAG_MIXER);
    EngineHeap.Free(0, HEAP_TAG_PLATFORM);
    EngineHeap.Free(0, HEAP_TAG_ENTITY_SOURCE);
    EngineHeap.Free(0, HEAP_TAG_ENTITY_VOICE);
    EngineHeap.Free(0, HEAP_TAG_SYSTEM_VOICE, HEAP_TAG_SYSTEM_GRAIN);
    EngineHeap.Free(0, HEAP_TAG_UNDEFINED);
    EngineHeap.Destroy();
#if CORE_PROFILE
    DebugArena.Free(0);
    DebugArena.Destroy();
#endif

    // Destroy virtual alloctor
    LinuxAllocator.Free(SourceBlock, Megabytes(100));
    LinuxAllocator.Free(EngineBlock, Megabytes(900));
    LinuxAllocator.Destroy();

    //Destroy logging function - close file
    core_DestroyLogger();
}