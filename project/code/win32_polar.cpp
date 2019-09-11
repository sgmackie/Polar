
#include "polar.h"

#define DEFAULT_WIDTH 1280
#define DEFAULT_HEIGHT 720
#define DEFAULT_HZ 120

#define DEFAULT_SAMPLERATE 48000
#define DEFAULT_CHANNELS 2
#define DEFAULT_AMPLITUDE 0.8
#define DEFAULT_LATENCY_FRAMES 4

//Latency frames determines update rate - 4 @ 120HZ = 30FPS

//IMGUI implementation - DirectX9
#include <d3d9.h>
#include "../external/imgui/win32/imgui_impl_dx9.cpp"
#include "../external/imgui/win32/imgui_impl_win32.cpp"

//WASAPI includes
#include <audioclient.h>                    //WASAPI
#include <initguid.h>
#include <mmdeviceapi.h>                    //Audio endpoints
#include <Functiondiscoverykeys_devpkey.h>  //Used for getting "FriendlyNames" from audio endpoints
#include <avrt.h>

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

//D3D9 contexts for GUI rendering
static LPDIRECT3D9              D3D9 = NULL;
static LPDIRECT3DDEVICE9        D3Device = NULL;
static D3DPRESENT_PARAMETERS    D3DeviceParamters = {};

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

bool CreateDeviceD3D(HWND hWnd)
{
    if ((D3D9 = Direct3DCreate9(D3D_SDK_VERSION)) == NULL)
        return false;

    // Create the D3DDevice
    ZeroMemory(&D3DeviceParamters, sizeof(D3DeviceParamters));
    D3DeviceParamters.Windowed = TRUE;
    D3DeviceParamters.SwapEffect = D3DSWAPEFFECT_DISCARD;
    D3DeviceParamters.BackBufferFormat = D3DFMT_UNKNOWN;
    D3DeviceParamters.EnableAutoDepthStencil = TRUE;
    D3DeviceParamters.AutoDepthStencilFormat = D3DFMT_D16;
    D3DeviceParamters.PresentationInterval = D3DPRESENT_INTERVAL_ONE;           // Present with vsync
    //D3DeviceParamters.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;   // Present without vsync, maximum unthrottled framerate
    if (D3D9->CreateDevice(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, hWnd, D3DCREATE_HARDWARE_VERTEXPROCESSING, &D3DeviceParamters, &D3Device) < 0)
        return false;

    return true;
}

void CleanupDeviceD3D()
{
    if (D3Device) { D3Device->Release(); D3Device = NULL; }
    if (D3D9) { D3D9->Release(); D3D9 = NULL; }
}

void ResetDevice()
{
    ImGui_ImplDX9_InvalidateDeviceObjects();
    HRESULT hr = D3Device->Reset(&D3DeviceParamters);
    if (hr == D3DERR_INVALIDCALL)
        IM_ASSERT(0);
    ImGui_ImplDX9_CreateDeviceObjects();
}

LRESULT CALLBACK WindowProc(HWND Window, UINT Message, WPARAM WParam, LPARAM LParam)
{
    if (ImGui_ImplWin32_WndProcHandler(Window, Message, WParam, LParam))
        return true;

    switch (Message)
    {
    case WM_SIZE:
        if (D3Device != NULL && WParam != SIZE_MINIMIZED)
        {
            D3DeviceParamters.BackBufferWidth = LOWORD(LParam);
            D3DeviceParamters.BackBufferHeight = HIWORD(LParam);
            ResetDevice();
        }
        return 0;
    case WM_SYSCOMMAND:
        if ((WParam & 0xfff0) == SC_KEYMENU) // Disable ALT application menu
            return 0;
        break;
    case WM_DESTROY:
        ::PostQuitMessage(0);
        GlobalRunning = false;
        return 0;
    case WM_CLOSE:
    {
        GlobalRunning = false;
        return 0;
    }
    }
    return ::DefWindowProc(Window, Message, WParam, LParam);
}

void win32_ProcessMessages()
{
    MSG Queue;

    while(PeekMessageW(&Queue, NULL, 0, 0, PM_REMOVE)) 
    {
        switch(Queue.message)
        {
            case WM_CLOSE:
            {
                GlobalRunning = false;

                break;
            }
            case WM_DESTROY:
            {
                GlobalRunning = false;
                break;
            }     
            // Virtual Keycode parsing
            case WM_KEYUP:
            {
                // Check key state
                u32 VKCode              = Queue.wParam;
                bool WasDown            = ((Queue.lParam & (1 << 30)) != 0);
                bool IsDown             = ((Queue.lParam & (1UL << 31)) == 0);

                // Switch
                if(WasDown != IsDown)
                {       
                    if(VKCode == 'P')
                    {
                        GlobalFireEvent = true;
                    }
                }

                break;
            } 
        }

        TranslateMessage(&Queue);
        DispatchMessageW(&Queue);
    }
}

//Reference times as variable
static const u64 REF_TIMES_PER_SECOND = 10000000;

//Convert WASAPI HRESULT to printable string
static const TCHAR *wasapi_HRString(HRESULT Result)
{
	switch(Result)
	{
		case S_OK:										return TEXT("S_OK");
		case S_FALSE:									return TEXT("S_FALSE");
		case AUDCLNT_E_NOT_INITIALIZED:					return TEXT("AUDCLNT_E_NOT_INITIALIZED");
		case AUDCLNT_E_ALREADY_INITIALIZED:				return TEXT("AUDCLNT_E_ALREADY_INITIALIZED");
		case AUDCLNT_E_WRONG_ENDPOINT_TYPE:				return TEXT("AUDCLNT_E_WRONG_ENDPOINT_TYPE");
		case AUDCLNT_E_DEVICE_INVALIDATED:				return TEXT("AUDCLNT_E_DEVICE_INVALIDATED");
		case AUDCLNT_E_NOT_STOPPED:						return TEXT("AUDCLNT_E_NOT_STOPPED");
		case AUDCLNT_E_BUFFER_TOO_LARGE:				return TEXT("AUDCLNT_E_BUFFER_TOO_LARGE");
		case AUDCLNT_E_OUT_OF_ORDER:					return TEXT("AUDCLNT_E_OUT_OF_ORDER");
		case AUDCLNT_E_UNSUPPORTED_FORMAT:				return TEXT("AUDCLNT_E_UNSUPPORTED_FORMAT");
		case AUDCLNT_E_INVALID_SIZE:					return TEXT("AUDCLNT_E_INVALID_SIZE");
		case AUDCLNT_E_DEVICE_IN_USE:					return TEXT("AUDCLNT_E_DEVICE_IN_USE");
		case AUDCLNT_E_BUFFER_OPERATION_PENDING:		return TEXT("AUDCLNT_E_BUFFER_OPERATION_PENDING");
		case AUDCLNT_E_THREAD_NOT_REGISTERED:			return TEXT("AUDCLNT_E_THREAD_NOT_REGISTERED");
		case AUDCLNT_E_EXCLUSIVE_MODE_NOT_ALLOWED:		return TEXT("AUDCLNT_E_EXCLUSIVE_MODE_NOT_ALLOWED");
		case AUDCLNT_E_ENDPOINT_CREATE_FAILED:			return TEXT("AUDCLNT_E_ENDPOINT_CREATE_FAILED");
		case AUDCLNT_E_SERVICE_NOT_RUNNING:				return TEXT("AUDCLNT_E_SERVICE_NOT_RUNNING");
		case AUDCLNT_E_EVENTHANDLE_NOT_EXPECTED:		return TEXT("AUDCLNT_E_EVENTHANDLE_NOT_EXPECTED");
		case AUDCLNT_E_EXCLUSIVE_MODE_ONLY:				return TEXT("AUDCLNT_E_EXCLUSIVE_MODE_ONLY");
		case AUDCLNT_E_BUFDURATION_PERIOD_NOT_EQUAL:	return TEXT("AUDCLNT_E_BUFDURATION_PERIOD_NOT_EQUAL");
		case AUDCLNT_E_EVENTHANDLE_NOT_SET:				return TEXT("AUDCLNT_E_EVENTHANDLE_NOT_SET");
		case AUDCLNT_E_INCORRECT_BUFFER_SIZE:			return TEXT("AUDCLNT_E_INCORRECT_BUFFER_SIZE");
		case AUDCLNT_E_BUFFER_SIZE_ERROR:				return TEXT("AUDCLNT_E_BUFFER_SIZE_ERROR");
		case AUDCLNT_E_CPUUSAGE_EXCEEDED:				return TEXT("AUDCLNT_E_CPUUSAGE_EXCEEDED");
		case AUDCLNT_E_BUFFER_ERROR:					return TEXT("AUDCLNT_E_BUFFER_ERROR");
		case AUDCLNT_E_BUFFER_SIZE_NOT_ALIGNED:			return TEXT("AUDCLNT_E_BUFFER_SIZE_NOT_ALIGNED");
		case AUDCLNT_E_INVALID_DEVICE_PERIOD:			return TEXT("AUDCLNT_E_INVALID_DEVICE_PERIOD");
		case AUDCLNT_E_INVALID_STREAM_FLAG:				return TEXT("AUDCLNT_E_INVALID_STREAM_FLAG");
		case AUDCLNT_E_ENDPOINT_OFFLOAD_NOT_CAPABLE:	return TEXT("AUDCLNT_E_ENDPOINT_OFFLOAD_NOT_CAPABLE");
		case AUDCLNT_E_OUT_OF_OFFLOAD_RESOURCES:		return TEXT("AUDCLNT_E_OUT_OF_OFFLOAD_RESOURCES");
		case AUDCLNT_E_OFFLOAD_MODE_ONLY:				return TEXT("AUDCLNT_E_OFFLOAD_MODE_ONLY");
		case AUDCLNT_E_NONOFFLOAD_MODE_ONLY:			return TEXT("AUDCLNT_E_NONOFFLOAD_MODE_ONLY");
		case AUDCLNT_E_RESOURCES_INVALIDATED:			return TEXT("AUDCLNT_E_RESOURCES_INVALIDATED");
		case AUDCLNT_E_RAW_MODE_UNSUPPORTED:			return TEXT("AUDCLNT_E_RAW_MODE_UNSUPPORTED");
		case REGDB_E_CLASSNOTREG:						return TEXT("REGDB_E_CLASSNOTREG");
		case CLASS_E_NOAGGREGATION:						return TEXT("CLASS_E_NOAGGREGATION");
		case E_NOINTERFACE:								return TEXT("E_NOINTERFACE");
		case E_POINTER:									return TEXT("E_POINTER");
		case E_INVALIDARG:								return TEXT("E_INVALIDARG");
		case E_OUTOFMEMORY:								return TEXT("E_OUTOFMEMORY");
		default:										return TEXT("UNKNOWN");
	}
}

#define NONE    //Blank space for returning nothing in void functions

//Use print and return on HRESULT codes
#define HR_TO_RETURN(Result, Text, Type)				                    \
	if(FAILED(Result))								                        \
	{												                        \
		char HRBuffer[256];													\
		OutputDebugString(HRBuffer);										\
		sprintf_s(HRBuffer, Text "\t[%s]\n", wasapi_HRString(Result));   	\
		return Type;								                        \
	}


typedef struct WASAPI
{
    //Data
    HRESULT HR;
    HANDLE RenderEvent;
    WAVEFORMATEXTENSIBLE *DeviceFormat;
	
    //Device endpoints
	IMMDeviceEnumerator *DeviceEnumerator;
	IMMDevice *AudioDevice;

	//Rendering clients
	IAudioClient *AudioClient;
	IAudioRenderClient *AudioRenderClient;

	u32 PaddingFrames;
	u32 BufferFrames;

    //Functions
    void Init()
    {
        HRESULT HR = 0;
        HANDLE RenderEvent = 0;
        WAVEFORMATEXTENSIBLE *DeviceFormat = 0;
	    IMMDeviceEnumerator *DeviceEnumerator = 0;
	    IMMDevice *AudioDevice = 0;
	    IAudioClient *AudioClient = 0;
	    IAudioRenderClient *AudioRenderClient = 0;
	    u32 PaddingFrames = 0;
	    u32 BufferFrames = 0;
    }

    void Create(MEMORY_ALLOCATOR *Allocator, u32 InputSampleRate, u32 InputChannelCount, u32 InputBitRate, size_t InputBufferSize)
    {
        Init();

        HR = CoInitializeEx(0, COINIT_SPEED_OVER_MEMORY);
	    HR_TO_RETURN(HR, "Failed to initialise COM", NONE);

        RenderEvent = CreateEvent(0, 0, 0, 0);
	    if(!RenderEvent)
	    {
	    	HR_TO_RETURN(HR, "Failed to create event", NONE);
	    }

        HR = CoCreateInstance(__uuidof(MMDeviceEnumerator), NULL, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator), (void **) &DeviceEnumerator);
        HR_TO_RETURN(HR, "Failed to create device COM", NONE);

        HR = DeviceEnumerator->GetDefaultAudioEndpoint(eRender, eConsole, &AudioDevice);
        HR_TO_RETURN(HR, "Failed to get default audio endpoint", NONE);

	    HR = AudioDevice->Activate(__uuidof(IAudioClient), CLSCTX_ALL, NULL, (void**) &AudioClient);
	    HR_TO_RETURN(HR, "Failed to activate audio endpoint", NONE);

        WAVEFORMATEXTENSIBLE *MixFormat;
	    HR = AudioClient->GetMixFormat((WAVEFORMATEX **) &MixFormat);
	    HR_TO_RETURN(HR, "Failed to activate audio endpoint", NONE);

        //Create output format
        DeviceFormat = 0;
        DeviceFormat = (WAVEFORMATEXTENSIBLE *) Allocator->Alloc(sizeof(WAVEFORMATEXTENSIBLE), HEAP_TAG_PLATFORM);
        DeviceFormat->Format.cbSize = sizeof(WAVEFORMATEXTENSIBLE);
        DeviceFormat->Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE;
        DeviceFormat->Format.wBitsPerSample = InputBitRate;
        DeviceFormat->Format.nChannels = InputChannelCount;
        DeviceFormat->Format.nSamplesPerSec = (DWORD) InputSampleRate;
        DeviceFormat->Format.nBlockAlign = (WORD) (DeviceFormat->Format.nChannels * DeviceFormat->Format.wBitsPerSample / 8);
        DeviceFormat->Format.nAvgBytesPerSec = DeviceFormat->Format.nSamplesPerSec * DeviceFormat->Format.nBlockAlign;
        DeviceFormat->Samples.wValidBitsPerSample = InputBitRate;
        DeviceFormat->dwChannelMask = KSAUDIO_SPEAKER_STEREO;
        DeviceFormat->SubFormat = KSDATAFORMAT_SUBTYPE_PCM;

        //If the current device sample rate doesn't equal the output, than set WASAPI to autoconvert
        DWORD Flags = 0;
        if(MixFormat->Format.nSamplesPerSec != DeviceFormat->Format.nSamplesPerSec)
        {
            Warning("WASAPI: Sample rate does not equal the requested rate, resampling\t Result: %lu\t Requested: %lu", MixFormat->Format.nSamplesPerSec, DeviceFormat->Format.nSamplesPerSec);
            Flags = AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY;
        }

        //Free reference format
        CoTaskMemFree(MixFormat);

        //Buffer size in 100 nano second units
        REFERENCE_TIME BufferDuration = 10000000ULL * InputBufferSize / DeviceFormat->Format.nSamplesPerSec;
	    HR = AudioClient->Initialize(AUDCLNT_SHAREMODE_SHARED, Flags, BufferDuration, 0, &DeviceFormat->Format, NULL);
        HR_TO_RETURN(HR, "Failed to initialise audio client", NONE);

	    HR = AudioClient->GetService(__uuidof(IAudioRenderClient), (void**) &AudioRenderClient);
	    HR_TO_RETURN(HR, "Failed to assign client to render client", NONE);

        HR = AudioClient->GetBufferSize(&BufferFrames);
	    HR_TO_RETURN(HR, "Failed to get maximum read buffer size for audio client", NONE);

	    HR = AudioClient->Reset();
	    HR_TO_RETURN(HR, "Failed to reset audio client before playback", NONE);

	    HR = AudioClient->Start();
	    HR_TO_RETURN(HR, "Failed to start audio client", NONE);

        if(BufferFrames != InputBufferSize)
        {
            Warning("WASAPI: WASAPI buffer size does not equal requested size!\t Result: %u\t Requested: %u", BufferFrames, InputBufferSize);
        }
    }

    void Destroy()
    {
	    AudioRenderClient->Release();
	    AudioClient->Reset();
	    AudioClient->Stop();
	    AudioClient->Release();
	    AudioDevice->Release();

	    CoUninitialize();

        Init();
    }

} WASAPI;

void win32_WASAPI_Callback(WASAPI *WASAPI, u32 SampleCount, u32 Channels, i16 *OutputBuffer)
{
    BYTE* BYTEBuffer;
    
    if(SUCCEEDED(WASAPI->AudioRenderClient->GetBuffer((UINT32) SampleCount, &BYTEBuffer)))
    {
        int16* SourceSample = OutputBuffer;
        int16* DestSample = (int16*) BYTEBuffer;
        for(size_t SampleIndex = 0; SampleIndex < SampleCount; ++SampleIndex)
        {
            for(u32 ChannelIndex = 0; ChannelIndex < Channels; ++ChannelIndex)
            {
                *DestSample++ = *SourceSample++;
            }
        }

        WASAPI->AudioRenderClient->ReleaseBuffer((UINT32) SampleCount, 0);
    }
}


int main()
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
        Info("win32: File logger created succesfully");
    }
    else
    {
        printf("win32: Failed to create logger!\n");
    }

#if CUDA
    //Get CUDA Device
    i32 DeviceCount = 0;
    GlobalUseCUDA = (cudaGetDeviceCount(&DeviceCount) == cudaSuccess && DeviceCount != 0);
    
    if(GlobalUseCUDA)
    {
        CUDA_DEVICE GPU = {};
        cuda_DeviceGet(&GPU, 0);
        cuda_DevicePrint(&GPU);
    }
#endif


    // Create allocators
    // Platform allocator for getting the fixed blocks
    MEMORY_ALLOCATOR Win32Allocator = {};
    Win32Allocator.Create(MEMORY_ALLOCATOR_VIRTUAL, "Win32 Virtual Allocator", 0, Gigabytes(1));
    void *EngineBlock = Win32Allocator.Alloc(Megabytes(100));
    void *SourceBlock = Win32Allocator.Alloc(Megabytes(900));

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

    //Create window and it's rendering handle
    WNDCLASSEX WindowClass = {sizeof(WNDCLASSEX), 
                            CS_CLASSDC, WindowProc, 0L, 0L, 
                            GetModuleHandle(NULL), NULL, NULL, NULL, NULL,
                            _T("PolarClass"), NULL };

    RegisterClassEx(&WindowClass);

    HWND WindowHandle = CreateWindow(WindowClass.lpszClassName, 
                        _T("Polar"), WS_OVERLAPPEDWINDOW, 100, 100, 
                        DEFAULT_WIDTH, DEFAULT_HEIGHT, 
                        NULL, NULL, WindowClass.hInstance, NULL);

    if(WindowHandle && CreateDeviceD3D(WindowHandle))
    {
        //Display window
        ShowWindow(WindowHandle, SW_SHOWDEFAULT);
        UpdateWindow(WindowHandle);

        //Get monitor refresh rate
        HDC RefreshDC = GetDC(WindowHandle);
        i32 MonitorRefresh = GetDeviceCaps(RefreshDC, VREFRESH);
        ReleaseDC(WindowHandle, RefreshDC);
        if(MonitorRefresh < 1)
        {
            MonitorRefresh = DEFAULT_HZ;
        }

        //Start timings
        LARGE_INTEGER PerformanceCounterFrequencyResult;
        QueryPerformanceFrequency(&PerformanceCounterFrequencyResult);
        GlobalPerformanceCounterFrequency = PerformanceCounterFrequencyResult.QuadPart;

        //Request 1ms period for timing functions
        UINT SchedulerPeriodInMS = 1;
        bool IsSleepGranular = (timeBeginPeriod(SchedulerPeriodInMS) == TIMERR_NOERROR);

        //Define engine update rate
        POLAR_ENGINE Engine = {};
        Engine.Init((u32) WallTime(), (MonitorRefresh / DEFAULT_LATENCY_FRAMES));

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

        //Bind to DX9 renderer
        ImGui_ImplWin32_Init(WindowHandle);
        ImGui_ImplDX9_Init(D3Device);
            
        //Start WASAPI
        WASAPI WASAPI = {};
        WASAPI.Create(&EngineHeap, DEFAULT_SAMPLERATE, 2, 16, DEFAULT_SAMPLERATE);

        //Fill out engine properties
        Engine.NoiseFloor           = DB(-120);
        Engine.Format.SampleRate    = WASAPI.DeviceFormat->Format.nSamplesPerSec;
        Engine.Format.Channels      = WASAPI.DeviceFormat->Format.nChannels;
        Engine.BytesPerSample       = sizeof(i16) * Engine.Format.Channels;
        Engine.BufferFrames         = WASAPI.BufferFrames;
        Engine.LatencyFrames        = DEFAULT_LATENCY_FRAMES * (Engine.Format.SampleRate / Engine.UpdateRate);

        //Buffer size:
        //The max buffer size is 1 second worth of samples
        //LatencySamples determines how many samples to render at a given frame delay (default is 2)
        //The sample count to write for each callback is the LatencySamples - any padding from the audio D3Device
        
        //Create ringbuffer with a specified block count (default is 3)
        Engine.CallbackBuffer.Create(&EngineHeap, sizeof(i16), 4096, 3);
        Assert(Engine.CallbackBuffer.Data, "win32: Failed to create callback buffer!");
        
        //OSC setup
        //!Replace with vanilla C version
        UdpSocket OSCSocket = polar_OSC_StartServer(4795);

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

        //!TODO: Fix WAV hash parser
        // FILE *File = 0;
        // fopen_s(&File, "data/sourcesWav_HASH.csv", "r");
        // int done = 0;
        // int err = 0;


        //Start timings
        LARGE_INTEGER LastCounter = win32_WallClock();
        LARGE_INTEGER FlipWallClock = win32_WallClock();
        u64 LastCycleCount = __rdtsc();

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
        while(GlobalRunning)
        {
            //Process incoming mouse/keyboard messages, check for QUIT command
            win32_ProcessMessages();
            if(!GlobalRunning) break;

            //Calculate size of callback sample block
            i32 SamplesToWrite = 0;
            i32 MaxSampleCount = 0;

            //Get current padding of the audio D3Device and determine samples to write for this callback
            if(SUCCEEDED(WASAPI.AudioClient->GetCurrentPadding(&WASAPI.PaddingFrames)))
            {
                MaxSampleCount = (i32) (Engine.BufferFrames - WASAPI.PaddingFrames);
                SamplesToWrite = (i32) (Engine.LatencyFrames - WASAPI.PaddingFrames);

                //Round the samples to write to the next power of 2
                MaxSampleCount = UpperPowerOf2(MaxSampleCount);
                SamplesToWrite = UpperPowerOf2(SamplesToWrite);

                if(SamplesToWrite < 0)
                {
                    UINT32 DeviceSampleCount = 0;
                    if(SUCCEEDED(WASAPI.AudioClient->GetBufferSize(&DeviceSampleCount)))
                    {
                        SamplesToWrite = DeviceSampleCount;
                    }
                }

                Assert(SamplesToWrite <= MaxSampleCount, "win32: Samples to write is bigger than the maximum!");
            }

            //Check the minimum update period for per-sample stepping states
            f64 MinPeriod = ((f64) SamplesToWrite / (f64) Engine.Format.SampleRate);

            //Get current time for update functions
            GlobalTime = WallTime();

            //Get OSC messages from Unreal
            //!Uses std::vector for message allocation: replace with pool allocations
            polar_OSC_UpdateMessages(&SoundSources, &SoundVoices, &GlobalListener, GlobalTime, OSCSocket, 10);

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
                
                Engine.Systems.Play.Start(&SoundVoices, GlobalBubbleVoice, 50.0, -1);
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
                win32_WASAPI_Callback(&WASAPI, SamplesToWrite, Engine.Format.Channels, Engine.CallbackBuffer.Read());

                //Update ringbuffer addresses
                Engine.CallbackBuffer.FinishRead();
            }

            END_BLOCK();
            BEGIN_BLOCK("GUI Render");

            //Start GUI frame
            ImGui_ImplDX9_NewFrame();
            ImGui_ImplWin32_NewFrame();
            
            // Any application code here
            ImGui::NewFrame();
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
            ImGui::SliderInt("Count", &GlobalBubbleCount, 1, MAX_BUBBLE_COUNT);
            ImGui::SliderFloat("BPS", &GlobalBubblesPerSec, 1, 10000);
            ImGui::SliderFloat("Maximum Radius", &GlobalRadius, 0.0001, 20);
            ImGui::SliderFloat("Probability", &GlobalProbability, 0.5, 5.0);
            ImGui::SliderFloat("Rise Threshold", &GlobalRiseCutoff, 0.1, 0.9);
            ImGui::SliderFloat("Amplitude", &GlobalAmplitude, 0.0000001, 1.0);
            ImGui::End();
            ImGui::EndFrame();

            // Rendering
            D3Device->SetRenderState(D3DRS_ZENABLE, false);
            D3Device->SetRenderState(D3DRS_ALPHABLENDENABLE, false);
            D3Device->SetRenderState(D3DRS_SCISSORTESTENABLE, false);
            D3DCOLOR clear_col_dx = D3DCOLOR_RGBA((int)(clear_color.x*255.0f), (int)(clear_color.y*255.0f), (int)(clear_color.z*255.0f), (int)(clear_color.w*255.0f));
            D3Device->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, clear_col_dx, 1.0f, 0);
            if(D3Device->BeginScene() >= 0)
            {
                ImGui::Render();
                ImGui_ImplDX9_RenderDrawData(ImGui::GetDrawData());
                D3Device->EndScene();
            }
            HRESULT result = D3Device->Present(NULL, NULL, NULL, NULL);

            // Handle loss of D3D9 device
            if (result == D3DERR_DEVICELOST && D3Device->TestCooperativeLevel() == D3DERR_DEVICENOTRESET)
                ResetDevice();

            END_BLOCK();
            BEGIN_BLOCK("Sleep");

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
                f32 Difference = (SecondsElapsedForFrame - TargetSecondsPerFrame);
                Fatal("win32: Missed frame rate!\tDifference: %f\t[Current: %f, Target: %f]", Difference, SecondsElapsedForFrame, TargetSecondsPerFrame);
            } 

            END_BLOCK();

            // Record frame time
            FRAME_MARKER(SecondsElapsedForFrame);
            
#if CORE_PROFILE
            GlobalDebugTable->CurrentEventArrayIndex = !GlobalDebugTable->CurrentEventArrayIndex;
            u64 ArrayIndex_EventIndex = AtomicExchangeU64(&GlobalDebugTable->EventArrayIndex_EventIndex, (u64)GlobalDebugTable->CurrentEventArrayIndex << 32);
            u32 EventArrayIndex = ArrayIndex_EventIndex >> 32;
            Assert(EventArrayIndex <= 1, "");
            GlobalEventCount = ArrayIndex_EventIndex & 0xFFFFFFFF;

            DEBUGStart(GlobalDebugState);
            CollateDebugRecords(GlobalDebugState, GlobalEventCount, GlobalDebugTable->Events[EventArrayIndex]);
            DEBUGEnd(GlobalDebugState);
#endif

            //Prepare timers before next loop
            LARGE_INTEGER EndCounter        = win32_WallClock();
            LastCounter = EndCounter;    
        }

        ImGui_ImplDX9_Shutdown();
        ImGui_ImplWin32_Shutdown();
        ImGui::DestroyContext();

        Engine.CallbackBuffer.Destroy();
        WASAPI.Destroy();

        //! TEMP
        for(size_t i = 0; i < SoundSources.Count; ++i)
        {
            if(SoundSources.Flags[i] & ENTITY_SOURCES::WAV)
            {
                SoundSources.Types[i].WAV.Destroy();
            } 
        }

        CleanupDeviceD3D();
        DestroyWindow(WindowHandle);
        UnregisterClass(WindowClass.lpszClassName, WindowClass.hInstance);        
    }
    else
    {
        CleanupDeviceD3D();
        DestroyWindow(WindowHandle);
        UnregisterClass(WindowClass.lpszClassName, WindowClass.hInstance);
        Fatal("win32: Failed to create window!");
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
    Win32Allocator.Free(SourceBlock, Megabytes(100));
    Win32Allocator.Free(EngineBlock, Megabytes(900));
    Win32Allocator.Destroy();

    //Destroy logging function - close file
    core_DestroyLogger();
}