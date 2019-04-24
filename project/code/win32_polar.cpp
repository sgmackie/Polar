
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

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <timeapi.h>
#define ANSI(code) ""

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

//D3D9 contexts for GUI rendering
static LPDIRECT3D9              D3D9 = NULL;
static LPDIRECT3DDEVICE9        D3Device = NULL;
static D3DPRESENT_PARAMETERS    D3DeviceParamters = {};

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
        return 0;
    }
    return ::DefWindowProc(Window, Message, WParam, LParam);
}

void win32_ProcessMessages()
{
    MSG Queue;

    while(PeekMessageW(&Queue, NULL, 0, 0, PM_REMOVE)) 
    {
        if(Queue.message == WM_QUIT)
        {
            GlobalRunning = false;
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

    void Create(MEMORY_ARENA *Arena, u32 InputSampleRate, u32 InputChannelCount, u32 InputBitRate, size_t InputBufferSize)
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
        DeviceFormat = (WAVEFORMATEXTENSIBLE *) Arena->Alloc(sizeof(WAVEFORMATEXTENSIBLE), MEMORY_ARENA_ALIGNMENT);
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
#if LOGGER_ERROR    
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

    //Get CUDA Device
    i32 DeviceCount = 0;
    GlobalUseCUDA = (cudaGetDeviceCount(&DeviceCount) == cudaSuccess && DeviceCount != 0);
    
    if(GlobalUseCUDA)
    {
        CUDA_DEVICE GPU = {};
        cuda_DeviceGet(&GPU, 0);
        cuda_DevicePrint(&GPU);
    }

    // f32 *Buffer = (f32 *) malloc(sizeof(f32) * 4096);
    // u32 ThreadBlock = 512;
    // cuda_SineArray(4096, 48000, ThreadBlock, Buffer);
    // printf("\n");
    // free(Buffer);
    // return 0;

    //Allocate memory arenas from virtual pages
    MEMORY_ARENA EngineArena = {};
    MEMORY_ARENA SourceArena = {};
    void *EngineBlock = VirtualAlloc(0, Kilobytes(500), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    void *SourceBlock = VirtualAlloc(0, Megabytes(100), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    EngineArena.Init(EngineBlock, Kilobytes(500));    
    SourceArena.Init(SourceBlock, Megabytes(100));
    Assert(EngineBlock && SourceBlock, "win32: Failed to create memory arenas!");

    //Create memory pools for component memory
    MEMORY_POOL SourcePoolNames = {};
    MEMORY_POOL SourcePoolBuffers = {};
    MEMORY_POOL SourcePoolBreakpoints = {};
    SourcePoolNames.Init(SourceArena.Alloc(Megabytes(10), MEMORY_ARENA_ALIGNMENT), Megabytes(10), (sizeof(char) * MAX_STRING_LENGTH), MEMORY_POOL_ALIGNMENT);
    SourcePoolBuffers.Init(SourceArena.Alloc(Megabytes(10), MEMORY_ARENA_ALIGNMENT), Megabytes(10), (sizeof(f32) * MAX_BUFFER_SIZE), MEMORY_POOL_ALIGNMENT);
    SourcePoolBreakpoints.Init(SourceArena.Alloc(Megabytes(10), MEMORY_ARENA_ALIGNMENT), Megabytes(10), (sizeof(CMP_BREAKPOINT_POINT) * MAX_BREAKPOINTS), MEMORY_POOL_ALIGNMENT);
    Assert(SourcePoolNames.Data && SourcePoolBuffers.Data && SourcePoolBreakpoints.Data, "win32: Failed to create source memory pools!");

    //Create memory pool for mixers
    MEMORY_POOL MixerPool = {};
    MEMORY_POOL MixerIntermediatePool = {};
    MixerPool.Init(SourceArena.Alloc(Megabytes(10), MEMORY_ARENA_ALIGNMENT), Megabytes(10), sizeof(SYS_MIX), MEMORY_POOL_ALIGNMENT);
    MixerIntermediatePool.Init(SourceArena.Alloc(Megabytes(10), MEMORY_ARENA_ALIGNMENT), Megabytes(10), (sizeof(f32) * MAX_BUFFER_SIZE), MEMORY_POOL_ALIGNMENT);
    Assert(MixerPool.Data && MixerIntermediatePool.Data, "win32: Failed to create mixer memory pools!");

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
        WASAPI.Create(&EngineArena, DEFAULT_SAMPLERATE, 2, 16, DEFAULT_SAMPLERATE);

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
        Engine.CallbackBuffer.Create(&EngineArena, sizeof(i16), 4096, 3);
        Assert(Engine.CallbackBuffer.Data, "win32: Failed to create callback buffer!");
        
        //OSC setup
        //!Replace with vanilla C version
        UdpSocket OSCSocket = polar_OSC_StartServer(4795);

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
        OscillatorModule.Triangle.Create(&SourceArena, MAX_SOURCES);
        OscillatorModule.Sawtooth.Create(&SourceArena, MAX_SOURCES);

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

        //!TODO: Fix WAV hash parser
        // FILE *File = 0;
        // fopen_s(&File, "data/sourcesWav_HASH.csv", "r");
        // int done = 0;
        // int err = 0;

        // for(u32 i = 0; i < MAX_SOURCES && done != 1; ++i)
        // {
        //     char *Line = fread_csv_line(File, MAX_STRING_LENGTH, &done, &err);
        //     if(done != 1)
        //     {
        //         char **Values = split_on_unescaped_newlines(Line);

        //         if(!err)
        //         {
        //             ID_SOURCE Hash;
        //             char WAV[MAX_STRING_LENGTH];
        //             sscanf(*Values, "%llu,%s", &Hash, WAV);
    
        //             ID_SOURCE Source = SoundSources.AddByHash(Hash);

        //             //Allocate wav
        //             SoundSources.WAVs[i].Init(WAV);
        //             SoundSources.Flags[i] |= ENTITY_SOURCES::WAV;  

        //             //Add to play system
        //             SoundSources.Buffers[i].CreateFromPool(&SourcePoolBuffers, sizeof(f32), MAX_BUFFER_SIZE);
        //             SoundSources.Flags[i] |= ENTITY_SOURCES::BUFFER;
        //             PlaySystem.Add(Source);     

        //             //Add to fade system
        //             SoundSources.Amplitudes[i].Init(0.1);
        //             SoundSources.Flags[i] |= ENTITY_SOURCES::AMPLITUDE;
        //             FadeSystem.Add(Source);

        //             //Add to mixer
        //             GlobalMixer.Mixes[0]->Add(Source);     
        //         }
        //     }
        // }   
        // fclose(File);

        //Start timings
        LARGE_INTEGER LastCounter = win32_WallClock();
        LARGE_INTEGER FlipWallClock = win32_WallClock();
        u64 LastCycleCount = __rdtsc();

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
            GlobalTime = core_WallTime();

            //Get OSC messages from Unreal
            //!Uses std::vector for message allocation: replace with pool allocations
            polar_OSC_UpdateMessages(GlobalTime, OSCSocket, 1);

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

                //Add to pan system
                SoundSources.Pans[Index].Init(CMP_PAN::MODE::WIDE, 0.0);
                SoundSources.Flags[Index] |= ENTITY_SOURCES::PAN;                

                //Add to oscillator system
                SoundSources.Oscillators[Index].Init(CMP_OSCILLATOR::SINE, SoundSources.Playbacks[Index].Format.SampleRate, 440);
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
                PlaySystem.Start(&SoundSources, ID, 5.0, 30);
                GlobalMixer.Mixes[0]->Add(ID);
            }

            //Update & Render
            //Write data
            if(Engine.CallbackBuffer.CanWrite())
            {
                //Update systems
                //Sample counts
                PlaySystem.Update(&SoundSources, GlobalTime, GlobalSamplesWritten, SamplesToWrite);
                
                //Source types
                //Oscillators
                OscillatorModule.Sine.Update(&SoundSources, SamplesToWrite);
                OscillatorModule.Square.Update(&SoundSources, SamplesToWrite);
                OscillatorModule.Triangle.Update(&SoundSources, SamplesToWrite);
                OscillatorModule.Sawtooth.Update(&SoundSources, SamplesToWrite);
                
                //Noise generators
                NoiseModule.White.Update(&SoundSources, SamplesToWrite);
                NoiseModule.Brown.Update(&SoundSources, SamplesToWrite);
                
                //Files
                WavSystem.Update(&SoundSources, 1.0, SamplesToWrite);
                
                //Amplitudes
                // BreakpointSystem.Update(&SoundSources, &FadeSystem, GlobalTime);
                // ADSRSystem.Update(&SoundSources, SamplesToWrite);
                FadeSystem.Update(&SoundSources, GlobalTime);
                
                //Clear mixer channels to 0
                f32 *MixerChannel0 = (f32 *) MixerIntermediatePool.Alloc();
                f32 *MixerChannel1 = (f32 *) MixerIntermediatePool.Alloc();
                memset(MixerChannel0, 0, (sizeof(f32) * SamplesToWrite));
                memset(MixerChannel1, 0, (sizeof(f32) * SamplesToWrite));

                //Render all sources in a mix the temporary buffer
                GlobalSamplesWritten = GlobalMixer.Mixes[0]->Update(&SoundSources, MixerChannel0, MixerChannel1, SamplesToWrite);
                
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

                //Free mixer channels
                MixerIntermediatePool.Free(MixerChannel0);
                MixerIntermediatePool.Free(MixerChannel1);

                //Update ringbuffer addresses
                Engine.CallbackBuffer.FinishWrite();
            }

            //Read data
            if(Engine.CallbackBuffer.CanRead())
            {
                //Fill WASAPI BYTE buffer
                win32_WASAPI_Callback(&WASAPI, SamplesToWrite, Engine.Format.Channels, Engine.CallbackBuffer.Read());

                //Update ringbuffer addresses
                Engine.CallbackBuffer.FinishRead();
            }

            //Start GUI frame
            ImGui_ImplDX9_NewFrame();
            ImGui_ImplWin32_NewFrame();
            ImGui::NewFrame();

            // Rendering
            ImGui::EndFrame();
            D3Device->SetRenderState(D3DRS_ZENABLE, false);
            D3Device->SetRenderState(D3DRS_ALPHABLENDENABLE, false);
            D3Device->SetRenderState(D3DRS_SCISSORTESTENABLE, false);
            D3DCOLOR clear_col_dx = D3DCOLOR_RGBA((int)(clear_color.x*255.0f), (int)(clear_color.y*255.0f), (int)(clear_color.z*255.0f), (int)(clear_color.w*255.0f));
            D3Device->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, clear_col_dx, 1.0f, 0);
            if (D3Device->BeginScene() >= 0)
            {
                ImGui::Render();
                ImGui_ImplDX9_RenderDrawData(ImGui::GetDrawData());
                D3Device->EndScene();
            }
            HRESULT result = D3Device->Present(NULL, NULL, NULL, NULL);

            // Handle loss of D3D9 device
            if (result == D3DERR_DEVICELOST && D3Device->TestCooperativeLevel() == D3DERR_DEVICENOTRESET)
                ResetDevice();

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
            LARGE_INTEGER EndCounter = win32_WallClock();
            LastCounter = EndCounter;         
        }

        ImGui_ImplDX9_Shutdown();
        ImGui_ImplWin32_Shutdown();
        ImGui::DestroyContext();

        Engine.CallbackBuffer.Destroy();
        WASAPI.Destroy();

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