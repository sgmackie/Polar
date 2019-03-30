#define DEFAULT_SAMPLERATE 48000
#define DEFAULT_CHANNELS 2
#define DEFAULT_AMPLITUDE 0.8
#define DEFAULT_LATENCY_FRAMES 3

#define DEFAULT_WIDTH 1280
#define DEFAULT_HEIGHT 720
#define DEFAULT_HZ 60

#include "polar.h"
#include "../external/external_code.h"
#if CUDA
#include "cuda/polar_cuda.h"
#endif
#include "win32_polar.h"

global_scope char AssetPath[MAX_STRING_LENGTH] = {"../../data/"};
global_scope i64 GlobalPerformanceCounterFrequency;
global_scope f64 GlobalTime = 0;
global_scope bool GlobalRunning = false;

//D3D9 contexts for GUI rendering
global_scope LPDIRECT3D9              D3D9 = NULL;
global_scope LPDIRECT3DDEVICE9        D3Device = NULL;
global_scope D3DPRESENT_PARAMETERS    D3DeviceParamters = {};

//Source
#include "polar.cpp"

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

    //Free reference format
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

#if CUDA
    //Get CUDA Device
    CUDA_DEVICE GPU = {};
    cuda_DeviceGet(&GPU, 0);

    printf("%f\n", cuda_Sine(0.63787, 1));

    cuda_Multiply(128, 256, 131072);

#endif

    if(EngineArena && SourceArena)
    {
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
            ShowWindow(WindowHandle, SW_SHOWDEFAULT);
            UpdateWindow(WindowHandle);

            //Start timings
            LARGE_INTEGER PerformanceCounterFrequencyResult;
            QueryPerformanceFrequency(&PerformanceCounterFrequencyResult);
            GlobalPerformanceCounterFrequency = PerformanceCounterFrequencyResult.QuadPart;

            //Request 1ms period for timing functions
            UINT SchedulerPeriodInMS = 1;
            bool IsSleepGranular = (timeBeginPeriod(SchedulerPeriodInMS) == TIMERR_NOERROR);

            //Get monitor refresh rate
            HDC RefreshDC = GetDC(WindowHandle);
            i32 MonitorRefresh = GetDeviceCaps(RefreshDC, VREFRESH);
            ReleaseDC(WindowHandle, RefreshDC);
            if(MonitorRefresh < 1)
            {
                MonitorRefresh = DEFAULT_HZ;
            }

            //Create GUI context
            IMGUI_CHECKVERSION();
            ImGui::CreateContext();
            ImGuiIO& io = ImGui::GetIO(); (void)io;

            //Set GUI style
            ImGui::StyleColorsDark();

            //Set GUI state
            bool show_demo_window = true;
            bool show_another_window = false;
            ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

            //Bind to DX9 renderer
            ImGui_ImplWin32_Init(WindowHandle);
            ImGui_ImplDX9_Init(D3Device);

            //Define engine update rate
            POLAR_ENGINE Engine = {};
            Engine.UpdateRate = (MonitorRefresh / DEFAULT_LATENCY_FRAMES);
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
            //The sample count to write for each callback is the LatencySamples - any padding from the audio D3Device

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
                LARGE_INTEGER LastCounter = win32_WallClock();
                LARGE_INTEGER FlipWallClock = win32_WallClock();
                u64 LastCycleCount = __rdtsc();

                //Loop
                i64 i = 0;
                GlobalTime = 0;
                GlobalRunning = true;
                Master->Amplitude = DB(-1);
                printf("Polar: Playback\n");
                while(GlobalRunning)
                {
                    //Updates
                    ++i;

                    //Process incoming mouse/keyboard messages
                    win32_ProcessMessages();

                    //Calculate size of callback sample block
                    i32 SamplesToWrite = 0;
                    i32 MaxSampleCount = 0;

                    //Get current padding of the audio D3Device and determine samples to write for this callback
                    if(SUCCEEDED(WASAPI->AudioClient->GetCurrentPadding(&WASAPI->PaddingFrames)))
                    {
                        MaxSampleCount = (i32) (Engine.BufferSize - WASAPI->PaddingFrames);
                        SamplesToWrite = (i32) (Engine.LatencySamples - WASAPI->PaddingFrames);

                        //Round the samples to write to the next power of 2
                        MaxSampleCount = UpperPowerOf2(MaxSampleCount);
                        SamplesToWrite = UpperPowerOf2(SamplesToWrite);

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
                        // polar_container_Play(Master, Hash("CO_FileContainer"), 0, FX_DRY, EN_NONE, AMP(-1));

                        polar_source_Play(Master, Hash("SO_Whiterun"), 0, FX_DRY, EN_NONE, AMP(-1));

                        // polar_container_Fade(Master, Hash("CO_FileContainer"), GlobalTime, AMP(-65), 12);


                        // polar_source_Play(Master, Hash("SO_Trumpet14_Partial_01"), 1, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial1.txt");
                        // polar_source_Play(Master, Hash("SO_Trumpet14_Partial_02"), 1, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial2.txt");
                        // polar_source_Play(Master, Hash("SO_Trumpet14_Partial_03"), 1, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial3.txt");
                        // polar_source_Play(Master, Hash("SO_Trumpet14_Partial_04"), 1, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial4.txt");
                        // polar_source_Play(Master, Hash("SO_Trumpet14_Partial_05"), 1, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial5.txt");
                        // polar_source_Play(Master, Hash("SO_Trumpet14_Partial_06"), 1, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial6.txt");
                        // polar_source_Play(Master, Hash("SO_Trumpet14_Partial_07"), 1, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial7.txt");
                        // polar_source_Play(Master, Hash("SO_Trumpet14_Partial_08"), 1, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial8.txt");
                        // polar_source_Play(Master, Hash("SO_Trumpet14_Partial_09"), 1, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial9.txt");
                        // polar_source_Play(Master, Hash("SO_Trumpet14_Partial_10"), 1, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial10.txt");
                        // polar_source_Play(Master, Hash("SO_Trumpet14_Partial_11"), 1, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial11.txt");
                        // polar_source_Play(Master, Hash("SO_Trumpet14_Partial_12"), 1, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial12.txt");
                        // polar_source_Play(Master, Hash("SO_Trumpet14_Partial_13"), 1, FX_DRY, EN_BREAKPOINT, "breakpoints/trumpet14/Trumpet_14_Partial13.txt");
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
                        //Fill WASAPI BYTE buffer
                        win32_WASAPI_Callback(WASAPI, MixBuffer->SampleCount, Engine.Channels, polar_ringbuffer_ReadData(CallbackBuffer));
                        // printf("Polar: Samples written: %u\n", MixBuffer->SampleCount);

                        //Update ringbuffer addresses
                        polar_ringbuffer_ReadFinish(CallbackBuffer);
                    }

                    //Start GUI frame
                    ImGui_ImplDX9_NewFrame();
                    ImGui_ImplWin32_NewFrame();
                    ImGui::NewFrame();

                    // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
                    if (show_demo_window)
                        ImGui::ShowDemoWindow(&show_demo_window);
            
                    // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
                    {
                        static float f = 0.0f;
                        static int counter = 0;
            
                        ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.
            
                        ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
                        ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
                        ImGui::Checkbox("Another Window", &show_another_window);
            
                        ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
                        ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color
            
                        if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
                            counter++;
                        ImGui::SameLine();
                        ImGui::Text("counter = %d", counter);
            
                        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
                        ImGui::End();
                    }

                    // 3. Show another simple window.
                    if (show_another_window)
                    {
                        ImGui::Begin("Another Window", &show_another_window);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
                        ImGui::Text("Hello from another window!");
                        if (ImGui::Button("Close Me"))
                            show_another_window = false;
                        ImGui::End();
                    }

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

            ImGui_ImplDX9_Shutdown();
            ImGui_ImplWin32_Shutdown();
            ImGui::DestroyContext();

            polar_ringbuffer_Destroy(EngineArena, CallbackBuffer);
            win32_WASAPI_Destroy(EngineArena, WASAPI);
        }

        else
        {
        }

        CleanupDeviceD3D();
        DestroyWindow(WindowHandle);
        UnregisterClass(WindowClass.lpszClassName, WindowClass.hInstance);
    }

    else
    {
    }

    memory_arena_Destroy(EngineArena);
    memory_arena_Destroy(SourceArena);

    return 0;
}
