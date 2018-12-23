//Polar
//TODO: Create entropy.lib for source code
#include "polar.h"
#include "polar_file.cpp"

//Windows includes
#include <Windows.h>
#include "win32_polar.h"

//Win32 globals
global bool GlobalRunning;
global bool GlobalPause;
global WIN32_OFFSCREEN_BUFFER GlobalDisplayBuffer;
global i32 MonitorRefreshRate = 60;
global i64 GlobalPerformanceCounterFrequency;

//!Test variables!
global WAVEFORM Waveform = SINE;

//WASAPI setup
//Create and initialise WASAPI struct
internal WASAPI_DATA *win32_WASAPI_Create(POLAR_BUFFER &Buffer, u32 UserSampleRate, u16 UserBitRate, u16 UserChannels)
{	
	WASAPI_DATA *WASAPI = wasapi_InterfaceCreate();
	wasapi_InterfaceInit(*WASAPI);

	if((WASAPI->DeviceReady = wasapi_DeviceInit(WASAPI->HR, *WASAPI, UserSampleRate, UserBitRate, UserChannels)) == false)
	{
		wasapi_DeviceDeInit(*WASAPI);
		wasapi_InterfaceDestroy(WASAPI);
		HR_TO_RETURN(WASAPI->HR, "Couldn't get WASAPI buffer for zero fill", nullptr);
	}

	Buffer.FramePadding = 0;
	Buffer.FramesAvailable = 0;
	Buffer.SampleBuffer = (f32 *) VirtualAlloc(0, ((sizeof *Buffer.SampleBuffer) * ((WASAPI->OutputBufferFrames * WASAPI->OutputWaveFormat->Format.nChannels))), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
	Buffer.DeviceBuffer = nullptr;

	// Initial zero fill	
	WASAPI->HR = WASAPI->AudioRenderClient->GetBuffer(WASAPI->OutputBufferFrames, (BYTE **) &Buffer.DeviceBuffer);
	HR_TO_RETURN(WASAPI->HR, "Couldn't get WASAPI buffer for zero fill", nullptr);

	WASAPI->HR = WASAPI->AudioRenderClient->ReleaseBuffer(WASAPI->OutputBufferFrames, AUDCLNT_BUFFERFLAGS_SILENT);
	HR_TO_RETURN(WASAPI->HR, "Couldn't release WASAPI buffer for zero fill", nullptr);

	return WASAPI;
}

//Remove WASAPI struct
internal void win32_WASAPI_Destroy(WASAPI_DATA *WASAPI)
{
	wasapi_DeviceDeInit(*WASAPI);
	wasapi_InterfaceDestroy(WASAPI);
}

//Get WASAPI buffer and release after filling with specified amount of samples
internal void win32_WASAPI_BufferGet(WASAPI_DATA *WASAPI, POLAR_BUFFER &Buffer)
{
	if(WASAPI->DeviceState == Playing)
	{
		WaitForSingleObject(WASAPI->RenderEvent, INFINITE);

		WASAPI->HR = WASAPI->AudioClient->GetCurrentPadding(&Buffer.FramePadding);
		HR_TO_RETURN(WASAPI->HR, "Couldn't get current padding", NONE);

		Buffer.FramesAvailable = WASAPI->OutputBufferFrames - Buffer.FramePadding;

		if(Buffer.FramesAvailable != 0)
		{
			WASAPI->HR = WASAPI->AudioRenderClient->GetBuffer(Buffer.FramesAvailable, (BYTE **) &Buffer.DeviceBuffer);
			HR_TO_RETURN(WASAPI->HR, "Couldn't get WASAPI buffer", NONE);
		}
	}
}

//Release byte buffer after the rendering loop
internal void win32_WASAPI_BufferRelease(WASAPI_DATA *WASAPI, POLAR_BUFFER &Buffer)
{
	if(WASAPI->DeviceState == Playing)
	{
		WASAPI->HR = WASAPI->AudioRenderClient->ReleaseBuffer(Buffer.FramesAvailable, 0);
		HR_TO_RETURN(WASAPI->HR, "Couldn't release WASAPI buffer", NONE);	
	}
}

//Windows file handling
//Find file name of current application
internal void win32_EXEFileNameGet(WIN32_STATE *State)
{
    //Gets full path of the current running process (0)
    GetModuleFileName(0, State->EXEPath, sizeof(State->EXEPath));
    State->EXEFileName = State->EXEPath;

    //Scan through the full path and remove until the final "\\"
    for(char *Scan = State->EXEPath; *Scan; ++Scan)
    {
        if(*Scan == '\\')
        {
            State->EXEFileName = Scan + 1;
        }
    }
}

//Get file path
internal void win32_BuildEXEPathGet(WIN32_STATE *State, char *FileName, char *Path)
{
    polar_StringConcatenate(State->EXEFileName - State->EXEPath, State->EXEPath, polar_StringLengthGet(FileName), FileName, Path);
}

//"Print" to a custom text file for looping edits
internal void win32_InputFilePathGet(WIN32_STATE *State, bool InputStream, i32 Index, char *Path)
{
    char Temp[64];
    wsprintf(Temp, "loop_point_%d_%s.pli", Index, InputStream ? "input" : "state");
    win32_BuildEXEPathGet(State, Temp, Path);
}

//Find the last time a file was written to
internal FILETIME win32_LastWriteTimeGet(char *Filename)
{
    FILETIME LastWriteTime = {};

    WIN32_FILE_ATTRIBUTE_DATA Data;
    if(GetFileAttributesEx(Filename, GetFileExInfoStandard, &Data))
    {
        LastWriteTime = Data.ftLastWriteTime;
    }

    return LastWriteTime;
}

//Load dll for dynamic render code
internal WIN32_ENGINE_CODE win32_EngineCodeLoad(char *SourceDLLName, char *TempDLLName)
{
    WIN32_ENGINE_CODE Result = {};
    Result.DLLLastWriteTime = win32_LastWriteTimeGet(SourceDLLName);

    CopyFile(SourceDLLName, TempDLLName, FALSE);
    Result.EngineDLL = LoadLibraryA(TempDLLName);
    
    if(Result.EngineDLL)
    {
        Result.UpdateAndRender = (polar_render_Update *) GetProcAddress(Result.EngineDLL, "RenderUpdate");
        Result.IsDLLValid = (Result.UpdateAndRender);
    }

    if(!Result.IsDLLValid)
    {
        Result.UpdateAndRender = 0;
    }

    return Result;
}

//Unload .dll
internal void win32_EngineCodeUnload(WIN32_ENGINE_CODE *EngineCode)
{
    if(EngineCode->EngineDLL)
    {
        FreeLibrary(EngineCode->EngineDLL);
        EngineCode->EngineDLL = 0;
    }

    EngineCode->IsDLLValid = false;
    EngineCode->UpdateAndRender = 0;
}

//State handling
//Get the current replay buffer
internal WIN32_REPLAY_BUFFER *win32_ReplayBufferGet(WIN32_STATE *State, u32 Index)
{
    Assert(Index < ArrayCount(State->ReplayBuffers));
    WIN32_REPLAY_BUFFER *Result = &State->ReplayBuffers[Index];
    return Result;
}

//Start recording the engine state
internal void win32_StateRecordingStart(WIN32_STATE *State, i32 InputRecordingIndex)
{
    WIN32_REPLAY_BUFFER *ReplayBuffer = win32_ReplayBufferGet(State, InputRecordingIndex);
    
    if(ReplayBuffer->MemoryBlock)
    {
        State->InputRecordingIndex = InputRecordingIndex;

        char FileName[WIN32_MAX_FILE_PATH];
        win32_InputFilePathGet(State, true, InputRecordingIndex, FileName);
        State->RecordingHandle = CreateFile(FileName, GENERIC_WRITE, 0, 0, CREATE_ALWAYS, 0, 0);
        
        CopyMemory(ReplayBuffer->MemoryBlock, State->EngineMemoryBlock, State->TotalSize);
    }
}

//Stop recording the engine state
internal void win32_StateRecordingStop(WIN32_STATE *State)
{
    CloseHandle(State->RecordingHandle);
    State->InputRecordingIndex = 0;
}

//Start playback of the recorded state
internal void win32_StatePlaybackStart(WIN32_STATE *State, i32 InputPlayingIndex)
{
    WIN32_REPLAY_BUFFER *ReplayBuffer = win32_ReplayBufferGet(State, InputPlayingIndex);
    if(ReplayBuffer->MemoryBlock)
    {
        State->InputPlayingIndex = InputPlayingIndex;

        char FileName[WIN32_MAX_FILE_PATH];
        win32_InputFilePathGet(State, true, InputPlayingIndex, FileName);
        State->PlaybackHandle = CreateFile(FileName, GENERIC_READ, 0, 0, OPEN_EXISTING, 0, 0);
        
        CopyMemory(State->EngineMemoryBlock, ReplayBuffer->MemoryBlock, State->TotalSize);
    }
}

//Stop playback of the recorded state
internal void win32_StatePlaybackStop(WIN32_STATE *State)
{
    CloseHandle(State->PlaybackHandle);
    State->InputPlayingIndex = 0;
}

//Input state handling
//Start recording input parameters
internal void win32_InputRecord(WIN32_STATE *State, POLAR_INPUT *NewInput)
{
    DWORD BytesWritten;
    WriteFile(State->RecordingHandle, NewInput, sizeof(*NewInput), &BytesWritten, 0);
}

//Start playback of recorded inputs
internal void win32_InputPlayback(WIN32_STATE *State, POLAR_INPUT *NewInput)
{
    DWORD BytesRead = 0;
    if(ReadFile(State->PlaybackHandle, NewInput, sizeof(*NewInput), &BytesRead, 0))
    {   
        //Hit end of recorded stream, loopback
        if(BytesRead == 0)
        {
            i32 PlayingIndex = State->InputPlayingIndex;
            win32_StateRecordingStop(State);
            win32_StatePlaybackStart(State, PlayingIndex);
            ReadFile(State->PlaybackHandle, NewInput, sizeof(*NewInput), &BytesRead, 0);
        }
    }
}

//Process inputs when released
internal void win32_InputMessageProcess(POLAR_INPUT_STATE *NewState, bool IsDown)
{
    if(NewState->EndedDown != IsDown)
    {
        NewState->EndedDown = IsDown;
        ++NewState->HalfTransitionCount;
    }
}

//Process the window message queue
internal void win32_WindowMessageProcess(WIN32_STATE *State, POLAR_INPUT_CONTROLLER *KeyboardController)
{
    MSG Message;
    while(PeekMessage(&Message, 0, 0, 0, PM_REMOVE))
    {
        switch(Message.message)
        {
            case WM_QUIT:
            {
                GlobalRunning = false;
                break;
            }
            
            case WM_SYSKEYDOWN:
            case WM_SYSKEYUP:
            case WM_KEYDOWN:
            case WM_KEYUP:
            {
                u32 VKCode = (uint32)Message.wParam;
                bool WasDown = ((Message.lParam & (1 << 30)) != 0);
                bool IsDown = ((Message.lParam & (1 << 31)) == 0);
                
                if(WasDown != IsDown)
                {
                    if(VKCode == 'W')
                    {
                        win32_InputMessageProcess(&KeyboardController->State.ButtonPress.MoveUp, IsDown);
                    }
                    else if(VKCode == 'A')
                    {
                        win32_InputMessageProcess(&KeyboardController->State.ButtonPress.MoveLeft, IsDown);
                    }
                    else if(VKCode == 'S')
                    {
                        win32_InputMessageProcess(&KeyboardController->State.ButtonPress.MoveDown, IsDown);
                    }
                    else if(VKCode == 'D')
                    {
                        win32_InputMessageProcess(&KeyboardController->State.ButtonPress.MoveRight, IsDown);
                    }
                    else if(VKCode == 'Q')
                    {
                        win32_InputMessageProcess(&KeyboardController->State.ButtonPress.LeftShoulder, IsDown);
                    }
                    else if(VKCode == 'E')
                    {
                        win32_InputMessageProcess(&KeyboardController->State.ButtonPress.RightShoulder, IsDown);
                    }
                    else if(VKCode == VK_UP)
                    {
                        win32_InputMessageProcess(&KeyboardController->State.ButtonPress.ActionUp, IsDown);
                    }
                    else if(VKCode == VK_LEFT)
                    {
                        win32_InputMessageProcess(&KeyboardController->State.ButtonPress.ActionLeft, IsDown);
                    }
                    else if(VKCode == VK_DOWN)
                    {
                        win32_InputMessageProcess(&KeyboardController->State.ButtonPress.ActionDown, IsDown);
                    }
                    else if(VKCode == VK_RIGHT)
                    {
                        win32_InputMessageProcess(&KeyboardController->State.ButtonPress.ActionRight, IsDown);
                    }
                    else if(VKCode == VK_ESCAPE)
                    {
                        win32_InputMessageProcess(&KeyboardController->State.ButtonPress.Start, IsDown);
                    }
                    else if(VKCode == VK_SPACE)
                    {
                        win32_InputMessageProcess(&KeyboardController->State.ButtonPress.Back, IsDown);
                    }
                    else if(VKCode == 'P')
                    {
                        if(IsDown)
                        {
                            GlobalPause = !GlobalPause;
                        }
                    }
                    else if(VKCode == 'L')
                    {
                        if(IsDown)
                        {
                            if(State->InputPlayingIndex == 0)
                            {
                                if(State->InputRecordingIndex == 0)
                                {
                                    win32_StateRecordingStart(State, 1);
                                }
                                else
                                {
                                    win32_StateRecordingStop(State);
                                    win32_StatePlaybackStart(State, 1);
                                }
                            }
                            else
                            {
                                win32_StatePlaybackStop(State);
                            }
                        }
                    }
                }

                bool32 AltKeyWasDown = (Message.lParam & (1 << 29));
                if((VKCode == VK_F4) && AltKeyWasDown)
                {
                    GlobalRunning = false;
                    break;
                }
            }

            default:
            {
                TranslateMessage(&Message);
                DispatchMessageA(&Message);
                break;
            }
        }
    }
}

//Display rendering
//Get the current window dimensions
internal WIN32_WINDOW_DIMENSIONS win32_WindowDimensionsGet(HWND Window)
{
    WIN32_WINDOW_DIMENSIONS Result;

    RECT ClientRect; 					//Rect structure for window dimensions
    GetClientRect(Window, &ClientRect); //Function to get current window dimensions in a RECT format
    
    Result.Width = ClientRect.right - ClientRect.left;
    Result.Height = ClientRect.bottom - ClientRect.top;

    return Result;
}

//Resize input buffer to a specific width and height
internal void win32_BitmapBufferResize(WIN32_OFFSCREEN_BUFFER *Buffer, i32 TargetWidth, i32 TargetHeight)
{
    if(Buffer->Data)
    {
        VirtualFree(Buffer->Data, 0, MEM_RELEASE);
    }

    Buffer->Width = TargetWidth;
    Buffer->Height = TargetHeight;
    Buffer->BytesPerPixel = 4;

    //Negative height field means the bitmap is top-down, not bottom-up, so first 3 bytes of the bitmap are the RGB values for the top left pixel
    Buffer->BitmapInfo.bmiHeader.biSize = sizeof(Buffer->BitmapInfo.bmiHeader);
    Buffer->BitmapInfo.bmiHeader.biWidth = Buffer->Width;
    Buffer->BitmapInfo.bmiHeader.biHeight = -Buffer->Height;
    Buffer->BitmapInfo.bmiHeader.biPlanes = 1;
    Buffer->BitmapInfo.bmiHeader.biBitCount = 32;
    Buffer->BitmapInfo.bmiHeader.biCompression = BI_RGB;

    Buffer->Pitch = Align16(Buffer->Width * Buffer->BytesPerPixel);
    i32 BitmapDataSize = (Buffer->Pitch * Buffer->Height);
    Buffer->Data = VirtualAlloc(0, BitmapDataSize, MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE);
}

//Copy buffer to a specified device
internal void win32_DisplayBufferInWindow(WIN32_OFFSCREEN_BUFFER *Buffer, HDC DeviceContext)
{
    StretchDIBits(DeviceContext, 0, 0, Buffer->Width, Buffer->Height, 0, 0, Buffer->Width, Buffer->Height, Buffer->Data, &Buffer->BitmapInfo, DIB_RGB_COLORS, SRCCOPY);
}

//Timing code
//Get the current position of the perfomance counter (https://msdn.microsoft.com/en-us/library/windows/desktop/ms644904(v=vs.85).aspx)
internal LARGE_INTEGER win32_WallClockGet()
{    
    LARGE_INTEGER Result;
    QueryPerformanceCounter(&Result);
    return Result;
}

//Determine the amount of seconfs elapised against the perfomance counter
internal f32 win32_SecondsElapsedGet(LARGE_INTEGER Start, LARGE_INTEGER End)
{
    f32 Result = ((f32) (End.QuadPart - Start.QuadPart) / (f32) GlobalPerformanceCounterFrequency);
    return Result ;
}

//Callback and main window
//Windows callback for message processing
LRESULT CALLBACK win32_MainCallback(HWND Window, UINT UserMessage, WPARAM WParam, LPARAM LParam)
{
    LRESULT Result = 0;

    switch(UserMessage)
    {
        case WM_SIZE:
        {   
            break;
        }

        case WM_DESTROY:
        {
            GlobalRunning = false;
            break;
        }

        case WM_CLOSE:
        {
            GlobalRunning = false;
            break;
        }

        case WM_ACTIVATEAPP:
        {
            break;
        }

        case WM_SYSKEYDOWN:
        case WM_SYSKEYUP:
        case WM_KEYDOWN:
        case WM_KEYUP:
        {
            OutputDebugString("Polar: Input came from an unrecognised source!\n");
            break;
        }

        case WM_PAINT: 
        {
            PAINTSTRUCT Paint;
            HDC PaintDevice = BeginPaint(Window, &Paint);
            WIN32_WINDOW_DIMENSIONS WindowDimensions = win32_WindowDimensionsGet(Window);
            //TODO: Visualise WASAPI buffer fills
            win32_DisplayBufferInWindow(&GlobalDisplayBuffer, PaintDevice);
            EndPaint(Window, &Paint);
            break;
        }

        default:
        {
            Result = DefWindowProc(Window, UserMessage, WParam, LParam); //Default window procedure
            break;
        }
    }

    return Result;
}  

//Windows entry point
int CALLBACK WinMain(HINSTANCE Instance, HINSTANCE PrevInstance, LPSTR CommandLine, int ShowCode)
{
    //Get .exe and .dll paths
    WIN32_STATE WindowState = {};
    win32_EXEFileNameGet(&WindowState);
    win32_BuildEXEPathGet(&WindowState, "polar.dll", WindowState.EngineSourceCodePath);
    win32_BuildEXEPathGet(&WindowState, "polar_temp.dll", WindowState.TempEngineSourceCodePath);

    //Start timings
    LARGE_INTEGER PerformanceCounterFrequencyResult;
    QueryPerformanceFrequency(&PerformanceCounterFrequencyResult);
    GlobalPerformanceCounterFrequency = PerformanceCounterFrequencyResult.QuadPart;

    //Request a 1ms period for timing functions
    UINT SchedulerPeriodInMS = 1;
    bool IsSleepGranular = (timeBeginPeriod(SchedulerPeriodInMS) == TIMERR_NOERROR);

    //Create initial display buffer
    win32_BitmapBufferResize(&GlobalDisplayBuffer, 1280, 720);

    //Set window properties
    WNDCLASS WindowClass = {};
    WindowClass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;                 //Set window redraw properties
    WindowClass.lpfnWndProc = win32_MainCallback;                           //Call the window process
    WindowClass.hInstance = Instance;                                       //Handle instance passed from win32_WinMain
    WindowClass.lpszClassName = "PolarWindowClass";                         //Name of Window class
    WindowClass.hCursor = LoadCursor(0, IDC_ARROW);                         //Load application cursor
    WindowClass.hbrBackground = (HBRUSH) GetStockObject(BLACK_BRUSH);       //Fill window background to black
    PrevInstance = 0;                                                       //Handle to previous instance of the window
    CommandLine = GetCommandLine();                                         //Get command line string for application
    ShowCode = SW_SHOW;                                                     //Activate and show window

	if(RegisterClass(&WindowClass))
    {
        HWND Window = CreateWindowEx(0, WindowClass.lpszClassName, "Polar", WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, 0, 0, Instance, 0); 

        if(Window) //Process message queue
        {
            //Create window device context
            HDC RendererDC = GetDC(Window);

            //Get the monitor refresh rate
            i32 win32_RefreshRate = GetDeviceCaps(RendererDC, VREFRESH);
            
            if(win32_RefreshRate > 1)
            {
                MonitorRefreshRate = win32_RefreshRate;
            }

            //Set target update FPS
            f32 EngineUpdateRate = (MonitorRefreshRate / 2.0f);
            f32 TargetSecondsPerFrame = 1.0f / (f32) EngineUpdateRate;

            //Initialise Polar
            POLAR_DATA PolarEngine = {};
            //TODO: User define properties in SHARED_MODE dont work, IsFormatSupported returns null Adjusted WAVEFORMATEX
            WASAPI_DATA *WASAPI =       win32_WASAPI_Create(PolarEngine.Buffer, 0, 0, 0);
            PolarEngine.BufferFrames =  WASAPI->OutputBufferFrames;
            PolarEngine.Channels =      WASAPI->OutputWaveFormat->Format.nChannels;
            PolarEngine.SampleRate =    WASAPI->OutputWaveFormat->Format.nSamplesPerSec;
            PolarEngine.BitRate =       WASAPI->OutputWaveFormat->Format.wBitsPerSample;

            //Start infinite loop
            GlobalRunning = true;

            //Create rendering output file
            //TODO: File writing is broken! Need to be external functions
            POLAR_WAV *OutputRenderFile = polar_render_WAVWriteCreate("Polar_Output.wav", &PolarEngine);

            //!Test source
            OSCILLATOR *Osc = entropy_wave_OscillatorCreate(PolarEngine.SampleRate, Waveform, 0);

            //Allocate engine memory block
            POLAR_MEMORY EngineMemory = {};
            EngineMemory.PermanentDataSize = Megabytes(64);
            EngineMemory.TemporaryDataSize = Megabytes(32);

            WindowState.TotalSize = EngineMemory.PermanentDataSize + EngineMemory.TemporaryDataSize;
            WindowState.EngineMemoryBlock = VirtualAlloc(0, ((size_t) WindowState.TotalSize), MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE);

            EngineMemory.PermanentData = WindowState.EngineMemoryBlock;
            EngineMemory.TemporaryData = ((uint8 *) EngineMemory.PermanentData + EngineMemory.PermanentDataSize);

            //Set file and memory map for state recording
            for(i32 ReplayIndex = 0; ReplayIndex < ArrayCount(WindowState.ReplayBuffers); ++ReplayIndex)
            {
                WIN32_REPLAY_BUFFER *CurrentReplayBuffer = &WindowState.ReplayBuffers[ReplayIndex];

                win32_InputFilePathGet(&WindowState, false, ReplayIndex, CurrentReplayBuffer->FileName);
                CurrentReplayBuffer->File = CreateFile(CurrentReplayBuffer->FileName, GENERIC_WRITE|GENERIC_READ, 0, 0, CREATE_ALWAYS, 0, 0);   //Generic file creation

                LARGE_INTEGER MaxSize;
                MaxSize.QuadPart = WindowState.TotalSize;
                CurrentReplayBuffer->MemoryMap = CreateFileMapping(CurrentReplayBuffer->File, 0, PAGE_READWRITE, MaxSize.HighPart, MaxSize.LowPart, 0);

                CurrentReplayBuffer->MemoryBlock = MapViewOfFile(CurrentReplayBuffer->MemoryMap, FILE_MAP_ALL_ACCESS, 0, 0, WindowState.TotalSize);
            }

            if(EngineMemory.PermanentData && EngineMemory.TemporaryData)
            {
                WIN32_ENGINE_CODE PolarState = win32_EngineCodeLoad(WindowState.EngineSourceCodePath, WindowState.TempEngineSourceCodePath);
                u32 LoadCounter = 0;

                POLAR_INPUT Input[2] = {};
                POLAR_INPUT *NewInput = &Input[0];
                POLAR_INPUT *OldInput = &Input[1];

                //Start timings
                LARGE_INTEGER LastCounter = win32_WallClockGet();
                LARGE_INTEGER FlipWallClock = win32_WallClockGet();

                u64 LastCycleCount = __rdtsc();

                while(GlobalRunning)
                {
                    FILETIME DLLWriteTime = win32_LastWriteTimeGet(WindowState.EngineSourceCodePath);
                    if(CompareFileTime(&DLLWriteTime, &PolarState.DLLLastWriteTime) != 0)
                    {
                            win32_EngineCodeUnload(&PolarState);
                            PolarState = win32_EngineCodeLoad(WindowState.EngineSourceCodePath, WindowState.TempEngineSourceCodePath);
                            LoadCounter = 0;
                    }         

                    POLAR_INPUT_CONTROLLER *OldKeyboardController = ControllerGet(OldInput, 0);
                    POLAR_INPUT_CONTROLLER *NewKeyboardController = ControllerGet(NewInput, 0);
                    *NewKeyboardController = {};
                    NewKeyboardController->IsConnected = true;
                    
                    for(i32 ButtonIndex = 0; ButtonIndex < ArrayCount(NewKeyboardController->State.Buttons); ++ButtonIndex)
                    {
                        NewKeyboardController->State.Buttons[ButtonIndex].EndedDown = OldKeyboardController->State.Buttons[ButtonIndex].EndedDown;
                    }

                    win32_WindowMessageProcess(&WindowState, NewKeyboardController);

                    if(!GlobalPause)
                    {
                        //Mouse message processing
                        POINT MousePointer;
                        GetCursorPos(&MousePointer);
                        ScreenToClient(Window, &MousePointer);
                        NewInput->MouseX = MousePointer.x;
                        NewInput->MouseY = MousePointer.y;
                        NewInput->MouseZ = 0; //Scroll wheel

                        win32_InputMessageProcess(&NewInput->MouseButtons[0], GetKeyState(VK_LBUTTON) & (1 << 15));
                        win32_InputMessageProcess(&NewInput->MouseButtons[1], GetKeyState(VK_MBUTTON) & (1 << 15));
                        win32_InputMessageProcess(&NewInput->MouseButtons[2], GetKeyState(VK_RBUTTON) & (1 << 15));
                        win32_InputMessageProcess(&NewInput->MouseButtons[3], GetKeyState(VK_XBUTTON1) & (1 << 15));
                        win32_InputMessageProcess(&NewInput->MouseButtons[4], GetKeyState(VK_XBUTTON2) & (1 << 15));

                        //Copy display buffer to render to
                        WIN32_OFFSCREEN_BUFFER Buffer = {};
                        Buffer.Data = GlobalDisplayBuffer.Data;
                        Buffer.Width = GlobalDisplayBuffer.Width; 
                        Buffer.Height = GlobalDisplayBuffer.Height;
                        Buffer.Pitch = GlobalDisplayBuffer.Pitch;
                        Buffer.BytesPerPixel = GlobalDisplayBuffer.BytesPerPixel;

                        if(WindowState.InputRecordingIndex)
                        {
                            win32_InputRecord(&WindowState, NewInput);
                        }

                        if(WindowState.InputPlayingIndex)
                        {
                            win32_InputPlayback(&WindowState, NewInput);
                        }

                        //Extern rendering function
                        if(PolarState.UpdateAndRender)
                        {
                            //Get the BYTE buffer and number of samples to fill    
                            win32_WASAPI_BufferGet(WASAPI, PolarEngine.Buffer);
                    
                            //Update objects and fill the buffer
                            if(OutputRenderFile != nullptr)
                            {
                                PolarState.UpdateAndRender(PolarEngine, OutputRenderFile, Osc, &EngineMemory, NewInput);
                                OutputRenderFile->TotalSampleCount += polar_render_WAVWriteFloat(OutputRenderFile, (PolarEngine.Buffer.FramesAvailable * PolarEngine.Channels), OutputRenderFile->Data);
                            }
                            else
                            {
                                PolarState.UpdateAndRender(PolarEngine, nullptr, Osc, &EngineMemory, NewInput);
                            }

                            //Give the requested samples back to WASAPI
	                        win32_WASAPI_BufferRelease(WASAPI, PolarEngine.Buffer);
                        }

                        //Check rendering work elapsed and sleep if time remaining
                        LARGE_INTEGER WorkCounter = win32_WallClockGet();
                        f32 WorkSecondsElapsed = win32_SecondsElapsedGet(LastCounter, WorkCounter);
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
                        
                            f32 TestSecondsElapsedForFrame = win32_SecondsElapsedGet(LastCounter, win32_WallClockGet());
                        
                            if(TestSecondsElapsedForFrame < TargetSecondsPerFrame)
                            {
                                //!Missed sleep!
                                // char SleepTimer[256];
                                // sprintf_s(SleepTimer, "Polar: Missed sleep timer!\n");
                                // OutputDebugString(SleepTimer);
                            }
                        
                            while(SecondsElapsedForFrame < TargetSecondsPerFrame)
                            {                            
                                SecondsElapsedForFrame = win32_SecondsElapsedGet(LastCounter, win32_WallClockGet());
                            }
                        }

                        else
                        {
                            //!Missed frame rate!
                            char FPSTimer[256];
                            sprintf_s(FPSTimer, "Polar: Missed frame rate target!\n");
                            OutputDebugString(FPSTimer);
                        }

                        //Prepare timers before display buffer copy
                        LARGE_INTEGER EndCounter = win32_WallClockGet();
                        f32 MSPerFrame = 1000.0f * win32_SecondsElapsedGet(LastCounter, EndCounter);                    
                        LastCounter = EndCounter;

                        //Set window device context for upcoming display
                        WIN32_WINDOW_DIMENSIONS WindowDimensions = win32_WindowDimensionsGet(Window);
                        HDC DisplayDevice = GetDC(Window);
                        //TODO: Debug info display
                        win32_DisplayBufferInWindow(&GlobalDisplayBuffer, DisplayDevice);
                        ReleaseDC(Window, DisplayDevice);

                        //Reset input for next loop
                        POLAR_INPUT *Temp = NewInput;
                        NewInput = OldInput;
                        OldInput = Temp;

                        //End performance timings
                        FlipWallClock = win32_WallClockGet();
                        u64 EndCycleCount = __rdtsc();
                        u64 CyclesElapsed = EndCycleCount - LastCycleCount;
                        LastCycleCount = EndCycleCount;
#if WIN32_METRICS   
                        UINT64 PositionFrequency;
                        UINT64 PositionUnits;

                        WASAPI->AudioClock->GetFrequency(&PositionFrequency);
                        WASAPI->AudioClock->GetPosition(&PositionUnits, 0);

                        //Sample cursor
                        u64 Cursor = PolarEngine.SampleRate * PositionUnits / PositionFrequency;
                    
                        //TODO: Actually calculate this
                        f32 FPS = win32_SecondsElapsedGet(LastCounter, EndCounter);
                        f64 MegaHzCyclesPerFrame = ((f64) CyclesElapsed / (1000.0f * 1000.0f));

                        char MetricsBuffer[256];
                        sprintf_s(MetricsBuffer, "Polar: %.02f ms/frame\t %.02f FPS\t %.02f cycles(MHz)/frame\t %llu samples\n", MSPerFrame, FPS, MegaHzCyclesPerFrame, Cursor);
                        OutputDebugString(MetricsBuffer);
                    }
#endif	
                }
			}

#if WIN32_METRICS
            char MetricsBuffer[256];
            sprintf_s(MetricsBuffer, "Polar: %llu frames written to %s\n", OutputRenderFile->TotalSampleCount, OutputRenderFile->Path);
            OutputDebugString(MetricsBuffer);
#endif

            //TODO: File is written but header not fully finalised, won't show the total time in file explorer
            polar_render_WAVWriteDestroy(OutputRenderFile);
            entropy_wave_OscillatorDestroy(Osc);
            win32_WASAPI_Destroy(WASAPI);
		}
	}

	return 0;
}