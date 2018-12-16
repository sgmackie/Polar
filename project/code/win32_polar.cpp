//CRT
#include <stdlib.h>
#include <Windows.h>

//Type defines
#include "misc/includes/typedefs.h"

//Debug
#include "library/debug/debug_macros.h"

//Includes
//Synthesis
#include "library/entropy/entropy.h"

//Polar
#include "polar.h"
#include "polar.cpp"
#include "polar_platform.cpp"
#include "polar_render.cpp"

//Win32 globals
global bool GlobalRunning = false;
global bool GlobalPause = false;
global WIN32_OFFSCREEN_BUFFER GlobalDisplayBuffer;
global i32 MonitorRefreshRate = 120;
global i64 GlobalPerformanceCounterFrequency;

//!Test variables!
global WAVEFORM Waveform = SINE;
global f32 Frequency = 880;
global f32 Amplitude = 0.35f;
global f32 Pan = 0;

//Find file name of Polar application
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

internal void win32_BuildEXEPathGet(WIN32_STATE *State, char *FileName, char *Path)
{
    polar_StringConcatenate(State->EXEFileName, (State->EXEFileName - State->EXEPath), FileName, polar_StringLengthGet(FileName), Path);
}

internal void win32_InputFilePathGet(WIN32_STATE *State, bool InputStream, i32 SlotIndex, char *Path)
{
    char Temp[64];
    wsprintf(Temp, "loop_edit_%d_%s.hmi", SlotIndex, InputStream ? "input" : "state");
    win32_BuildEXEPathGet(State, Temp, Path);
}


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

internal WIN32_ENGINE_CODE win32_EngineCodeLoad(char *SourceDLLName, char *TempDLLName)
{
    WIN32_ENGINE_CODE Result = {};
    Result.DLLLastWriteTime = win32_LastWriteTimeGet(SourceDLLName);

    CopyFile(SourceDLLName, TempDLLName, FALSE);
    Result.EngineDLL = LoadLibraryA(TempDLLName);
    
    // if(Result.EngineDLL)
    // {
    //     Result.UpdateAndRender = (game_update_and_render *) GetProcAddress(Result.EngineDLL, "GameUpdateAndRender");
    //     Result.IsDLLValid = (Result.UpdateAndRender);
    // }

    // if(!Result.IsDLLValid)
    // {
    //     Result.UpdateAndRender = 0;
    // }

    return Result;
}

internal void win32_EngineCodeUnload(WIN32_ENGINE_CODE *EngineCode)
{
    if(EngineCode->EngineDLL)
    {
        FreeLibrary(EngineCode->EngineDLL);
        EngineCode->EngineDLL = 0;
    }

    // EngineCode->IsDLLValid = false;
    // EngineCode->UpdateAndRender = 0;
}


internal WIN32_REPLAY_BUFFER *win32_ReplayBufferGet(WIN32_STATE *State, u32 Index)
{
    Assert(Index < ArrayCount(State->ReplayBuffers));
    WIN32_REPLAY_BUFFER *Result = &State->ReplayBuffers[Index];
    return Result;
}


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


internal void win32_StateRecordingStop(WIN32_STATE *State)
{
    CloseHandle(State->RecordingHandle);
    State->InputRecordingIndex = 0;
}


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

internal void win32_StatePlaybackStop(WIN32_STATE *State)
{
    CloseHandle(State->PlaybackHandle);
    State->InputPlayingIndex = 0;
}



internal void win32_InputRecord(WIN32_STATE *State, POLAR_INPUT *NewInput)
{
    DWORD BytesWritten;
    WriteFile(State->RecordingHandle, NewInput, sizeof(*NewInput), &BytesWritten, 0);
}

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

internal WIN32_WINDOW_DIMENSIONS win32_WindowDimensionsGet(HWND Window)
{
    WIN32_WINDOW_DIMENSIONS WindowDimensions;

    RECT ClientRect; 					//Rect structure for window dimensions
    GetClientRect(Window, &ClientRect); //Function to get current window dimensions in a RECT format
    
    WindowDimensions.Width = ClientRect.right - ClientRect.left;
    WindowDimensions.Height = ClientRect.bottom - ClientRect.top;

    return WindowDimensions;
}

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

    i32 BitmapDataSize = ((Buffer->Width * Buffer->Height) * Buffer->BytesPerPixel);
    Buffer->Data = VirtualAlloc(0, BitmapDataSize, MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE);
    Buffer->Pitch = Buffer->Width * Buffer->BytesPerPixel;
}


internal LARGE_INTEGER win32_WallClockGet()
{    
    LARGE_INTEGER Result;
    QueryPerformanceCounter(&Result);
    return Result;
}


internal f32 win32_SecondsElapsedGet(LARGE_INTEGER Start, LARGE_INTEGER End)
{
    f32 Result = ((f32)(End.QuadPart - Start.QuadPart) / (f32)GlobalPerformanceCounterFrequency);
    return Result ;
}


internal void win32_InputMessageProcess(POLAR_INPUT_STATE *NewState, bool IsDown)
{
    if(NewState->EndedDown != IsDown)
    {
        NewState->EndedDown = IsDown;
        ++NewState->HalfTransitionCount;
    }
}


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
                        win32_InputMessageProcess(&KeyboardController->MoveUp, IsDown);
                    }
                    else if(VKCode == 'A')
                    {
                        win32_InputMessageProcess(&KeyboardController->MoveLeft, IsDown);
                    }
                    else if(VKCode == 'S')
                    {
                        win32_InputMessageProcess(&KeyboardController->MoveDown, IsDown);
                    }
                    else if(VKCode == 'D')
                    {
                        win32_InputMessageProcess(&KeyboardController->MoveRight, IsDown);
                    }
                    else if(VKCode == 'Q')
                    {
                        win32_InputMessageProcess(&KeyboardController->LeftShoulder, IsDown);
                    }
                    else if(VKCode == 'E')
                    {
                        win32_InputMessageProcess(&KeyboardController->RightShoulder, IsDown);
                    }
                    else if(VKCode == VK_UP)
                    {
                        win32_InputMessageProcess(&KeyboardController->ActionUp, IsDown);
                    }
                    else if(VKCode == VK_LEFT)
                    {
                        win32_InputMessageProcess(&KeyboardController->ActionLeft, IsDown);
                    }
                    else if(VKCode == VK_DOWN)
                    {
                        win32_InputMessageProcess(&KeyboardController->ActionDown, IsDown);
                    }
                    else if(VKCode == VK_RIGHT)
                    {
                        win32_InputMessageProcess(&KeyboardController->ActionRight, IsDown);
                    }
                    else if(VKCode == VK_ESCAPE)
                    {
                        win32_InputMessageProcess(&KeyboardController->Start, IsDown);
                    }
                    else if(VKCode == VK_SPACE)
                    {
                        win32_InputMessageProcess(&KeyboardController->Back, IsDown);
                    }
#if POLAR_LOOP                    
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
#endif
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


//Windows callback for message processing
LRESULT CALLBACK win32_MainCallback(HWND Window, UINT UserMessage, WPARAM WParam, LPARAM LParam)
{
    LRESULT Result = 0;

    switch(UserMessage)
    {
        case WM_SIZE:
        {   
            OutputDebugString("WM_SIZE\n");
            break;
        }

        case WM_DESTROY:
        {
            GlobalRunning = false;
            OutputDebugString("WM_DESTROY\n");
            break;
        }

        case WM_CLOSE:
        {
            GlobalRunning = false;
            OutputDebugString("WM_CLOSE\n");
            break;
        }

        case WM_ACTIVATEAPP:
        {
            OutputDebugString("WM_ACTIVATEAPP\n");
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


int CALLBACK WinMain(HINSTANCE Instance, HINSTANCE PrevInstance, LPSTR CommandLine, int CommandShow)
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
    WindowClass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC; //Create unique device context for this window
    WindowClass.lpfnWndProc = win32_MainCallback;           //Call the window process
    WindowClass.hInstance = Instance;                       //Handle instance passed from win32_WinMain
    WindowClass.lpszClassName = "PolarWindowClass";         //Name of Window class
    PrevInstance = 0;                                       //Handle to previous instance of the window
    CommandLine = GetCommandLine();                         //Get command line string for application
    CommandShow = SW_SHOW;                                  //Activate and show window

	if(RegisterClass(&WindowClass))
    {
        HWND Window = CreateWindowEx(0, WindowClass.lpszClassName, "Polar", WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, 0, 0, Instance, 0); 

        if(Window) //Process message queue
        {
            //Get the monitor refresh rate
            HDC RefreshDevice = GetDC(Window);
            i32 win32_RefreshRate = GetDeviceCaps(RefreshDevice, VREFRESH);
            ReleaseDC(Window, RefreshDevice);
            
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
            PolarEngine.WASAPI = polar_WASAPI_Create(PolarEngine.Buffer, 0, 0, 0);
            PolarEngine.BufferFrames = PolarEngine.WASAPI->OutputBufferFrames;
            PolarEngine.Channels = PolarEngine.WASAPI->OutputWaveFormat->Format.nChannels;
            PolarEngine.SampleRate = PolarEngine.WASAPI->OutputWaveFormat->Format.nSamplesPerSec;
            PolarEngine.BitRate = PolarEngine.WASAPI->OutputWaveFormat->Format.wBitsPerSample;
            
            //Create rendering output file
            POLAR_WAV *OutputRenderFile = polar_render_WAVWriteCreate("Polar_Output.wav", &PolarEngine);

            //!Test source
            OSCILLATOR *Osc = entropy_wave_OscillatorCreate(PolarEngine.SampleRate, Waveform, Frequency);

            //Start infinite loop
            GlobalRunning = true;

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
                    
                for(int ButtonIndex = 0; ButtonIndex < ArrayCount(NewKeyboardController->Buttons); ++ButtonIndex)
                {
                    NewKeyboardController->Buttons[ButtonIndex].EndedDown = OldKeyboardController->Buttons[ButtonIndex].EndedDown;
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

                    // if(Game.UpdateAndRender)
                    // {
                        // Game.UpdateAndRender(&Thread, &GameMemory, NewInput, &Buffer);
                    // }

                    //TODO: To pass variables to change over time, HH025 win32_WindowMessageProcess        
                    polar_render_BufferCopy(PolarEngine, OutputRenderFile, Osc, Amplitude, Pan);


                    //Check rendering work elapsed and sleep if time remaining
                    LARGE_INTEGER WorkCounter = win32_WallClockGet();
                    f32 WorkSecondsElapsed = win32_SecondsElapsedGet(LastCounter, WorkCounter);
                    f32 SecondsElapsedForFrame = WorkSecondsElapsed;
                    
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
                        // char FPSTimer[256];
                        // sprintf_s(FPSTimer, "Polar: Missed frame rate timer!\n");
                        // OutputDebugString(FPSTimer);
                    }

                    //Prepare timers before display buffer copy
                    LARGE_INTEGER EndCounter = win32_WallClockGet();
                    f32 MSPerFrame = 1000.0f * win32_SecondsElapsedGet(LastCounter, EndCounter);                    
                    LastCounter = EndCounter;

                    //Set window device context for upcoming display
                    WIN32_WINDOW_DIMENSIONS WindowDimensions = win32_WindowDimensionsGet(Window);
                    HDC DeviceContext = GetDC(Window);
                    //TODO: Debug info display
                    ReleaseDC(Window, DeviceContext);

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

                    PolarEngine.WASAPI->AudioClock->GetFrequency(&PositionFrequency);
                    PolarEngine.WASAPI->AudioClock->GetPosition(&PositionUnits, 0);

                    //Sample cursor
                    u64 Cursor = PolarEngine.SampleRate * PositionUnits / PositionFrequency;
                    
                    //TODO: Actually calculate this
                    f64 FPS = 0.0f;
                    f64 MegaHzCyclesPerFrame = ((f64)CyclesElapsed / (1000.0f * 1000.0f));

                    char MetricsBuffer[256];
                    sprintf_s(MetricsBuffer, "Polar: %.02f ms/frame\t %.02f frames/sec\t %.02f cycles(MHz)/frame\t %llu samples\n", MSPerFrame, FPS, MegaHzCyclesPerFrame, Cursor);
                    OutputDebugString(MetricsBuffer);
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
            polar_WASAPI_Destroy(PolarEngine.WASAPI);
		}
	}

	return 0;
}