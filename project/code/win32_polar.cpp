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
#include "polar_platform.cpp"
#include "polar_render.cpp"

//Current running state
global bool GlobalRunning = false;

//!Test variables!
global WAVEFORM Waveform = SINE;
global f32 Frequency = 440;
global f32 Amplitude = 0.35f;
global f32 Pan = 1;

//Find file name of Polar application
void win32_EXEFileNameGet(WIN32_STATE *State)
{
    //Gets full path of the current running process (0)
    GetModuleFileName(0, State->EXEPathFull, sizeof(State->EXEPathFull));
    State->EXEPath = State->EXEPathFull;

    //Scan through the full path and remove until the final "\\"
    for(char *Scan = State->EXEPathFull; *Scan; ++Scan)
    {
        if(*Scan == '\\')
        {
            State->EXEPath = Scan + 1;
        }
    }
}

WIN32_WINDOW_DIMENSIONS win32_WindowDimensionsGet(HWND Window)
{
    WIN32_WINDOW_DIMENSIONS WindowDimensions;

    RECT ClientRect; 					//Rect structure for window dimensions
    GetClientRect(Window, &ClientRect); //Function to get current window dimensions in a RECT format
    
    WindowDimensions.Width = ClientRect.right - ClientRect.left;
    WindowDimensions.Height = ClientRect.bottom - ClientRect.top;

    return WindowDimensions;
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
            //Virtual Keycode, keys without direct ANSI mappings
            u32 VKCode = (u32) WParam;

            //Check if key was down, if bit == 30 then it's true, otherwise false
            bool WasDown = ((LParam & (1 << 30)) != 0);
            //Check if being held down
            bool IsDown = ((LParam & (1 << 31)) == 0);

            if(WasDown != IsDown)
            {
                if(VKCode == 'W')
                {
                    OutputDebugString("W\n");
                }
                else if(VKCode == 'A')
                {
                    OutputDebugString("A\n");
                }
                else if(VKCode == 'S')
                {
                    OutputDebugString("S\n");
                }
                else if(VKCode == 'D')
                {
                    OutputDebugString("D\n");
                }
                else if(VKCode == 'Q')
                {
                    OutputDebugString("Q\n");
                }
                else if(VKCode == 'E')
                {
                    OutputDebugString("E\n");
                }
                else if(VKCode == VK_UP)
                {
                    OutputDebugString("Up\n");
                    Amplitude = Amplitude + 0.1f;
                }
                else if(VKCode == VK_DOWN)
                {
                    OutputDebugString("Down\n");
                    Amplitude = Amplitude - 0.1f;
                }
                else if(VKCode == VK_LEFT)
                {
                    OutputDebugString("Left\n");
                }
                else if(VKCode == VK_RIGHT)
                {
                    OutputDebugString("Right\n");
                }
                else if(VKCode == VK_SPACE)
                {
                    OutputDebugString("Space\n");
                }
                else if(VKCode == VK_ESCAPE)
                {
                    OutputDebugString("Escape\n");
                }
            }

            bool AltKeyWasDown = ((LParam & (1 << 29)) != 0);

            if((VKCode == VK_F4) && AltKeyWasDown)
            {
                GlobalRunning = false;
            }
            
            break;
        }

        case WM_PAINT: 
        {
            //TODO: Visualise WASAPI buffer fills

            WIN32_WINDOW_DIMENSIONS WindowDimensions = win32_WindowDimensionsGet(Window);
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
    WIN32_STATE WindowState = {};
    win32_EXEFileNameGet(&WindowState);

    WNDCLASS WindowClass = {};

    WindowClass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC; //Create unique device context for this window
    WindowClass.lpfnWndProc = win32_MainCallback; //Call the window process
    WindowClass.hInstance = Instance; //Handle instance passed from win32_WinMain
    WindowClass.lpszClassName = "PolarWindowClass"; //Name of Window class

    PrevInstance = 0; //Handle to previous instance of the window
    CommandLine = GetCommandLine(); //Get command line string for application
    CommandShow = SW_SHOW; //Activate and show window

	if(RegisterClass(&WindowClass))
    {
        HWND Window = CreateWindowEx(0, WindowClass.lpszClassName, "Polar", WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, 0, 0, Instance, 0); 

        if(Window) //Process message queue
        {
            //Specified CS_OWNDC so get one device context and use it forever
            HDC DeviceContext = GetDC(Window);

            i32 Win32RefreshRate = GetDeviceCaps(DeviceContext, VREFRESH);
            i32 MonitorRefreshRate = 60;
            
            //If GetDeviceCaps fails, use default 60Hz 
            if(Win32RefreshRate > 1)
            {
                MonitorRefreshRate = Win32RefreshRate;
            }

            //? Needed?
            // f32 EngineUpdateRate = (MonitorRefreshRate / 2.0f);

            POLAR_DATA PolarEngine = {};
            //TODO: User define properties in SHARED_MODE dont work, IsFormatSupported returns null Adjusted WAVEFORMATEX
            PolarEngine.WASAPI = polar_WASAPI_Create(PolarEngine.Buffer, 0, 0, 0);
            PolarEngine.BufferFrames = PolarEngine.WASAPI->OutputBufferFrames;
            PolarEngine.Channels = PolarEngine.WASAPI->OutputWaveFormat->Format.nChannels;
            PolarEngine.SampleRate = PolarEngine.WASAPI->OutputWaveFormat->Format.nSamplesPerSec;
            PolarEngine.BitRate = PolarEngine.WASAPI->OutputWaveFormat->Format.wBitsPerSample;
            
            OSCILLATOR *Osc = entropy_wave_OscillatorCreate(PolarEngine.SampleRate, Waveform, Frequency);

            POLAR_WAV *OutputRenderFile = polar_render_WAVWriteCreate("Polar_Output.wav", &PolarEngine);
            
            GlobalRunning = true;
#if WIN32_METRICS
            LARGE_INTEGER PerformanceCounterFrequencyResult;
            QueryPerformanceFrequency(&PerformanceCounterFrequencyResult);
            i64 PerformanceCounterFrequency = PerformanceCounterFrequencyResult.QuadPart;

            LARGE_INTEGER LastCounter;
            QueryPerformanceCounter(&LastCounter);
            u64 LastCycleCount = __rdtsc();
#endif 
            while(GlobalRunning)
            {
                MSG Messages;

                //PeekMessage keeps processing message queue without blocking when there are no messages available
                while(PeekMessage(&Messages, 0, 0, 0, PM_REMOVE))
                {
                    if(Messages.message == WM_QUIT)
                    {
                        PolarEngine.WASAPI->DeviceState = Stopped;
                        GlobalRunning = false;
                    }

                    TranslateMessage(&Messages);
                    DispatchMessage(&Messages);
                }

                WIN32_WINDOW_DIMENSIONS WindowDimensions = win32_WindowDimensionsGet(Window);

                //TODO: To pass variables to change over time, HH025 Win32ProcessPendingMessages        
                polar_render_BufferCopy(PolarEngine, OutputRenderFile, Osc, Amplitude, Pan);

                ReleaseDC(Window, DeviceContext);
#if WIN32_METRICS                
                UINT64 PositionFrequency;
                UINT64 PositionUnits;

                PolarEngine.WASAPI->AudioClock->GetFrequency(&PositionFrequency);
                PolarEngine.WASAPI->AudioClock->GetPosition(&PositionUnits, 0);

                //Sample cursor
                u64 Cursor = PolarEngine.SampleRate * PositionUnits / PositionFrequency;

                LARGE_INTEGER EndCounter;
                QueryPerformanceCounter(&EndCounter);

                u64 EndCycleCount = __rdtsc();

                i64 CounterElapsed = EndCounter.QuadPart - LastCounter.QuadPart;
                u64 CyclesElapsed = EndCycleCount - LastCycleCount;
                f32 MSPerFrame = (f32) (((1000.0f * (f32) CounterElapsed) / (f32) PerformanceCounterFrequency));
                f32 FramesPerSecond = (f32) PerformanceCounterFrequency / (f32) CounterElapsed;
                f32 MegaHzCyclesPerFrame = (f32) (CyclesElapsed / (1000.0f * 1000.0f));

                char MetricsBuffer[256];
                sprintf_s(MetricsBuffer, "Polar: %0.2f ms/frame\t %0.2f FPS\t %0.2f cycles(MHz)/frame\t %llu samples\n", MSPerFrame, FramesPerSecond, MegaHzCyclesPerFrame, Cursor);
                OutputDebugString(MetricsBuffer);

                LastCounter = EndCounter;
                LastCycleCount = EndCycleCount;	
#endif	
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



