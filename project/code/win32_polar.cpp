//TODO: Get .wav playback working before copying over Polar_main and POLAR objects

//CRT
#include <stdlib.h>
#include <Windows.h>

//Type defines
#include "misc/includes/win32_types.h"

//Debug
#include "library/debug/debug_macros.h"

//Includes
//Libraries
#include "library/dsp/dsp_wave.h"

//Polar
#include "polar_platform.cpp"

//Current running state
global bool GlobalRunning = false;

//Struct to hold platform specific audio API important engine properties
typedef struct POLAR_DATA
{
	WASAPI_DATA *WASAPI;
	WASAPI_BUFFER Buffer;
	i8 Channels;
	i32 SampleRate;
} POLAR_DATA;

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
            u32 VKCode = WParam;

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
                }
                else if(VKCode == VK_DOWN)
                {
                    OutputDebugString("Down\n");
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
    WNDCLASS WindowClass = {};

    WindowClass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC; //Create unique device context for this window
    WindowClass.lpfnWndProc = win32_MainCallback; //Call the window process
    WindowClass.hInstance = Instance; //Handle instance passed from win32_WinMain
    WindowClass.lpszClassName = "PolarWindowClass"; //Name of Window class

	if(RegisterClass(&WindowClass))
    {
        HWND Window = CreateWindowEx(0, WindowClass.lpszClassName, "Polar", WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, 0, 0, Instance, 0); 

        if(Window) //Process message queue
        {
            //Specified CS_OWNDC so get one device context and use it forever
            HDC DeviceContext = GetDC(Window);

            POLAR_DATA PolarEngine = {};
            PolarEngine.WASAPI = polar_WASAPI_Create(PolarEngine.Buffer);
            PolarEngine.Channels = PolarEngine.WASAPI->OutputWaveFormat->Format.nChannels;
            PolarEngine.SampleRate = PolarEngine.WASAPI->OutputWaveFormat->Format.nSamplesPerSec;
            
            OSCILLATOR *Osc = dsp_wave_CreateOscillator();
            dsp_wave_InitOscillator(Osc, SINE, PolarEngine.SampleRate);
            Osc->FrequencyCurrent = 880;
            
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
                
                polar_WASAPI_Render(PolarEngine.WASAPI, PolarEngine.Buffer, Osc);
                ReleaseDC(Window, DeviceContext);
#if WIN32_METRICS
        		LARGE_INTEGER EndCounter;
        		QueryPerformanceCounter(&EndCounter);
                
        		u64 EndCycleCount = __rdtsc();

        		i64 CounterElapsed = EndCounter.QuadPart - LastCounter.QuadPart;
        		u64 CyclesElapsed = EndCycleCount - LastCycleCount;
        		f32 MSPerFrame = (f32) (((1000.0f * (f32) CounterElapsed) / (f32) PerformanceCounterFrequency));
        		f32 FramesPerSecond = (f32) PerformanceCounterFrequency / (f32) CounterElapsed;
        		f32 MegaHzCyclesPerFrame = (f32) (CyclesElapsed / (1000.0f * 1000.0f));

        		char MetricsBuffer[256];
        		sprintf(MetricsBuffer, "Polar: %0.2f ms/frame\t %0.2f FPS\t %0.2f cycles(MHz)/frame\n", MSPerFrame, FramesPerSecond, MegaHzCyclesPerFrame);
        		OutputDebugString(MetricsBuffer);

        		LastCounter = EndCounter;
        		LastCycleCount = EndCycleCount;	
#endif	
			}

		dsp_wave_DestroyOscillator(Osc);
		polar_WASAPI_Destroy(PolarEngine.WASAPI);

		}
	}


	return 0;
}