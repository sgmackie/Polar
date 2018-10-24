#include "polar_WASAPI.h"

void polar_WASAPI_Start(bool PrintDevices)
{
    //Initialise COM instance
    if(FAILED(CoInitializeEx(NULL, COINIT_MULTITHREADED)))
    {
        debug_PrintLine(Console, "WASAPI: Failed to initalise COM object!");
    }
  
    if(PrintDevices)
    {
        //Display connected audio devices
        wasapi_device_PrintAudioDevices();
    }
}

void polar_WASAPI_Stop()
{
    //Uninitialise COM instance
    CoUninitialize();
}



RENDER_STREAM *polar_WASAPI_CreateStream(f64 Amplitude)
{
    IMMDevice *Device = wasapi_device_CreateRenderDevice();
    if(!Device)
    {
        debug_PrintLine(Console, "WASAPI: Failed to create rendering device!");
        return nullptr;
    }

    RENDER_STREAM *Stream = wasapi_render_CreateRenderStream(Device, Amplitude);
    if(!Stream)
    {
        debug_PrintLine(Console, "WASAPI: Failed to create rendering stream!");
        return nullptr;
    }

    return Stream;
}

void polar_WASAPI_DestroyStream(RENDER_STREAM *Stream)
{
    if(Stream)
    {
        wasapi_render_DestroyRenderStream(Stream);
    }
}




// RENDER_THREAD *polar_WASAPI_CreateThread(RENDER_STREAM &OutputStream, OSCILLATOR &SineOsc)
// {
//     RENDER_THREAD *OutputThread = wasapi_render_CreateRenderThread(OutputStream, SineOsc);
    
//     return OutputThread;
// }

