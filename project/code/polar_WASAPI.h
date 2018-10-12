#ifndef polar_WASAPI_h
#define polar_WASAPI_h

void polar_WASAPI_Start(bool PrintDevices);

void polar_WASAPI_Stop();

RENDER_STREAM *polar_WASAPI_CreateStream();

void polar_WASAPI_DestroyStream(RENDER_STREAM *Stream);



// RENDER_THREAD *polar_WASAPI_CreateThread(RENDER_STREAM &OutputStream, OSCILLATOR &SineOsc)
// {
//     RENDER_THREAD *OutputThread = wasapi_render_CreateRenderThread(OutputStream, SineOsc);
    
//     return OutputThread;
// }


#endif