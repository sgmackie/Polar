#ifndef polar_render_cpp
#define polar_render_cpp

#include "polar_render.h"


void polar_render_FillBuffer(i8 ChannelCount, u32 FramesToWrite, BYTE *Data, OSCILLATOR *Osc, f32 Amplitude)
{
	f32 *FloatData = reinterpret_cast<f32 *>(Data);

	for(i32 FrameIndex = 0; FrameIndex < FramesToWrite; ++FrameIndex) 
	{
    	f32 CurrentSample = Osc->Tick(Osc) * Amplitude;

    	for(i32 ChannelIndex = 0; ChannelIndex < ChannelCount; ++ChannelIndex) 
		{
    	    *FloatData++ = CurrentSample;
    	}      
	}

    // char MetricsBuffer[256];
    // sprintf(MetricsBuffer, "Fill!\n");
    // OutputDebugString(MetricsBuffer);
}



void polar_UpdateRender(POLAR_DATA &Engine, OSCILLATOR *Osc, f32 Amplitude)
{
	polar_WASAPI_PrepareBuffer(Engine.WASAPI, Engine.Buffer);

	polar_render_FillBuffer(Engine.Channels, Engine.Buffer.FramesAvailable, Engine.Buffer.Data, Osc, Amplitude);

	polar_WASAPI_ReleaseBuffer(Engine.WASAPI, Engine.Buffer);
}


#endif