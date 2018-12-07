#ifndef polar_render_cpp
#define polar_render_cpp

#include "polar_render.h"

f32 polar_render_GetPanPosition(i8 Position, f32 Amplitude, f32 PanFactor)
{
	f32 PanPosition; 

	//Left panning
	if(Position == 0)
	{
		PanPosition = Amplitude * sqrt(2.0) * (1 - PanFactor) / (2* sqrt(1 + PanFactor * PanFactor));
	}

	//Right panning
	if(Position == 1)
	{
		PanPosition = Amplitude * sqrt(2.0) * (1 + PanFactor) / (2* sqrt(1 + PanFactor * PanFactor));
	}

	return PanPosition;
}

void polar_render_FillBuffer(i8 ChannelCount, u32 FramesToWrite, f32 *SampleBuffer, BYTE *ByteBuffer, OSCILLATOR *Osc, f32 Amplitude, f32 PanValue)
{
	//? Check how exactly reinterpret_cast works
	SampleBuffer = reinterpret_cast<f32 *>(ByteBuffer);

	for(i32 FrameIndex = 0; FrameIndex < FramesToWrite; ++FrameIndex)
	{
		f32 CurrentSample = Osc->Tick(Osc);

		for(i8 ChannelIndex = 0; ChannelIndex < ChannelCount; ++ChannelIndex)
		{
			*SampleBuffer++ = CurrentSample * polar_render_GetPanPosition(ChannelIndex, Amplitude, PanValue);
		}
	}

    // char MetricsBuffer[256];
    // sprintf(MetricsBuffer, "Fill!\n");
    // OutputDebugString(MetricsBuffer);
}


void polar_UpdateRender(POLAR_DATA &Engine, OSCILLATOR *Osc, f32 Amplitude, f32 PanValue)
{
	polar_WASAPI_PrepareBuffer(Engine.WASAPI, Engine.Buffer);

	polar_render_FillBuffer(Engine.Channels, Engine.Buffer.FramesAvailable, Engine.Buffer.SampleBuffer, Engine.Buffer.ByteBuffer, Osc, Amplitude, PanValue);

	polar_WASAPI_ReleaseBuffer(Engine.WASAPI, Engine.Buffer);
}


#endif