#ifndef polar_render_h
#define polar_render_h

//Struct to hold platform specific audio API important engine properties
typedef struct POLAR_DATA
{
	WASAPI_DATA *WASAPI;
	WASAPI_BUFFER Buffer;
	WASAPI_CLOCK Clock;
	i8 Channels;
	i32 SampleRate;
} POLAR_DATA;

void polar_render_FillBuffer(i8 ChannelCount, u32 FramesToWrite, BYTE *Data, OSCILLATOR *Osc, f32 Amplitude);
void polar_UpdateRender(WASAPI_DATA *WASAPI, WASAPI_BUFFER &Buffer, OSCILLATOR *Osc);

#endif