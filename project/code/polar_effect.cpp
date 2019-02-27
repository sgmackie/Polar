#ifndef polar_effect_cpp
#define polar_effect_cpp

void EffectAM(u32 FramesToProcess, u32 SampleRate, f32 *InputBuffer, f32 Frequency)
{
    f32 Index = 0;

    for(u32 FrameIndex = 0; FrameIndex < FramesToProcess; ++FrameIndex)
    {
        f32 Increment = (Frequency * TWO_PI32 * Index / SampleRate);
    	InputBuffer[FrameIndex] *= sinf(Increment);
        Index += 1;
    }
}

void EffectEcho(u32 FramesToProcess, u32 SampleRate, f32 *InputBuffer, u16 Duration)
{
    u32 Delay = (Duration * SampleRate);
    f32 DelaySamples[FramesToProcess];

    for(u32 FrameIndex = 0; FrameIndex < FramesToProcess; ++FrameIndex)
    {
        DelaySamples[FrameIndex] = InputBuffer[FrameIndex] + (FrameIndex >= Delay ? 0.5f * InputBuffer[(FrameIndex - Delay)] : 0.0f);
        InputBuffer[FrameIndex] = DelaySamples[FrameIndex];
    }
}



#endif