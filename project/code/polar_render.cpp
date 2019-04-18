
u32 polar_render_Callback(POLAR_ENGINE *Engine)
{
    i16 *ConvertedSamples = Engine->CallbackBuffer.Write();
    //Clear to 0 first
    memset(ConvertedSamples, 0, (sizeof(i16) * Engine->MixBuffer.Count));

    for(u32 SampleIndex = 0; SampleIndex < Engine->MixBuffer.Count; ++SampleIndex)
    {
        f32 FloatSample = Engine->MixBuffer.Data[SampleIndex];
        i16 IntSample = FloatToInt16(FloatSample);

        for(u8 ChannelIndex = 0; ChannelIndex < Engine->Format.Channels; ++ChannelIndex)
        {
            *ConvertedSamples++ = IntSample;
        }
    }

    return Engine->MixBuffer.Count;
}