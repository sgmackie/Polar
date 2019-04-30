
void CMP_FORMAT::Init(u32 InputRate, u32 InputChannels)
{
    SampleRate  = 0;
    Channels    = 0;

    if(InputRate)
    {
        SampleRate = InputRate;
    }
    if(InputChannels)
    {
        Channels = InputChannels;
    }
}