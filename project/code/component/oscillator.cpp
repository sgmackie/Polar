
void CMP_OSCILLATOR::Init(i32 Type, u32 SampleRate, f64 InputFrequency, f64 Limit)
{
    Flag |= Type;
    Phasor = 0;
    PhaseIncrement = 0;
    SizeOverSampleRate = Limit / SampleRate;
    Frequency = InputFrequency;
}