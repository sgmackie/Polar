
void SYS_OSCILLATOR_TRIANGLE::Create(MEMORY_ARENA *Arena, size_t Size)
{
    SystemVoices = (ID_VOICE *) Arena->Alloc((sizeof(ID_VOICE) * Size), MEMORY_ARENA_ALIGNMENT);
    SystemCount = 0;
}

void SYS_OSCILLATOR_TRIANGLE::Destroy(MEMORY_ARENA *Arena)
{
    Arena->FreeAll();
}

void SYS_OSCILLATOR_TRIANGLE::Add(ID_VOICE ID)
{
    SystemVoices[SystemCount] = ID;
    ++SystemCount;
}

bool SYS_OSCILLATOR_TRIANGLE::Remove(ID_VOICE ID)
{
    for(size_t i = 0; i <= SystemCount; ++i)
    {
        if(SystemVoices[i] == ID)
        {
            SystemVoices[i] = 0;
            --SystemCount;
            return true;
        }
    }
    //!Log
    return false;
}

void SYS_OSCILLATOR_TRIANGLE::RenderToBuffer(CMP_OSCILLATOR &Oscillator, CMP_BUFFER &Buffer, size_t BufferCount)
{
	if(!Buffer.Data)
	{
        Fatal("Sine: No buffer found!");
        return;
	}

	f64 Sample = 0;
	for(size_t i = 0; i < BufferCount; ++i)
	{
        Sample = ((2.0 * (Oscillator.Phasor * (1.0 / TWO_PI32))) - 1.0);
        if(Sample < 0.0)
        {
            Sample = -Sample;
        }
        Sample = (2.0 * (Sample - 0.5));
		Oscillator.PhaseIncrement = Oscillator.SizeOverSampleRate * Oscillator.Frequency;
		Oscillator.Phasor += Oscillator.PhaseIncrement;
		polar_dsp_PhaseWrap(Oscillator.Phasor, TWO_PI32);
		
        Buffer.Data[i] = Sample;
	}    
}

void SYS_OSCILLATOR_TRIANGLE::Update(ENTITY_VOICES *Voices, size_t BufferCount)
{
    //Loop through every source that was added to the system
    for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
    {
        //Find active sources in the system
        ID_VOICE Voice = SystemVoices[SystemIndex];
        if(Voice != 0)
        {
            //Source is valid - get component
            size_t VoiceIndex = Voices->RetrieveIndex(Voice);
            RenderToBuffer(Voices->Types[VoiceIndex].Oscillator, Voices->Playbacks[VoiceIndex].Buffer, BufferCount);
        }
    }
}
