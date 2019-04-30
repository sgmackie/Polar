
void SYS_OSCILLATOR_SINE::Create(MEMORY_ARENA *Arena, size_t Size)
{
    SystemVoices = (ID_VOICE *) Arena->Alloc((sizeof(ID_VOICE) * Size), MEMORY_ARENA_ALIGNMENT);
    SystemCount = 0;
}

void SYS_OSCILLATOR_SINE::Destroy(MEMORY_ARENA *Arena)
{
    Arena->FreeAll();
}

void SYS_OSCILLATOR_SINE::Add(ID_VOICE ID)
{
    SystemVoices[SystemCount] = ID;
    ++SystemCount;
}

bool SYS_OSCILLATOR_SINE::Remove(ID_VOICE ID)
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

void SYS_OSCILLATOR_SINE::RenderToBufferWithModulation(CMP_OSCILLATOR &Oscillator, CMP_MODULATOR &Modulator, CMP_BUFFER &Buffer, size_t BufferCount)
{
	if(!Buffer.Data)
	{
        Fatal("Sine: No buffer found!");
        return;
	}

	f64 Modulation = 0;
	f64 Sample = 0;
	for (size_t i = 0; i < BufferCount; ++i)
	{
        //Modulator sample
		Modulation = sin(Modulator.Oscillator.Phasor);
		Modulator.Oscillator.PhaseIncrement = Modulator.Oscillator.SizeOverSampleRate * Modulator.Oscillator.Frequency;
		Modulator.Oscillator.Phasor += Modulator.Oscillator.PhaseIncrement;
		polar_dsp_PhaseWrap(Modulator.Oscillator.Phasor, TWO_PI32);

        //Output sample
		Sample = sin(Oscillator.Phasor);
		Oscillator.PhaseIncrement = Oscillator.SizeOverSampleRate * (Oscillator.Frequency * Modulation);
		Oscillator.Phasor += Oscillator.PhaseIncrement;
		polar_dsp_PhaseWrap(Oscillator.Phasor, TWO_PI32);
		
        Buffer.Data[i] = Sample;
	}
}

void SYS_OSCILLATOR_SINE::RenderToBuffer(CMP_OSCILLATOR &Oscillator, CMP_BUFFER &Buffer, size_t BufferCount)
{
	if(!Buffer.Data)
	{
        Fatal("Sine: No buffer found!");
        return;
	}

	f64 Sample = 0;
	for (size_t i = 0; i < BufferCount; ++i)
	{
		Sample = sin(Oscillator.Phasor);
		Oscillator.PhaseIncrement = Oscillator.SizeOverSampleRate * Oscillator.Frequency;
		Oscillator.Phasor += Oscillator.PhaseIncrement;
		polar_dsp_PhaseWrap(Oscillator.Phasor, TWO_PI32);
		
        Buffer.Data[i] = Sample;
	}    
}

void SYS_OSCILLATOR_SINE::Update(ENTITY_VOICES *Voices, size_t BufferCount)
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
