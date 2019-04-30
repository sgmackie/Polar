
void SYS_NOISE_BROWN::Create(MEMORY_ARENA *Arena, size_t Size)
{
    SystemVoices = (ID_VOICE *) Arena->Alloc((sizeof(ID_VOICE) * Size), MEMORY_ARENA_ALIGNMENT);
    SystemCount = 0;
}

void SYS_NOISE_BROWN::Destroy(MEMORY_ARENA *Arena)
{
    Arena->FreeAll();
}

void SYS_NOISE_BROWN::Add(ID_VOICE ID)
{
    SystemVoices[SystemCount] = ID;
    ++SystemCount;
}

bool SYS_NOISE_BROWN::Remove(ID_VOICE ID)
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

void SYS_NOISE_BROWN::RenderToBuffer(CMP_NOISE &Noise, f64 Amplitude, CMP_BUFFER &Buffer, size_t BufferCount)
{
	if(!Buffer.Data)
	{
        Fatal("Sine: No buffer found!");
        return;
	}

    f64 Normalisation = 2.0;
	f64 Sample = 0;
	for(size_t i = 0; i < BufferCount; ++i)
	{
        Sample = RandomFloat64(-Amplitude, Amplitude);
        Noise.Accumulator += Sample;
        
        //Normalise to max amplitude
        if(Noise.Accumulator < -Normalisation || Noise.Accumulator > Normalisation)
        {
            Noise.Accumulator -= Sample;
        }

        Buffer.Data[i] = Noise.Accumulator;
	}    
}

void SYS_NOISE_BROWN::Update(ENTITY_VOICES *Voices, size_t BufferCount)
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
            CMP_FADE &Amplitude = Voices->Amplitudes[VoiceIndex];
            RenderToBuffer(Voices->Types[VoiceIndex].Noise, Amplitude.Current, Voices->Playbacks[VoiceIndex].Buffer, BufferCount);
        }
    }
}
