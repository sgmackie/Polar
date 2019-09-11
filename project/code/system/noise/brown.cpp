
void SYS_NOISE_BROWN::Create(MEMORY_ALLOCATOR *Allocator, size_t Size)
{
    SystemVoices = (ID_VOICE *) Allocator->Alloc((sizeof(ID_VOICE) * Size), HEAP_TAG_SYSTEM_NSE_BROWN);
    SystemCount = 0;
}

void SYS_NOISE_BROWN::Destroy(MEMORY_ALLOCATOR *Allocator)
{
    Allocator->Free(0, HEAP_TAG_SYSTEM_NSE_BROWN);
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

void SYS_NOISE_BROWN::RenderToBuffer(CMP_NOISE &Noise, RANDOM_PCG *RNG, f64 Amplitude, CMP_BUFFER &Buffer, size_t BufferCount)
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
        Sample = RandomF32Range(RNG, -Amplitude, Amplitude);
        Noise.Accumulator += Sample;
        
        //Normalise to max amplitude
        if(Noise.Accumulator < -Normalisation || Noise.Accumulator > Normalisation)
        {
            Noise.Accumulator -= Sample;
        }

        Buffer.Data[i] = Noise.Accumulator;
	}    
}

void SYS_NOISE_BROWN::Update(ENTITY_VOICES *Voices, RANDOM_PCG *RNG, size_t BufferCount)
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
            RenderToBuffer(Voices->Types[VoiceIndex].Noise, RNG, Amplitude.Current, Voices->Playbacks[VoiceIndex].Buffer, BufferCount);
        }
    }
}
