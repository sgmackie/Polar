
void SYS_NOISE_WHITE::Create(MEMORY_ALLOCATOR *Allocator, size_t Size)
{
    SystemVoices = (ID_VOICE *) Allocator->Alloc((sizeof(ID_VOICE) * Size), HEAP_TAG_SYSTEM_NSE_WHITE);
    SystemCount = 0;
}

void SYS_NOISE_WHITE::Destroy(MEMORY_ALLOCATOR *Allocator)
{
    Allocator->Free(0, HEAP_TAG_SYSTEM_NSE_WHITE);
}

void SYS_NOISE_WHITE::Add(ID_VOICE ID)
{
    SystemVoices[SystemCount] = ID;
    ++SystemCount;
}

bool SYS_NOISE_WHITE::Remove(ID_VOICE ID)
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

void SYS_NOISE_WHITE::RenderToBuffer(RANDOM_PCG *RNG, f64 Amplitude, CMP_BUFFER &Buffer, size_t BufferCount)
{
	if(!Buffer.Data)
	{
        Fatal("Sine: No buffer found!");
        return;
	}

	f64 Sample = 0;
	for(size_t i = 0; i < BufferCount; ++i)
	{
        Sample = RandomF32Range(RNG, -Amplitude, Amplitude);
        Buffer.Data[i] = Sample;
	}    
}

void SYS_NOISE_WHITE::Update(ENTITY_VOICES *Voices, RANDOM_PCG *RNG, size_t BufferCount)
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
            RenderToBuffer(RNG, Amplitude.Current, Voices->Playbacks[VoiceIndex].Buffer, BufferCount);
        }
    }
}
