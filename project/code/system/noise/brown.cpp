
void SYS_NOISE_BROWN::Create(MEMORY_ARENA *Arena, size_t Size)
{
    SystemSources = (ID_SOURCE *) Arena->Alloc((sizeof(ID_SOURCE) * Size), MEMORY_ARENA_ALIGNMENT);
    SystemCount = 0;
}

void SYS_NOISE_BROWN::Destroy(MEMORY_ARENA *Arena)
{
    Arena->FreeAll();
}

void SYS_NOISE_BROWN::Add(ID_SOURCE ID)
{
    SystemSources[SystemCount] = ID;
    ++SystemCount;
}

bool SYS_NOISE_BROWN::Remove(ID_SOURCE ID)
{
    for(size_t i = 0; i <= SystemCount; ++i)
    {
        if(SystemSources[i] == ID)
        {
            SystemSources[i] = 0;
            --SystemCount;
            return true;
        }
    }
    //!Log
    return false;
}

void SYS_NOISE_BROWN::RenderToBuffer(CMP_NOISE &Noise, f64 Amplitude, CMP_BUFFER &Buffer, size_t BufferCount)
{
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

void SYS_NOISE_BROWN::Update(ENTITY_SOURCES *Sources, size_t BufferCount)
{
    //Loop through every source that was added to the system
    for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
    {
        //Find active sources in the system
        ID_SOURCE Source = SystemSources[SystemIndex];
        if(Source != 0)
        {
            //Source is valid - get component
            size_t SourceIndex = Sources->RetrieveIndex(Source);
            CMP_FADE &Ampliutde = Sources->Amplitudes[SourceIndex];
            RenderToBuffer(Sources->Noises[SourceIndex], Ampliutde.Current, Sources->Playbacks[SourceIndex].Buffer, BufferCount);
        }
    }
}
