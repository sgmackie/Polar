
void SYS_OSCILLATOR_NOISE::Create(MEMORY_ARENA *Arena, size_t Size)
{
    SystemSources = (ID_SOURCE *) Arena->Alloc((sizeof(ID_SOURCE) * Size), MEMORY_ARENA_ALIGNMENT);
    SystemCount = 0;
}

void SYS_OSCILLATOR_NOISE::Destroy(MEMORY_ARENA *Arena)
{
    Arena->FreeAll();
}

void SYS_OSCILLATOR_NOISE::Add(ID_SOURCE ID)
{
    SystemSources[SystemCount] = ID;
    ++SystemCount;
}

bool SYS_OSCILLATOR_NOISE::Remove(ID_SOURCE ID)
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

void SYS_OSCILLATOR_NOISE::RenderToBuffer(CMP_OSCILLATOR &Oscillator, CMP_BUFFER &Buffer, size_t BufferCount)
{
    f64 Sample = 0;
    for(size_t i = 0; i < BufferCount; ++i)
    {
        Sample = RandomFloat64(-1.0, 1.0);
        Buffer.Data[i] = Sample;
    }
}

void SYS_OSCILLATOR_NOISE::Update(ENTITY_SOURCES *Sources, size_t BufferCount)
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
            RenderToBuffer(Sources->Oscillators[SourceIndex], Sources->Playbacks[SourceIndex].Buffer, BufferCount);
        }
    }
}
