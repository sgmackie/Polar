
void SYS_WAV::Create(MEMORY_ARENA *Arena, size_t Size)
{
    SystemSources = (ID_SOURCE *) Arena->Alloc((sizeof(ID_SOURCE) * Size), MEMORY_ARENA_ALIGNMENT);
    SystemCount = 0;
}

void SYS_WAV::Destroy(MEMORY_ARENA *Arena)
{
    Arena->FreeAll();
}

void SYS_WAV::Add(ID_SOURCE ID)
{
    SystemSources[SystemCount] = ID;
    ++SystemCount;
}

bool SYS_WAV::Remove(ID_SOURCE ID)
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

void SYS_WAV::RenderToBuffer(CMP_WAV &WAV, CMP_BUFFER &Buffer, i32 Rate, size_t BufferCount)
{
    //Positions in the file
    u64 Position = 0;
    u64 SamplesRemaining = (WAV.Length - WAV.ReadIndex);

    //Adjust buffer size to the remaining samples
    if(SamplesRemaining < BufferCount)
    {
        BufferCount = SamplesRemaining;
    }

    //Reached end of the file
    //!Looping for now
    if(WAV.ReadIndex == WAV.Length)
    {
        WAV.ReadIndex = 0;
    }

    for(size_t i = 0; i < BufferCount; ++i)
    {
        f64 Sample = 0;
        Sample = WAV.Data[Position];
        Buffer.Data[i] = Sample;
        
		Position = ++WAV.ReadIndex * WAV.Channels;
		//Position = (WAV.ReadIndex += i * Rate);
    }
}

void SYS_WAV::Update(ENTITY_SOURCES *Sources, f64 Pitch, size_t BufferCount)
{
    //Loop through every source that was added to the system
    for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
    {
        //Find active sources in the system
        ID_SOURCE Source = SystemSources[SystemIndex];
        if(Source != 0)
        {
            //Source is valid - check for component
            size_t SourceIndex = Sources->RetrieveIndex(Source);
            if(Sources->Flags[SourceIndex] & ENTITY_SOURCES::WAV)
            {
                //Rate is File rate / Mixer rate * Pitch
                //!Truncated to int!
                //TODO: Add interpolation for .f values
                i32 Rate = 48000 / (f64) 48000 * Pitch;
                if(Rate <= 0)
                {
                    Rate = 1;
                }

                // RenderToBuffer(Sources->WAVs[SourceIndex], Sources->Playbacks[SourceIndex].Buffer, Rate, BufferCount);
            }
        }
    }
}