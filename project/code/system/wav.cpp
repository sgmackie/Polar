
void SYS_WAV::Create(MEMORY_ALLOCATOR *Allocator, size_t Size)
{
    SystemVoices = (ID_VOICE *) Allocator->Alloc((sizeof(ID_VOICE) * Size), HEAP_TAG_SYSTEM_WAV);
    SystemCount = 0;
}

void SYS_WAV::Destroy(MEMORY_ALLOCATOR *Allocator)
{
    Allocator->Free(0, HEAP_TAG_SYSTEM_WAV);
}

void SYS_WAV::Add(ID_VOICE ID)
{
    SystemVoices[SystemCount] = ID;
    ++SystemCount;
}

bool SYS_WAV::Remove(ID_VOICE ID)
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

void SYS_WAV::RenderToBuffer(CMP_WAV &WAV, CMP_BUFFER &Buffer, i32 Rate, size_t BufferCount)
{
    //Positions in the file
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
        f32 Sample      = 0;
        Sample          = WAV.Data[WAV.ReadIndex];
        Buffer.Data[i]  = Sample;
		WAV.ReadIndex   += WAV.Channels;
    }
}

void SYS_WAV::Update(ENTITY_VOICES *Voices, f64 Pitch, size_t BufferCount)
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
            RenderToBuffer(Voices->Types[VoiceIndex].WAV, Voices->Playbacks[VoiceIndex].Buffer, 1, BufferCount);
            
            //! WIP Pitch resampler
            // if(Voices->Flags[VoiceIndex] & ENTITY_VOICES::WAV)
            // {
            //     //Rate is File rate / Mixer rate * Pitch
            //     //!Truncated to int!
            //     //TODO: Add interpolation for .f values
            //     i32 Rate = 48000 / (f64) 48000 * Pitch;
            //     if(Rate <= 0)
            //     {
            //         Rate = 1;
            //     }
            // }
        }
    }
}