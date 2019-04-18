
void SYS_MIX::Create(MEMORY_ARENA *Arena, size_t Size)
{
    SystemSources = (ID_SOURCE *) Arena->Alloc((sizeof(ID_SOURCE) * Size), MEMORY_ARENA_ALIGNMENT);
    SystemCount = 0;
}

void SYS_MIX::Destroy(MEMORY_ARENA *Arena)
{
    Arena->FreeAll();
}

void SYS_MIX::Add(ID_SOURCE ID)
{
    SystemSources[SystemCount] = ID;
    ++SystemCount;
}

bool SYS_MIX::Remove(ID_SOURCE ID)
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

void SYS_MIX::RenderToBuffer(f32 *MixBuffer, size_t SamplesToWrite, u32 Channels, CMP_BUFFER &SourceBuffer, CMP_FADE &SourceAmplitude, f64 TargetAmplitude)
{
    //Check if amplitude is already at the target
    if(fabsl(SourceAmplitude.Previous - TargetAmplitude) < 0.1e-5)
    {
        for(u32 FrameIndex = 0; FrameIndex < SamplesToWrite; ++FrameIndex)
        {
            MixBuffer[FrameIndex] += SourceBuffer.Data[FrameIndex] * TargetAmplitude;
        }
    }

    //If not then increment to target
    else
    {
        f64 Current = SourceAmplitude.Previous;
        f64 Step = (TargetAmplitude - Current) * 1.0f / SamplesToWrite;
        for(u32 FrameIndex = 0; FrameIndex < SamplesToWrite; ++FrameIndex)
        {
            Current += Step;
            MixBuffer[FrameIndex] += SourceBuffer.Data[FrameIndex] * Current;
        }

        SourceAmplitude.Previous = TargetAmplitude;
    }
}   

void SYS_MIX::Update(ENTITY_SOURCES *Sources, f32 *MixBuffer, size_t SamplesToWrite)
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
            if(Sources->Flags[SourceIndex] & ENTITY_SOURCES::PLAYBACK)
            {
                TPL_PLAYBACK &Playback      = Sources->Playbacks[SourceIndex];
                
                switch(Playback.Duration.States)
                {
                    case PLAYING:
                    {
                        if(Sources->Flags[SourceIndex] & ENTITY_SOURCES::AMPLITUDE)
                        {
                            CMP_FADE &SourceAmplitude   = Sources->Amplitudes[SourceIndex];
                            RenderToBuffer(MixBuffer, SamplesToWrite, Playback.Format.Channels, Playback.Buffer, SourceAmplitude, SourceAmplitude.Current);
                        }
                        
                        break;
                    }
                    case STOPPED:
                    {
                        break;
                    }
                    case STOPPING:
                    {
                        if(Sources->Flags[SourceIndex] & ENTITY_SOURCES::AMPLITUDE)
                        {
                            CMP_FADE &SourceAmplitude   = Sources->Amplitudes[SourceIndex];
                            RenderToBuffer(MixBuffer, SamplesToWrite, Playback.Format.Channels, Playback.Buffer, SourceAmplitude, 0.0f);
                        }

                        break;
                    }                
                    case PAUSED:
                    {
                        break;
                    }
                    default:
                    {
                        break;
                    }
                }  
            }
        }
    }    
}