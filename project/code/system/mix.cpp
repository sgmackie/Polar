
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

void SYS_MIX::RenderToBuffer(f32 *Channel0, f32 *Channel1, size_t SamplesToWrite, CMP_BUFFER &SourceBuffer, CMP_FADE &SourceAmplitude, CMP_PAN &SourcePan, f64 TargetAmplitude)
{
    //Check if amplitude is already at the target
    if(fabsl(SourceAmplitude.Previous - TargetAmplitude) < 0.1e-5)
    {
        f64 LeftAmplitude   = 0;
        f64 RightAmplitude  = 0;

        switch(SourcePan.Flag)
        {
            case CMP_PAN::MODE::MONO:
            {
                LeftAmplitude   = TargetAmplitude * 0.707;
                RightAmplitude  = TargetAmplitude * 0.707;            
                break;
            }
            case CMP_PAN::MODE::STEREO:
            {
                //Calculate constant power values
                f64 Angle = ((1.0 - SourcePan.Amplitude) * PI32 * 0.25);
                LeftAmplitude   = TargetAmplitude * sin(Angle);
                RightAmplitude  = TargetAmplitude * cos(Angle);          
                break;
            }
            case CMP_PAN::MODE::WIDE:
            {
                LeftAmplitude   = -TargetAmplitude;
                RightAmplitude  = TargetAmplitude;         
                break;
            }                  
            default:
            {
                LeftAmplitude   = TargetAmplitude;
                RightAmplitude  = TargetAmplitude;
                break;
            }                        
        }

        for(u32 FrameIndex = 0; FrameIndex < SamplesToWrite; ++FrameIndex)
        {
            //Get sample
            f32 Sample = SourceBuffer.Data[FrameIndex];

            //Pan
            *Channel0++ += Sample * LeftAmplitude;
            *Channel1++ += Sample * RightAmplitude;
        }
    }

    //If not then increment to target
    else
    {
        f64 Current = SourceAmplitude.Previous;
        f64 Step = (TargetAmplitude - Current) * 1.0f / SamplesToWrite;
        f64 LeftAmplitude   = 0;
        f64 RightAmplitude  = 0;
        
        for(u32 FrameIndex = 0; FrameIndex < SamplesToWrite; ++FrameIndex)
        {
            //Update amplitude
            Current += Step;

            //Get sample
            f32 Sample = SourceBuffer.Data[FrameIndex];

            switch(SourcePan.Flag)
            {
                case CMP_PAN::MODE::MONO:
                {
                    LeftAmplitude   = Current * 0.707;
                    RightAmplitude  = Current * 0.707;            
                    break;
                }
                case CMP_PAN::MODE::STEREO:
                {
                    //Calculate constant power values
                    f64 Angle = ((1.0 - SourcePan.Amplitude) * PI32 * 0.25);
                    LeftAmplitude   = Current * sin(Angle);
                    RightAmplitude  = Current * cos(Angle);          
                    break;
                }
                case CMP_PAN::MODE::WIDE:
                {
                    LeftAmplitude   = -Current;
                    RightAmplitude  = Current;      
                    break;
                }                
                default:
                {
                    LeftAmplitude   = Current;
                    RightAmplitude  = Current;
                    break;
                }                        
            }

            //Pan
            *Channel0++ += Sample * LeftAmplitude;
            *Channel1++ += Sample * RightAmplitude;
        }

        SourceAmplitude.Previous = TargetAmplitude;
    }
}   

size_t SYS_MIX::Update(ENTITY_SOURCES *Sources, f32 *MixerChannel0, f32 *MixerChannel1, size_t SamplesToWrite)
{
    //Assign mixing channels
    f32 *Channel0 = MixerChannel0;
    f32 *Channel1 = MixerChannel1;

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
                            CMP_FADE &SourceAmplitude = Sources->Amplitudes[SourceIndex];
                            CMP_PAN &SourcePan = Sources->Pans[SourceIndex];
                            RenderToBuffer(Channel0, Channel1, SamplesToWrite, Playback.Buffer, SourceAmplitude, SourcePan, SourceAmplitude.Current);
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
                            CMP_FADE &SourceAmplitude = Sources->Amplitudes[SourceIndex];
                            CMP_PAN &SourcePan = Sources->Pans[SourceIndex];
                            RenderToBuffer(Channel0, Channel1, SamplesToWrite, Playback.Buffer, SourceAmplitude, SourcePan, 0.0f);
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

    return SamplesToWrite;
}