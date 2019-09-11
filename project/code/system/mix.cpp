

void SYS_MIX::Create(MEMORY_ALLOCATOR *Allocator, size_t Size)
{
    SystemSources = (ID_VOICE *) Allocator->Alloc((sizeof(ID_VOICE) * Size), HEAP_TAG_SYSTEM_MIX);
    SystemCount = 0;
}

 void SYS_MIX::Destroy(MEMORY_ALLOCATOR *Allocator)
{
    Allocator->Free(0, HEAP_TAG_SYSTEM_MIX);
}

 void SYS_MIX::Add(ID_VOICE Voice)
{
    SystemSources[SystemCount] = Voice;
    ++SystemCount;
}

 bool SYS_MIX::Remove(ID_VOICE ID)
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

 void SYS_MIX::RenderToBuffer(f32 *Channel0, f32 *Channel1, size_t SamplesToWrite, CMP_BUFFER &SourceBuffer, CMP_FADE &Amplitude, CMP_PAN &SourcePan, f64 TargetAmplitude)
{
    //Check if amplitude is already at the target
    if(fabsl(Amplitude.Previous - TargetAmplitude) < 0.1e-5)
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
        f64 Current = Amplitude.Previous;
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

        Amplitude.Previous = TargetAmplitude;
    }
}   

 size_t SYS_MIX::Update(ENTITY_SOURCES *Sources, ENTITY_VOICES *Voices, f32 *MixerChannel0, f32 *MixerChannel1, size_t SamplesToWrite)
{
    //Assign mixing channels
    f32 *Channel0 = MixerChannel0;
    f32 *Channel1 = MixerChannel1;

    //Loop through every source that was added to the system
    for(size_t SystemIndex = 0; SystemIndex < SystemCount; ++SystemIndex)
    {
        //Find active voices
        ID_VOICE Voice = SystemSources[SystemIndex];
        if(Voice != 0)
        {
            //Get voice buffer
            size_t VoiceIndex = Voices->RetrieveIndex(Voice);
            CMP_BUFFER &VoicePlayback = Voices->Playbacks[VoiceIndex].Buffer;
            CMP_STATE &State = Voices->States[VoiceIndex];

            //Get handle to associated source
            HANDLE_SOURCE Source = Voices->Sources[VoiceIndex];

            //Check playback state
            switch(State.Play)
            {
                case PLAYING:
                {
                    if(Sources->Flags[Source.Index] & ENTITY_SOURCES::AMPLITUDE)
                    {
                        CMP_FADE &Amplitude = Voices->Amplitudes[VoiceIndex];
                        CMP_DISTANCE &Distance = Voices->Distances[VoiceIndex];                        
                        CMP_PAN &SourcePan = Sources->Pans[Source.Index];

                        f64 FinalAmplitude = Amplitude.Current - Distance.Attenuation;

                        RenderToBuffer(Channel0, Channel1, SamplesToWrite, VoicePlayback, Amplitude, SourcePan, FinalAmplitude);
                    }
                    break;
                }
                case STOPPED:
                {
                    break;
                }
                case STOPPING:
                {
                    //Fade to 0
                    if(Sources->Flags[Source.Index] & ENTITY_SOURCES::AMPLITUDE)
                    {
                        CMP_FADE &Amplitude = Voices->Amplitudes[VoiceIndex];
                        CMP_PAN &SourcePan = Sources->Pans[Source.Index];
                        RenderToBuffer(Channel0, Channel1, SamplesToWrite, VoicePlayback, Amplitude, SourcePan, 0.0f);
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

     return SamplesToWrite;
}