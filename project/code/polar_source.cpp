#ifndef polar_source_cpp
#define polar_source_cpp


POLAR_SOURCE *polar_source_Retrieval(POLAR_MIXER *Mixer, const char UID[MAX_STRING_LENGTH], u32 &SourceIndex)
{
    u64 Hash = Hash(UID);

    for(POLAR_SUBMIX *SubmixIndex = Mixer->FirstInList; SubmixIndex; SubmixIndex = SubmixIndex->NextSubmix)
    {
        for(u32 i = 0; i <= SubmixIndex->Containers.CurrentContainers; ++i)
        {
            //Looking for blank source in a specifc container
            if(SubmixIndex->Containers.UID[i] == Hash)
            {
                for(u32 j = 0; j <= SubmixIndex->Containers.Sources[i].CurrentSources; ++j)
                {
                    if(SubmixIndex->Containers.Sources[i].UID[j] == 0)
                    {
                        SourceIndex = j;
                        return &SubmixIndex->Containers.Sources[i];
                    }
                }
            }

            //Looking for specific source in all containers
            else
            {
                for(u32 j = 0; j <= SubmixIndex->Containers.Sources[i].CurrentSources; ++j)
                {
                    if(SubmixIndex->Containers.Sources[i].UID[j] == Hash)
                    {
                        SourceIndex = j;
                        return &SubmixIndex->Containers.Sources[i];
                    }
                }
            }
        }

        //Same as above but for child submixes
        for(POLAR_SUBMIX *ChildSubmixIndex = SubmixIndex->ChildSubmix; ChildSubmixIndex; ChildSubmixIndex = ChildSubmixIndex->ChildSubmix)
        {
            for(u32 i = 0; i <= ChildSubmixIndex->Containers.CurrentContainers; ++i)
            {
                if(ChildSubmixIndex->Containers.UID[i] == Hash)
                {
                    for(u32 j = 0; j <= ChildSubmixIndex->Containers.Sources[i].CurrentSources; ++j)
                    {
                        if(ChildSubmixIndex->Containers.Sources[i].UID[j] == 0)
                        {
                            SourceIndex = j;
                            return &ChildSubmixIndex->Containers.Sources[i];
                        }
                    }
                }

                else
                {
                    for(u32 j = 0; j <= ChildSubmixIndex->Containers.Sources[i].CurrentSources; ++j)
                    {
                        if(ChildSubmixIndex->Containers.Sources[i].UID[j] == Hash)
                        {
                            SourceIndex = j;
                            return &ChildSubmixIndex->Containers.Sources[i];
                        }
                    }
                }
            }
        }
    }

    printf("Polar\tERROR: Failed to retrieve %s\n", UID);
    return 0;
}


void polar_source_Create(MEMORY_ARENA *Arena, POLAR_MIXER *Mixer, POLAR_ENGINE Engine, const char ContainerUID[MAX_STRING_LENGTH], const char SourceUID[MAX_STRING_LENGTH], u32 Channels, u32 Type, ...)
{
    u64 SourceHash = Hash(SourceUID);

    u32 j = 0;
    POLAR_SOURCE *Sources = polar_source_Retrieval(Mixer, ContainerUID, j);
    
    if(!Sources)
    {
        printf("Polar\tERROR: Failed to add source %s to container %s\n", SourceUID, ContainerUID);
        return;
    }
    
    Sources->UID[j] = SourceHash;
    Sources->PlayState[j] = Stopped;
    Sources->Channels[j] = Channels;

    Sources->States[j].CurrentEnvelopes = 0;

    Sources->States[j].AmplitudeCurrent = 0;
    Sources->States[j].AmplitudePrevious = Sources->States[j].AmplitudeCurrent;

    Sources->SampleRate[j] = Engine.SampleRate;
    Sources->BufferSize[j] = Engine.BufferSize;
    Sources->SampleCount[j] = 0;
    Sources->States[j].PanPositions = (f32 *) memory_arena_Push(Arena, Sources->States[j].PanPositions, Sources->Channels[j]);
    Sources->Buffer[j] = (f32 *) memory_arena_Push(Arena, Sources->Buffer[j], Sources->BufferSize[j]);

    bool SourceCreated = false;
    Sources->Type[j].Flag = Type;
    switch(Sources->Type[j].Flag)
    {
        case SO_FILE:
        {
            va_list ArgList;
            va_start(ArgList, Type);
            char *Name = va_arg(ArgList, char *);
            
            char FilePath[MAX_STRING_LENGTH];
            polar_StringConcatenate(StringLength(AssetPath), AssetPath, StringLength(Name), Name, FilePath);

            Sources->Type[j].File = (POLAR_FILE *) memory_arena_Push(Arena, Sources->Type[j].File, (sizeof (POLAR_FILE)));
            Sources->Type[j].File->Samples = drwav_open_file_and_read_pcm_frames_f32(FilePath, &Sources->Type[j].File->Channels, &Sources->Type[j].File->SampleRate, &Sources->Type[j].File->FrameCount);  

            if(!Sources->Type[j].File->Samples)
            {
                printf("Polar\tERROR: Cannot open file %s\n", FilePath);
                break;
            }

            if(Sources->Type[j].File->SampleRate != Engine.SampleRate)
            {
                f32 *ResampleBuffer = 0;
                ResampleBuffer = (f32 *) memory_arena_Push(Arena, ResampleBuffer, (sizeof (f32) * Sources->Type[j].File->FrameCount));

                switch(Engine.SampleRate)
                {
                    case 48000:
                    {
                        switch(Sources->Type[j].File->SampleRate)
                        {
                            case 44100:
                            {
                                Resample(Sources->Type[j].File->Samples, Sources->Type[j].File->FrameCount, ResampleBuffer, (Sources->Type[j].File->FrameCount) * 0.91875f);
                            }
                        }
                    }

                    case 192000:
                    {
                        switch(Sources->Type[j].File->SampleRate)
                        {
                            case 44100:
                            {
                                // Resample(Sources->Type[j].File->Samples, Sources->Type[j].File->FrameCount, ResampleBuffer, (Sources->Type[j].File->FrameCount) * 0.435374f);
                                // Resample(Sources->Type[j].File->Samples, Sources->Type[j].File->FrameCount, ResampleBuffer, (Sources->Type[j].File->FrameCount) * 1.0f);
                            }
                        }
                    }
                }

                Sources->Type[j].File->Samples = ResampleBuffer;
            }

            va_end(ArgList);
            SourceCreated = true;
            break;
        }

        case SO_OSCILLATOR:
        {
            va_list ArgList;
            va_start(ArgList, Type);
            u32 Wave = va_arg(ArgList, u32);
            f32 Frequency = va_arg(ArgList, f64);
            Sources->Type[j].Oscillator = polar_dsp_OscillatorCreate(Arena, Engine.SampleRate, Wave, Frequency);
                                
            va_end(ArgList);
            SourceCreated = true;
            break;
        }

        default:
        {
            Sources->Type[j].Oscillator = 0;
            Sources->Type[j].File = 0;
            break;
        }
    }

    if(SourceCreated)
    {
        Sources->FX[j] = FX_DRY;
        Sources->CurrentSources += 1;

        return;
    }

    printf("Polar\tERROR: Failed to add source %s to container %s\n", SourceUID, ContainerUID);
    return;
}

void polar_source_CreateFromFile(MEMORY_ARENA *Arena, POLAR_MIXER *Mixer, POLAR_ENGINE Engine, const char *FileName)
{
    FILE *File;

    char FilePath[MAX_STRING_LENGTH];
    polar_StringConcatenate(StringLength(AssetPath), AssetPath, StringLength(FileName), FileName, FilePath);

#if _WIN32    
    fopen_s(&File, FilePath, "r");
#else
    File = fopen(FilePath, "r");
#endif

    Assert(File);

    char SourceLine[256];
    i32 FileError = 0;
    u64 Index = 0;
    
    char Container[128];
    char Source[128];
    u32 Channels = 0;
    char Type[32];
    f64 Frequency = 0;

    while(fgets(SourceLine, 256, File))
    {
        FileError = sscanf(SourceLine, "%s %s %d %s %lf", Container, Source, &Channels, Type, &Frequency);
        if(strncmp(Type, "SO_OSCILLATOR", 32) == 0)
        {
            polar_source_Create(Arena, Mixer, Engine, Container, Source, Channels, SO_OSCILLATOR, WV_SINE, Frequency);
        }
        ++Index;
    }
}

#define ZERO 1e-10
#define isBetween(A, B, C) ( ((A-B) > -ZERO) && ((A-C) < ZERO) )

void polar_source_Update(POLAR_MIXER *Mixer, POLAR_SOURCE *Sources, u32 &SourceIndex, f64 GlobalTime, f32 NoiseFloor)
{
    f64 UpdatePeriod = 0.1f;
    bool IsEnvelope = false;

    //Update envelopes if they exist
    for(u8 EnvelopeIndex = 0; EnvelopeIndex < Sources->States[SourceIndex].CurrentEnvelopes; ++EnvelopeIndex)
    {
        if(Sources->States[SourceIndex].Envelope[EnvelopeIndex].CurrentPoints > 0 && Sources->States[SourceIndex].Envelope[EnvelopeIndex].Index < Sources->States[SourceIndex].Envelope[EnvelopeIndex].CurrentPoints)
        {
            IsEnvelope = true;

            switch(Sources->States[SourceIndex].Envelope[EnvelopeIndex].Assignment)
            {
                case EN_AMPLITUDE:
                {
                    u32 PointIndex = Sources->States[SourceIndex].Envelope[EnvelopeIndex].Index;
                    f32 Current = Sources->States[SourceIndex].AmplitudeCurrent;

                    if(PointIndex == 0)
                    {
                        f64 Amplitude = Sources->States[SourceIndex].Envelope[EnvelopeIndex].Points[PointIndex].Value;
                        f64 Duration = Sources->States[SourceIndex].Envelope[EnvelopeIndex].Points[PointIndex + 1].Time - Sources->States[SourceIndex].Envelope[EnvelopeIndex].Points[PointIndex].Time;
                        
                        Sources->States[SourceIndex].FadeStartAmplitude = Sources->States[SourceIndex].AmplitudeCurrent;
                        Sources->States[SourceIndex].FadeEndAmplitude = Amplitude;
                        Sources->States[SourceIndex].FadeDuration = MAX(Duration, 0.0f);
                        Sources->States[SourceIndex].FadeStartTime = GlobalTime;
                        Sources->States[SourceIndex].IsFading = true;

                        ++PointIndex;
                        ++Sources->States[SourceIndex].Envelope[EnvelopeIndex].Index;
                    }

                    else
                    {
                        if(Current != Sources->States[SourceIndex].Envelope[EnvelopeIndex].Points[PointIndex + 1].Value)
                        {
                            f64 Amplitude = Sources->States[SourceIndex].Envelope[EnvelopeIndex].Points[PointIndex].Value;
                            f64 Duration = Sources->States[SourceIndex].Envelope[EnvelopeIndex].Points[PointIndex + 1].Time - Sources->States[SourceIndex].Envelope[EnvelopeIndex].Points[PointIndex].Time;
                        
                            Sources->States[SourceIndex].FadeStartAmplitude = Sources->States[SourceIndex].AmplitudeCurrent;
                            Sources->States[SourceIndex].FadeEndAmplitude = Amplitude;
                            Sources->States[SourceIndex].FadeDuration = MAX(Duration, 0.0f);
                            Sources->States[SourceIndex].FadeStartTime = GlobalTime;
                            Sources->States[SourceIndex].IsFading = true;

                            ++PointIndex;
                            ++Sources->States[SourceIndex].Envelope[EnvelopeIndex].Index;
                        }
                    }

                    break;
                }

                //!Create a per-sample parameter struct to use for frequency ramping!
                case EN_FREQUENCY:
                {
                    u32 PointIndex = Sources->States[SourceIndex].Envelope[EnvelopeIndex].Index;
                    f32 Current = Sources->Type[SourceIndex].Oscillator->FrequencyCurrent;

                    if(PointIndex == 0)
                    {
                        f64 Frequency = Sources->States[SourceIndex].Envelope[EnvelopeIndex].Points[PointIndex].Value;
                        Sources->Type[SourceIndex].Oscillator->FrequencyCurrent = Frequency;

                        ++PointIndex;
                        ++Sources->States[SourceIndex].Envelope[EnvelopeIndex].Index;
                    }

                    else
                    {
                        if(Current != Sources->States[SourceIndex].Envelope[EnvelopeIndex].Points[PointIndex + 1].Value)
                        {
                            f64 Frequency = Sources->States[SourceIndex].Envelope[EnvelopeIndex].Points[PointIndex].Value;
                            Sources->Type[SourceIndex].Oscillator->FrequencyCurrent = Frequency;

                            ++PointIndex;
                            ++Sources->States[SourceIndex].Envelope[EnvelopeIndex].Index;
                        }
                    }

                    break;
                }
                
                default:
                {
                    break;
                }
            }
        }
    }

    //Update fades
    GlobalTime = polar_WallTime();
    f32 TimePassed = GlobalTime - Sources->States[SourceIndex].FadeStartTime;
    f32 FadeCompletion = TimePassed / Sources->States[SourceIndex].FadeDuration;
    Sources->States[SourceIndex].AmplitudeCurrent = ((Sources->States[SourceIndex].FadeEndAmplitude - Sources->States[SourceIndex].FadeStartAmplitude) * FadeCompletion) + Sources->States[SourceIndex].FadeStartAmplitude;

    if(FadeCompletion >= 1.0f)
    {
        Sources->States[SourceIndex].AmplitudeCurrent = Sources->States[SourceIndex].FadeEndAmplitude;
        Sources->States[SourceIndex].IsFading = false;
        Sources->States[SourceIndex].FadeDuration = 0.0f;
    }

    //Update distance attenuation
    if(Sources->States[SourceIndex].IsDistanceAttenuated)
    {
        f32 Attenuation = Sources->States[SourceIndex].AmplitudeCurrent;
        f32 Distance = polar_listener_DistanceFromListener(Mixer->Listener, Sources->States[SourceIndex], NoiseFloor);
        
        if(Distance != 0)
        {
            Attenuation = Sources->States[SourceIndex].AmplitudeCurrent - Distance;
        }

        Sources->States[SourceIndex].AmplitudeCurrent = Attenuation;
        
        // printf("Amplitude:\tCurrent: %f\tPrev: %f\tDistance: %f\n", Sources->States[SourceIndex].AmplitudeCurrent, Sources->States[SourceIndex].AmplitudePrevious, Attenuation);
    }

    // printf("Amplitude:\tCurrent: %f\tPrev: %f\n", Sources->States[SourceIndex].AmplitudeCurrent, Sources->States[SourceIndex].AmplitudePrevious);
}

void polar_source_UpdatePlaying(POLAR_MIXER *Mixer, f64 GlobalTime, f32 NoiseFloor)
{
    for(POLAR_SUBMIX *SubmixIndex = Mixer->FirstInList; SubmixIndex; SubmixIndex = SubmixIndex->NextSubmix)
    {
        for(u32 i = 0; i <= SubmixIndex->Containers.CurrentContainers; ++i)
        {
            for(u32 j = 0; j <= SubmixIndex->Containers.Sources[i].CurrentSources; ++j)
            {
                if(SubmixIndex->Containers.Sources[i].PlayState[j] == Playing)
                {
                    POLAR_SOURCE *Sources = &SubmixIndex->Containers.Sources[i];
                    polar_source_Update(Mixer, Sources, j, GlobalTime, NoiseFloor);
                }
            }
        }

        for(POLAR_SUBMIX *ChildSubmixIndex = SubmixIndex->ChildSubmix; ChildSubmixIndex; ChildSubmixIndex = ChildSubmixIndex->ChildSubmix)
        {
            for(u32 i = 0; i <= ChildSubmixIndex->Containers.CurrentContainers; ++i)
            {
                for(u32 j = 0; j <= ChildSubmixIndex->Containers.Sources[i].CurrentSources; ++j)
                {
                    if(ChildSubmixIndex->Containers.Sources[i].PlayState[j] == Playing)
                    {
                        POLAR_SOURCE *Sources = &ChildSubmixIndex->Containers.Sources[i];
                        polar_source_Update(Mixer, Sources, j, GlobalTime, NoiseFloor);
                    }
                }
            }
        }
    }
}


void polar_source_Play(POLAR_MIXER *Mixer, const char *SourceUID, f64 GlobalTime, f32 Duration, f32 *PanPositions, u32 FX, u32 EnvelopeType, ...)
{
    u32 i = 0;
    POLAR_SOURCE *Sources = polar_source_Retrieval(Mixer, SourceUID, i);
    if(!Sources)
    {
        printf("Polar\tERROR: Cannot play source %s\n", SourceUID);
        return;
    }

    if(Sources->Type[i].Flag == SO_FILE)
    {
        if(Duration <= 0)
        {
            Sources->SampleCount[i] = Sources->Type[i].File->FrameCount;
        }

        else
        {   
            Sources->SampleCount[i] = ((Sources->SampleRate[i] * Sources->Channels[i])  * Duration);
        }
        
    }

    if(Sources->Type[i].Flag == SO_OSCILLATOR)
    {
        Sources->SampleCount[i] = ((Sources->SampleRate[i] * Sources->Channels[i])  * Duration);
    }

    switch(EnvelopeType)
    {
        case EN_NONE:
        {
            va_list ArgList;
            va_start(ArgList, EnvelopeType);

            f64 AmpNew = va_arg(ArgList, f64);

            Sources->States[i].AmplitudeCurrent = AmpNew;
            Sources->States[i].AmplitudePrevious = Sources->States[i].AmplitudeCurrent;

            Sources->States[i].MinDistance = 1.0f;
            Sources->States[i].MaxDistance = 100.0f;
            Sources->States[i].Rolloff = 1.0f;
            Sources->States[i].RolloffDirty = true; //!Must be set true if min/max or rolloff are changed!
            Sources->States[i].IsDistanceAttenuated = true;

            
            //!Source won't play unless this is called - check how amplitude is first updated/automatic fading
            polar_source_Fade(Mixer, SourceUID, GlobalTime, AmpNew, 0.001);


            va_end(ArgList);
            break;
        }

        case EN_ADSR:
        {
            va_list ArgList;
            va_start(ArgList, EnvelopeType);
                            
            Sources->States[i].Envelope[0].Assignment = EN_AMPLITUDE;
            Sources->States[i].Envelope[0].Points[0].Time = 0.5;
            Sources->States[i].Envelope[0].Points[1].Time = 0.1;
            Sources->States[i].Envelope[0].Points[2].Time = 1.3;
            Sources->States[i].Envelope[0].Points[3].Time = 0.6;
            Sources->States[i].Envelope[0].Points[0].Value = AMP(-1);
            Sources->States[i].Envelope[0].Points[1].Value = AMP(-4);
            Sources->States[i].Envelope[0].Points[2].Value = AMP(-1);
            Sources->States[i].Envelope[0].Points[3].Value = AMP(-8);
            Sources->States[i].Envelope[0].CurrentPoints = 4;
            Sources->States[i].Envelope[0].Index = 0;
            Sources->States[i].CurrentEnvelopes = 1;

            va_end(ArgList);
            break;
        }

        case EN_BREAKPOINT:
        {
            va_list ArgList;
            va_start(ArgList, EnvelopeType);
            char *FileName = va_arg(ArgList, char *);

            FILE *BreakpointFile;

            char FilePath[MAX_STRING_LENGTH];
            polar_StringConcatenate(StringLength(AssetPath), AssetPath, StringLength(FileName), FileName, FilePath);
#if _WIN32    
            fopen_s(&BreakpointFile, FilePath, "r");
#else
            BreakpointFile = fopen(FilePath, "r");
#endif
            polar_envelope_BreakpointsFromFile(BreakpointFile, Sources->States[i].Envelope[0], Sources->States[i].Envelope[1]);
            Sources->States[i].CurrentEnvelopes = 2;

            Sources->States[i].AmplitudeCurrent = 0;
            Sources->States[i].AmplitudePrevious = Sources->States[i].AmplitudeCurrent;

            va_end(ArgList);
            break;
        }

        default:
        {
            Sources->States[i].AmplitudeCurrent = DEFAULT_AMPLITUDE;
            Sources->States[i].AmplitudePrevious = Sources->States[i].AmplitudeCurrent;

            break;
        }
    }

    Sources->FX[i] = FX;
    Sources->PlayState[i] = Playing;

    for(u32 f = 0; f < ArrayCount(PanPositions); ++f)
    {
        Sources->States[i].PanPositions[f] = PanPositions[f];
    }

    return;
}

//!Add some kind of check for IsFading when a new fade event comes in - overwrite with the new fade? pause?
void polar_source_Fade(POLAR_MIXER *Mixer, const char *SourceUID, f64 GlobalTime, f32 NewAmplitude, f32 Duration)
{
    u32 i = 0;
    POLAR_SOURCE *Sources = polar_source_Retrieval(Mixer, SourceUID, i);
    if(!Sources)
    {
        printf("Polar\tERROR: Cannot fade source %s\n", SourceUID);
        return;
    }

    Sources->States[i].FadeStartAmplitude = Sources->States[i].AmplitudeCurrent;
    Sources->States[i].FadeEndAmplitude = NewAmplitude;
    Sources->States[i].FadeDuration = MAX(Duration, 0.0f);
    Sources->States[i].FadeStartTime = GlobalTime;
    Sources->States[i].IsFading = true;

    return; 
}



#endif