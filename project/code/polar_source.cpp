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
    Sources->States[j].AmplitudeCurrent = 0;
    Sources->States[j].AmplitudeTarget = 0;
    Sources->States[j].AmplitudeDelta = 0;
    Sources->States[j].CurrentEnvelopes = 0;
                    
    Sources->SampleRate[j] = Engine.SampleRate;
    Sources->BufferSize[j] = (Engine.BufferFrames / Engine.Channels);
    Sources->SampleCount[j] = 0;
    Sources->States[j].PanPositions = (f32 *) memory_arena_Push(Arena, Sources->States[j].PanPositions, Sources->Channels[j]);
    Sources->Buffer[j] = (f32 *) memory_arena_Push(Arena, Sources->Buffer[j], Sources->BufferSize[j]);

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
                return;
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
            break;
        }

        default:
        {
            Sources->Type[j].Oscillator = 0;
            Sources->Type[j].File = 0;
            break;
        }
    }

    Sources->FX[j] = FX_DRY;

    Sources->CurrentSources += 1;
                        
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

void polar_source_Update(POLAR_SOURCE *Sources, u32 &SourceIndex)
{
    f64 UpdatePeriod = 0.1f;
    bool IsEnvelope = false;

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

                    if(math_Truncate(Sources->States[SourceIndex].AmplitudeCurrent, 1) == math_Truncate(Sources->States[SourceIndex].Envelope[EnvelopeIndex].Points[PointIndex].Value, 1))
                    {
                        ++PointIndex;
                        ++Sources->States[SourceIndex].Envelope[EnvelopeIndex].Index;
                    }

                    Sources->States[SourceIndex].AmplitudeTarget = Sources->States[SourceIndex].Envelope[EnvelopeIndex].Points[PointIndex].Value;
                    u32 Precision = 2;
                    UpdatePeriod = (Sources->States[SourceIndex].Envelope[EnvelopeIndex].Points[PointIndex].Time / Precision);

                    if(UpdatePeriod <= 0.0f)
                    {
                        Sources->States[SourceIndex].AmplitudeCurrent = Sources->States[SourceIndex].AmplitudeTarget;
                    }

                    else
                    {
                        f64 OneOverFade = 1.0f / UpdatePeriod;
                        Sources->States[SourceIndex].AmplitudeDelta = (OneOverFade * (Sources->States[SourceIndex].AmplitudeTarget - Sources->States[SourceIndex].AmplitudeCurrent));
                    }

                    // printf("Index: %u\tPeriod: %f\n", PointIndex, UpdatePeriod);

                    break;
                }

                case EN_FREQUENCY:
                {
                    u32 PointIndex = Sources->States[SourceIndex].Envelope[EnvelopeIndex].Index;

                    if(round(Sources->Type[SourceIndex].Oscillator->FrequencyCurrent) == round(Sources->States[SourceIndex].Envelope[EnvelopeIndex].Points[PointIndex].Value))
                    {
                        ++PointIndex;
                        ++Sources->States[SourceIndex].Envelope[EnvelopeIndex].Index;
                    }

                    Sources->Type[SourceIndex].Oscillator->FrequencyTarget = Sources->States[SourceIndex].Envelope[EnvelopeIndex].Points[PointIndex].Value;
                    u32 Precision = 4;
                    UpdatePeriod = (Sources->States[SourceIndex].Envelope[EnvelopeIndex].Points[PointIndex].Time / Precision);

                    if(UpdatePeriod <= 0.0f)
                    {
                        Sources->Type[SourceIndex].Oscillator->FrequencyCurrent = Sources->Type[SourceIndex].Oscillator->FrequencyTarget;
                    }

                    else
                    {
                        f64 OneOverFade = 1.0f / UpdatePeriod;
                        Sources->Type[SourceIndex].Oscillator->FrequencyDelta = (OneOverFade * (Sources->Type[SourceIndex].Oscillator->FrequencyTarget - Sources->Type[SourceIndex].Oscillator->FrequencyCurrent));
                    }

                    // printf("Frequency:\tCurrent: %f\tTarget: %f\tDelta: %f\n", Sources->Type[SourceIndex].Oscillator->FrequencyCurrent, Sources->Type[SourceIndex].Oscillator->FrequencyTarget, Sources->Type[SourceIndex].Oscillator->FrequencyDelta);

                    break;
                }
                
                default:
                {
                    break;
                }
            }
        }
    }

    if(IsEnvelope == false)
    {
        if(UpdatePeriod <= 0.0f)
        {
            Sources->States[SourceIndex].AmplitudeCurrent = Sources->States[SourceIndex].AmplitudeTarget;
        }

        else
        {
            f64 OneOverFade = 1.0f / UpdatePeriod;
            Sources->States[SourceIndex].AmplitudeDelta = (OneOverFade * (Sources->States[SourceIndex].AmplitudeTarget - Sources->States[SourceIndex].AmplitudeCurrent));
        }
    }

    // printf("Amplitude:\tCurrent: %f\tTarget: %f\tDelta: %f\n", Sources->States[SourceIndex].AmplitudeCurrent, Sources->States[SourceIndex].AmplitudeTarget, Sources->States[SourceIndex].AmplitudeDelta);
}

void polar_source_UpdatePlaying(POLAR_MIXER *Mixer)
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
                    polar_source_Update(Sources, j);
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
                        polar_source_Update(Sources, j);
                    }
                }
            }
        }
    }
}

void polar_source_UpdateAmplitude(POLAR_MIXER *Mixer, const char *SourceUID, f32 UpdatePeriod, f32 NewAmplitude)
{
    u32 j = 0;
    POLAR_SOURCE *Sources = polar_source_Retrieval(Mixer, SourceUID, j);
    Assert(Sources);

    if(UpdatePeriod <= 0.0f)
    {
        Sources->States[j].AmplitudeCurrent = Sources->States[j].AmplitudeTarget;
    }

    else
    {
        f32 OneOverFade = 1.0f / UpdatePeriod;
        Sources->States[j].AmplitudeTarget = NewAmplitude;
        Sources->States[j].AmplitudeDelta = OneOverFade * (Sources->States[j].AmplitudeTarget - Sources->States[j].AmplitudeCurrent);
    }   
   
    return; 
}


void polar_source_Play(POLAR_MIXER *Mixer, const char *SourceUID, f32 Duration, f32 *PanPositions, u32 FX, u32 EnvelopeType, ...)
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

    Sources->FX[i] = FX;
    Sources->PlayState[i] = Playing;

    switch(EnvelopeType)
    {
        case EN_NONE:
        {
            va_list ArgList;
            va_start(ArgList, EnvelopeType);

            f64 AmpNew = va_arg(ArgList, f64);
            Sources->States[i].AmplitudeTarget = AmpNew;

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

            va_end(ArgList);
        }

        default:
        {
            Sources->States[i].AmplitudeTarget = DEFAULT_AMPLITUDE;

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

#endif