
void SYS_PLAY::Create(MEMORY_ARENA *Arena, size_t Size)
{
    SystemSources = (ID_SOURCE *) Arena->Alloc((sizeof(ID_SOURCE) * Size), MEMORY_ARENA_ALIGNMENT);
    SystemCount = 0;
}

void SYS_PLAY::Destroy(MEMORY_ARENA *Arena)
{
    Arena->FreeAll();
}

void SYS_PLAY::Add(ID_SOURCE ID)
{
    SystemSources[SystemCount] = ID;
    ++SystemCount;
}

bool SYS_PLAY::Remove(ID_SOURCE ID)
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


bool SYS_PLAY::Start(ENTITY_SOURCES *Sources, ID_SOURCE ID, f64 InputDuration, bool IsAligned)
{
    //Grab the component
    for(size_t i = 0; i <= SystemCount; ++i)
    {
        if(SystemSources[i] == ID)
        {
            TPL_PLAYBACK &Playback              = Sources->Playbacks[i];
            Playback.Duration.SampleCount       = InputDuration * (Playback.Format.SampleRate);
            Playback.Duration.States            = PLAYING;

            if(IsAligned)
            {
                Playback.Duration.SampleCount = NearestPowerOf2(Playback.Duration.SampleCount);
            }

            Info("Play: Source %llu:\tDuration: %f | SampleCount: %llu", SystemSources[i], InputDuration, Playback.Duration.SampleCount);

            return true;
        }
    }

    //!Log
    return false;
}


bool SYS_PLAY::Pause(ENTITY_SOURCES *Sources, ID_SOURCE ID)
{
    //Grab the component
    for(size_t i = 0; i <= SystemCount; ++i)
    {
        if(SystemSources[i] == ID)
        {
            CMP_DURATION &Duration              = Sources->Playbacks[i].Duration;
            Duration.States                     = PAUSED;

            Info("Play: Source %llu:\tPaused");

            return true;
        }
    }

    //!Log
    return false;
}

bool SYS_PLAY::Resume(ENTITY_SOURCES *Sources, ID_SOURCE ID)
{
    //Grab the component
    for(size_t i = 0; i <= SystemCount; ++i)
    {
        if(SystemSources[i] == ID)
        {
            CMP_DURATION &Duration              = Sources->Playbacks[i].Duration;
            Duration.States                     = PLAYING;

            Info("Play: Source %llu:\tResuming");

            return true;
        }
    }

    //!Log
    return false;
}

void SYS_PLAY::Update(ENTITY_SOURCES *Sources, f64 Time, u32 PreviousSamplesWritten, u32 SamplesToWrite)
{
    //Loop through every source that was added to the system
    for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
    {
        //Find active sources in the system
        ID_SOURCE Source = SystemSources[SystemIndex];
        if(Source != 0)
        {
            size_t SourceIndex			= Sources->RetrieveIndex(Source);
            CMP_DURATION &Duration      = Sources->Playbacks[SourceIndex].Duration;
            
            if(Duration.States != STOPPED)
            {
                if(Duration.SampleCount <= 0)
                {
                    Duration.SampleCount = 0;
                    Duration.States = STOPPED;
                }

                else if(Duration.SampleCount <= PreviousSamplesWritten)
                {
                    Duration.SampleCount = 0;
                    Duration.States = STOPPED;
                }

                //If ending in the next callback, mark as stopping (fade out)
                else if(Duration.SampleCount <= (SamplesToWrite))
                {
                    Duration.SampleCount = SamplesToWrite;
                    Duration.States = STOPPING;
                }

				else
                {
                    Duration.SampleCount -= PreviousSamplesWritten;
                    Debug("Play: Source %llu:\tSampleCount: %llu", Source, Duration.SampleCount);
                }
            }
        }
    }
}