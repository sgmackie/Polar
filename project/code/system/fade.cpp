
void SYS_FADE::Create(MEMORY_ARENA *Arena, size_t Size)
{
    SystemSources = (ID_SOURCE *) Arena->Alloc((sizeof(ID_SOURCE) * Size), MEMORY_ARENA_ALIGNMENT);
    SystemCount = 0;
}

void SYS_FADE::Destroy(MEMORY_ARENA *Arena)
{
    Arena->FreeAll();
}

void SYS_FADE::Add(ID_SOURCE ID)
{
    SystemSources[SystemCount] = ID;
    ++SystemCount;
}

bool SYS_FADE::Remove(ID_SOURCE ID)
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

bool SYS_FADE::Start(ENTITY_SOURCES *Sources, ID_SOURCE ID, f64 Time, f64 Amplitude, f64 Duration)
{
    //Grab the component
    for(size_t i = 0; i <= SystemCount; ++i)
    {
        if(SystemSources[i] == ID)
        {
            CMP_FADE &Fade          = Sources->Amplitudes[i];
            Fade.StartValue         = Fade.Current;
            Fade.EndValue           = CLAMP(Amplitude, 0.0, 1.0);
            Fade.Duration           = MAX(Duration, 0.0);
            Fade.StartTime          = Time;
            Fade.IsFading           = true;
            return true;
        }
    }
    //!Log
    return false;
}

void SYS_FADE::Update(ENTITY_SOURCES *Sources, f64 Time)
{
    //Loop through every source that was added to the system
    for(size_t i = 0; i <= SystemCount; ++i)
    {
        //Find active sources in the system
        ID_SOURCE Source = SystemSources[i];
        if(Source != 0 && Sources->Flags[i] & ENTITY_SOURCES::AMPLITUDE)
        {
            //Grab the component
            CMP_FADE &Fade = Sources->Amplitudes[i];

            if(Fade.Current == Fade.EndValue)
            {
                Fade.IsFading = false;
            }

            //If it's marked to fade, update
            if(Fade.IsFading)
            {
                f64 TimePassed = (Time - Fade.StartTime);
                f64 FadeCompletion = (TimePassed / Fade.Duration);
                Fade.Current = (((Fade.EndValue - Fade.StartValue) * FadeCompletion) + Fade.StartValue);
                
                if(FadeCompletion >= 1.0f)
                {
                    Fade.Current    = Fade.EndValue;
                    Fade.Duration   = 0.0f;
                    Fade.IsFading   = false;
                }

                Debug("Fade: Source %llu:\tCurrent: %f | Previous: %f | Duration: %f", Source, Fade.Current, Fade.Previous, Fade.Duration);
            }
        }
    }
}