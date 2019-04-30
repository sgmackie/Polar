
void SYS_FADE::Create(MEMORY_ARENA *Arena, size_t Size)
{
    SystemVoices = (ID_SOURCE *) Arena->Alloc((sizeof(ID_SOURCE) * Size), MEMORY_ARENA_ALIGNMENT);
    SystemCount = 0;
}

void SYS_FADE::Destroy(MEMORY_ARENA *Arena)
{
    Arena->FreeAll();
}

void SYS_FADE::Add(ID_SOURCE ID)
{
    SystemVoices[SystemCount] = ID;
    ++SystemCount;
}

bool SYS_FADE::Remove(ID_SOURCE ID)
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

bool SYS_FADE::Start(ENTITY_VOICES *Voices, ID_VOICE ID, f64 Time, f64 Amplitude, f64 Duration)
{
    //Grab the component
    for(size_t i = 0; i <= SystemCount; ++i)
    {
        if(SystemVoices[i] == ID)
        {
            CMP_FADE &Fade          = Voices->Amplitudes[i];
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

void SYS_FADE::Update(ENTITY_VOICES *Voices, f64 Time)
{
    //Loop through every source that was added to the system
    for(size_t i = 0; i <= SystemCount; ++i)
    {
        //Find active sources in the system
        ID_VOICE Voice = SystemVoices[i];
        if(Voice != 0)
        {
            //Grab the component
            CMP_FADE &Fade = Voices->Amplitudes[i];

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

                Debug("Fade: Voice %llu:\tCurrent: %f | Previous: %f | Duration: %f", Voice, Fade.Current, Fade.Previous, Fade.Duration);
            }
        }
    }
}