

void SYS_ENVELOPE_BREAKPOINT::Create(MEMORY_ARENA *Arena, size_t Size)
{
    SystemVoices = (ID_VOICE *) Arena->Alloc((sizeof(ID_VOICE) * Size), MEMORY_ARENA_ALIGNMENT);
    SystemCount = 0;
}

void SYS_ENVELOPE_BREAKPOINT::Destroy(MEMORY_ARENA *Arena)
{
    Arena->FreeAll();
}

void SYS_ENVELOPE_BREAKPOINT::Add(ID_VOICE ID)
{
    SystemVoices[SystemCount] = ID;
    ++SystemCount;
}

bool SYS_ENVELOPE_BREAKPOINT::Remove(ID_VOICE ID)
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

void SYS_ENVELOPE_BREAKPOINT::CreateFromFile(ENTITY_VOICES *Voices, ID_VOICE ID, char const *File)
{
    //Loop through every voice that was added to the system
    for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
    {
        //Find active voices in the system
        ID_VOICE Voice = SystemVoices[SystemIndex];
        if(Voice == ID)
        {
            //Voice is valid - get component
            size_t Index                = Voices->RetrieveIndex(Voice);
            CMP_BREAKPOINT &Breakpoint  = Voices->Breakpoints[Index];

            FILE *InputFile = 0;
#ifdef _WIN32        
            fopen_s(&InputFile, File, "r");
#else       
            InputFile = fopen(File, "r");
#endif        
            int done = 0;
            int err = 0;

            for(u32 i = 0; i < MAX_BREAKPOINTS && done != 1; ++i)
            {
                char *Line = fread_csv_line(InputFile, MAX_STRING_LENGTH, &done, &err);
                if(done != 1)
                {
                    char **Values = split_on_unescaped_newlines(Line);

                    if(!err)
                    {
                        sscanf(*Values, "%lf,%lf", &Breakpoint.Points[i].Time, &Breakpoint.Points[i].Value);
                        ++Breakpoint.Count;
                    }
                }
            }
        }
    }
}



void SYS_ENVELOPE_BREAKPOINT::EditPoint(ENTITY_VOICES *Voices, ID_VOICE ID, size_t PointIndex, f64 Value, f64 Time)
{
    //Loop through every voice that was added to the system
    for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
    {
        //Find active voice in the system
        ID_VOICE Voice = SystemVoices[SystemIndex];
        if(Voice == ID)
        {
            //Voice is valid - get component
            size_t Index                = Voices->RetrieveIndex(Voice);
            CMP_BREAKPOINT &Breakpoint  = Voices->Breakpoints[Index];
            
            Breakpoint.Init(PointIndex, Value, Time);
        }
    }
}

void SYS_ENVELOPE_BREAKPOINT::Update(ENTITY_VOICES *Voices, SYS_FADE *Fade, f64 Time)
{
    //Loop through every voice that was added to the system
    for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
    {
        //Find active voice in the system
        ID_VOICE Voice = SystemVoices[SystemIndex];
        if(Voice != 0)
        {
            //Voice is valid - get component
            size_t Index                = Voices->RetrieveIndex(Voice);
            CMP_BREAKPOINT &Breakpoint  = Voices->Breakpoints[Index];
            CMP_FADE &Amplitude         = Voices->Amplitudes[Index];

            if(!Amplitude.IsFading)
            {
                bool PointSet = false;
                while(Breakpoint.Index < Breakpoint.Count && !PointSet)
                {
                    Fade->Start(Voices, Voice, Time, Breakpoint.Points[Breakpoint.Index].Value, Breakpoint.Points[Breakpoint.Index].Time);
                    ++Breakpoint.Index;
                    PointSet = true;
                }
            }
        }
    }
}