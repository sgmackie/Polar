void LOGGER::Init(FILE *InputFile, u8 InputLevel, bool EnableQuiet)
{
    //Initialise
    Data        = 0;
    IsLocked    = false;
    File        = 0;
	Level		= 0;
    IsQuiet     = false;

    //Set optionss
    if(InputFile)
    {
        File = InputFile;
    }

    if(InputLevel)
    {
        Level = InputLevel;
    }

    if(EnableQuiet)
    {
        IsQuiet = true;
    }
}

void *LOGGER::Lock()
{
    IsLocked = true;
    return Data;
}

void LOGGER::Unlock(void *InputData)
{
    IsLocked = false;
    Data = InputData;
}

void LOGGER::Log(u8 InputLevel, const char *SourceFile, u64 SourceLine, const char *Format, ...)
{
    if(InputLevel < Level) 
    {
        return;
    }

    //Lock recorded data by making a copy
    void *Data = Lock();

    //Get system time
    time_t Time = time(0);
    struct tm LocalTime = {};
    localtime_s(&LocalTime, &Time);

    //Log to console
    if(!IsQuiet) 
    {
        va_list args;
        char Buffer[16];
        Buffer[strftime(Buffer, sizeof(Buffer), "%H:%M:%S", &LocalTime)] = '\0';
        fprintf(stderr, "%s %-5s %s:%llu: ", Buffer, LOG_LEVELS[InputLevel], SourceFile, SourceLine);
        va_start(args, Format);
        vfprintf(stderr, Format, args);
        va_end(args);
        fprintf(stderr, "\n");
        fflush(stderr);
    }

    //Log to file
    if(File)
    {
        va_list args;
        char Buffer[32];
        Buffer[strftime(Buffer, sizeof(Buffer), "%Y-%m-%d %H:%M:%S", &LocalTime)] = '\0';
        fprintf(File, "%s %-5s %s:%llu: ", Buffer, LOG_LEVELS[InputLevel], SourceFile, SourceLine);
        va_start(args, Format);
        vfprintf(File, Format, args);
        va_end(args);
        fprintf(File, "\n");
        fflush(File);
    }

    //Release lock by copying back
    Unlock(Data);
}