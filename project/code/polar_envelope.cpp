#ifndef polar_envelope_cpp
#define polar_envelope_cpp


void polar_envelope_BreakpointsFromFile(FILE *File, POLAR_ENVELOPE &Amplitude, POLAR_ENVELOPE &Frequency)
{
    Assert(File);
    char BreakpointLine[MAX_BREAKPOINT_LINE_LENGTH];
    i32 FileError = 0;
    u64 Index = 0;

    while(fgets(BreakpointLine, MAX_BREAKPOINT_LINE_LENGTH, File))
    {
        FileError = sscanf(BreakpointLine, "%f, %f, %f", &Amplitude.Points[Index].Time, &Frequency.Points[Index].Value, &Amplitude.Points[Index].Value);
        
        if(FileError != 3)
        {
            printf("Polar\tERROR: Failed to read breakpoint file\n");
            return;
        }

        ++Amplitude.CurrentPoints;
        ++Frequency.CurrentPoints;
        ++Index;
    }

    for(u32 i = 0; i < Amplitude.CurrentPoints; ++i)
    {
        Frequency.Points[i].Time = Amplitude.Points[i].Time;
    }

    Amplitude.Assignment = EN_AMPLITUDE;
    Frequency.Assignment = EN_FREQUENCY;

    Amplitude.Index = 0;
    Frequency.Index = 0;
}


#endif