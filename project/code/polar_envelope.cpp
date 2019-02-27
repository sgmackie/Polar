#ifndef polar_envelope_cpp
#define polar_envelope_cpp


void polar_envelope_BreakpointsFromFile(FILE *File, POLAR_ENVELOPE &Amplitude, POLAR_ENVELOPE &Frequency)
{
    Assert(File);
    char BreakpointLine[MAX_BREAKPOINT_LINE_LENGTH];
    i32 FileError = 0;
    u64 Index = 0;
    f64 CurrentLastTime = 0;

    while(fgets(BreakpointLine, MAX_BREAKPOINT_LINE_LENGTH, File))
    {
        FileError = sscanf(BreakpointLine, "%lf, %lf, %lf", &Amplitude.Points[Index].Time, &Frequency.Points[Index].Value, &Amplitude.Points[Index].Value);
        
        CurrentLastTime = Amplitude.Points[Index].Time;
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

// //Function for reading input text file
// BREAKPOINT_FORMAT *breakpoint_GetPoints(MEMORY_ARENA *Arena, FILE *File)
// {
//     Assert(File);
//     BREAKPOINT_FORMAT *Result = 0;
//     Result = (BREAKPOINT_FORMAT *) memory_arena_Push(Arena, Result, (sizeof (BREAKPOINT_FORMAT) * MAX_BREAKPOINTS));

//     char BreakpointLine[MAX_BREAKPOINT_LINE_LENGTH];
//     i32 FileError = 0;
//     u64 Index = 0;
//     f64 CurrentLastTime = 0;

//     while(fgets(BreakpointLine, MAX_BREAKPOINT_LINE_LENGTH, File))
//     {
//         FileError = sscanf(BreakpointLine, "%lf, %lf, %lf", &Result->Time[Index], &Result->Frequency[Index], &Result->Amplitude[Index]);

//         // if(FileError == 0)
//         // {
//         //     fprintf(stderr, "Error: Line %lu has non-nummeric value\n", Index+1);
//         //     break;
//         // }
   
//         // if(FileError == 1)
//         // {
//         //     fprintf(stderr, "Error: Incomplete breakpoint found at point %lu\n", Index+1);
//         //     break;
//         // }

//         // if(Result->Time[Index] < CurrentLastTime)
//         // {
//         //     fprintf(stderr, "Error: Time not increasing at point %lu\n", Index + 1);
//         //     break;
//         // }

//         CurrentLastTime = Result->Time[Index];
//         ++Result->CurrentPoints;
//         ++Index;
//     }

//     return Result;
// }

// // //Scans input file for out of range values, returning "1" if passing the test
// // int32 breakpoint_Input_RangeCheck(const BREAKPOINT_FORMAT *ReadPoints, float64 ValueMin, float64 ValueMax, uint64 CurrentPoints)
// // {
// //     int32 RangeCheck = 1;

// //     for(uint64 i = 0; i < CurrentPoints; i++)
// //     {
// //         if(ReadPoints[i].States.Amplitude < ValueMin || ReadPoints[i].States.Amplitude > ValueMax)
// //         {
// //             RangeCheck = 0;
// //             break;
// //         }
// //     }
    
// //     return RangeCheck;
// // }




// //Find value for specified time, interpolating between breakpoints
// f64 breakpoint_Point_ValueAtTime(BREAKPOINT_FORMAT *ReadPoints, float64 TimeSpan, bool IsAmp)
// {
//     uint64 i;
//     BREAKPOINT_FORMAT TimeLeft, TimeRight;
//     float64 Fraction, Value, Width;

//     //Scan until span of specified time is found
//     for(i = 1; i < ReadPoints->CurrentPoints; i++)
//     {
//         if(TimeSpan <= ReadPoints->Time[i])
//         break;
//     }

//     if(IsAmp)
//     {
//         if(i == ReadPoints->CurrentPoints)
//         {
//             return ReadPoints->Amplitude[i-1];
//         }

//         TimeLeft.Time[0] = ReadPoints->Time[i-1];
//         TimeLeft.Amplitude[0] = ReadPoints->Amplitude[i-1];

//         TimeRight.Time[0] = ReadPoints->Time[i];
//         TimeRight.Amplitude[0] = ReadPoints->Amplitude[i];

//         //Check for instant jump, where two points have the same time
//         Width = TimeRight.Time[0] - TimeLeft.Time[0];

//         if(Width == 0.0)
//         {
//             return TimeRight.Amplitude[0];
//         }

//         //Get value from this span of times using linear interpolation
//         Fraction = (TimeSpan - TimeLeft.Time[0]) / Width;
//         Value = TimeLeft.Amplitude[0] + ((TimeRight.Amplitude[0] - TimeLeft.Amplitude[0]) * Fraction);

//         return Value;
//     }

//     else
//     {
//         if(i == ReadPoints->CurrentPoints)
//         {
//             return ReadPoints->Frequency[i-1];
//         }
    
//         TimeLeft.Time[0] = ReadPoints->Time[i-1];
//         TimeLeft.Frequency[0] = ReadPoints->Frequency[i-1];
    
//         TimeRight.Time[0] = ReadPoints->Time[i];
//         TimeRight.Frequency[0] = ReadPoints->Frequency[i];
    
//         //Check for instant jump, where two points have the same time
//         Width = TimeRight.Time[0] - TimeLeft.Time[0];
    
//         if(Width == 0.0)
//         {
//             return TimeRight.Frequency[0];
//         }
    
//         //Get value from this span of times using linear interpolation
//         Fraction = (TimeSpan - TimeLeft.Time[0]) / Width;
//         Value = TimeLeft.Frequency[0] + ((TimeRight.Frequency[0] - TimeLeft.Frequency[0]) * Fraction);
    
//         return Value;
//     }
// }


#endif