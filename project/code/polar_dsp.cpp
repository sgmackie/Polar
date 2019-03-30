#ifndef polar_dsp_cpp
#define polar_dsp_cpp

//Allocation and initialisation functions in one
POLAR_OSCILLATOR *polar_dsp_OscillatorCreate(MEMORY_ARENA *Arena, u32 SampleRate, u32 WaveformSelect, f32 InitialFrequency)
{
    POLAR_OSCILLATOR *Result = 0;
    Result = (POLAR_OSCILLATOR *) memory_arena_Push(Arena, Result, (sizeof (POLAR_OSCILLATOR)));

    if(!Result)
    {
        return 0;
    }

    if(SampleRate != 0 || WaveformSelect != 0 || InitialFrequency != 0)
    {
        polar_dsp_OscillatorInit(Result, SampleRate, WaveformSelect, InitialFrequency);
    }

    return Result;
}

//Initialise elements of oscillator (can be used to reset)
void polar_dsp_OscillatorInit(POLAR_OSCILLATOR *Oscillator, u32 SampleRate, u32 WaveformSelect, f32 InitialFrequency)
{   
    Oscillator->Waveform = WaveformSelect;

    switch(Oscillator->Waveform)
    {
        case WV_SINE:
        {
            Oscillator->Tick = polar_dsp_TickSine;
            break;
        }
        case WV_SQUARE:
        {
            Oscillator->Tick = polar_dsp_TickSquare;
            break;
        }
        case WV_SAWDOWN:
        {
            Oscillator->Tick = polar_dsp_TickSawDown;
            break;
        }
        case WV_SAWUP:
        {
            Oscillator->Tick = polar_dsp_TickSawUp;
            break;
        }
        case WV_TRIANGLE:
        {
            Oscillator->Tick = polar_dsp_TickTriangle;
            break;
        }
        default:
        {
            Oscillator->Tick = polar_dsp_TickSine;
        }
    }

    Oscillator->TwoPiOverSampleRate = TWO_PI32 / SampleRate;
    
    if(InitialFrequency != 0)
    {
        Oscillator->Frequency.Current = InitialFrequency;
        Oscillator->Frequency.Target = 0;
        Oscillator->Frequency.Delta = 0;
    }

    else
    {
        Oscillator->Frequency.Current = 0;
        Oscillator->Frequency.Target = 0;
        Oscillator->Frequency.Delta = 0;
    }

    Oscillator->PhaseCurrent = 0;
    Oscillator->PhaseIncrement = 0;
}

//Wrap phase 2*PI32 as precaution against sin(x) function on different compilers failing to wrap large scale values internally
f32 polar_dsp_PhaseWrap(f32 &Phase, f64 Size)
{    
    while(Phase >= Size)
    {
        Phase -= Size;
    }

    while(Phase < 0)
    {
        Phase += Size;
    }

    return Phase;
}

//Calculate sine wave samples
f32 polar_dsp_TickSine(POLAR_OSCILLATOR *Oscillator)
{
    f32 SineValue;

#if CUDA
    SineValue = cuda_Sine(Oscillator->PhaseCurrent, 1);
#else
    SineValue = MiniMax(Oscillator->PhaseCurrent);
#endif

    Oscillator->PhaseIncrement = Oscillator->TwoPiOverSampleRate * Oscillator->Frequency.Current; 
    Oscillator->PhaseCurrent += Oscillator->PhaseIncrement; //Increase phase by the calculated cycle increment
    
    polar_dsp_PhaseWrap(Oscillator->PhaseCurrent, TWO_PI32);

    return SineValue;
}

//Calculate square wave samples
f32 polar_dsp_TickSquare(POLAR_OSCILLATOR *Oscillator)
{
    f32 SquareValue;
    
    Oscillator->PhaseIncrement = Oscillator->TwoPiOverSampleRate * Oscillator->Frequency.Current;
    
    if(Oscillator->PhaseCurrent <= PI32)
    {
        SquareValue = 1.0;
    }

    else
    {
        SquareValue = -1.0;
    }
    
    Oscillator->PhaseCurrent += Oscillator->PhaseIncrement;
    polar_dsp_PhaseWrap(Oscillator->PhaseCurrent, TWO_PI32);

    return SquareValue;
}

//Calculate downward square wave samples
f32 polar_dsp_TickSawDown(POLAR_OSCILLATOR *Oscillator)
{
    f32 SawDownValue;
    
    SawDownValue = ((-1.0) * (Oscillator->PhaseCurrent * (1.0 / TWO_PI32)));

    Oscillator->PhaseIncrement = Oscillator->TwoPiOverSampleRate * Oscillator->Frequency.Current;
    Oscillator->PhaseCurrent += Oscillator->PhaseIncrement;
    
    polar_dsp_PhaseWrap(Oscillator->PhaseCurrent, TWO_PI32);
    
    return SawDownValue;
}

//Calculate upward square wave samples
f32 polar_dsp_TickSawUp(POLAR_OSCILLATOR *Oscillator)
{
    f32 SawUpValue;
    
    SawUpValue = ((2.0 * (Oscillator->PhaseCurrent * (1.0 / TWO_PI32))) - 1.0);

    Oscillator->PhaseIncrement = Oscillator->TwoPiOverSampleRate * Oscillator->Frequency.Current;
    Oscillator->PhaseCurrent += Oscillator->PhaseIncrement;
    
    polar_dsp_PhaseWrap(Oscillator->PhaseCurrent, TWO_PI32);
    
    return SawUpValue;
}

//Calculate triangle wave samples
f32 polar_dsp_TickTriangle(POLAR_OSCILLATOR *Oscillator)
{
    f32 TriangleValue;
    
    TriangleValue = ((2.0 * (Oscillator->PhaseCurrent * (1.0 / TWO_PI32))) - 1.0);

    if(TriangleValue < 0.0)
    {
        TriangleValue = -TriangleValue;
    }
    
    TriangleValue = (2.0 * (TriangleValue - 0.5));

    Oscillator->PhaseIncrement = Oscillator->TwoPiOverSampleRate * Oscillator->Frequency.Current;
    Oscillator->PhaseCurrent += Oscillator->PhaseIncrement;
    
    polar_dsp_PhaseWrap(Oscillator->PhaseCurrent, TWO_PI32);
    
    return TriangleValue;
}


void FillTable(f64 *Table, u64 Length)
{
    f64 Step = TWO_PI32 / Length;

    u64 i = 0;
    for(i = 0; i < Length; ++i) 
    {
        Table[i] = MiniMax(Step * i);
    }

    //Wraparound
    Table[i] = Table[0];
}

void RenderTable(f32 *outframe, u32 Samples, f64 *Table, u64 Length, f64 Frequency, u32 SampleRate, bool IsTruncated)
{
    f64 curphase = 0.0;
    f64 tablen = (f64) Length;

    f64 sizeovrsr = (f64) Length / SampleRate;
    f64 phaseincr = sizeovrsr * Frequency;

    if(IsTruncated)
    {   
        // Truncated loop
        for(u64 i = 0; i < Samples; ++i)
        {
            i32 Index = (i32) curphase; //Truncation

            outframe[i] = (f32) Table[Index];
            // printf("Sample: %f\tTable: %f\n", outframe[i], Table[Index]);
            curphase += phaseincr;

            while(curphase >= tablen) 
            {
                curphase -= tablen;
            }

            while(curphase < 0)
            {
                curphase += tablen;
            }
        }
    }
    
    else
    {
        //Interpolated loop
        i32 Base = 0;
        i32 Next = 0;
        f64 Fraction = 0;
        f64 Value = 0;
        f64 Slope = 0;

        for(u64 i = 0; i < Samples; ++i)
        {
            Base = (i32) curphase;
            Next = (Base + 1);

            Fraction = curphase - Base;
            Value = Table[Base];
            Slope = (Table[Next] - Value);

            Value += (Fraction * Slope);
            outframe[i] = (f32) Value;
            // printf("Sample: %f\n", outframe[i]);
            curphase += phaseincr;

            while(curphase >= tablen) 
            {
                curphase -= tablen;
            }

            while(curphase < 0)
            {
                curphase += tablen;
            }
        }
    }
}

void TableTest()
{
    MEMORY_ARENA *TableMemory = memory_arena_Create(Kilobytes(100));
    
    u64 Length = (8192 + 1); //+1 for guard point at end of the table for wraparound
    f64 *Table = 0;
    Table = (f64 *) memory_arena_Push(TableMemory, Table, (sizeof(* Table) * Length));
    FillTable(Table, Length);

    u32 Samples = 4096;
    f32 *outframe = 0;
    outframe = (f32 *) memory_arena_Push(TableMemory, outframe, (sizeof(* outframe) * Samples));
    RenderTable(outframe, Samples, Table, Length, 440, 48000, false);

}





#endif