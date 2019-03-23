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
f32 polar_dsp_PhaseWrap(f32 &Phase)
{    
    if(Phase >= TWO_PI32)
    {
        Phase -= TWO_PI32;
    }

    if(Phase < 0)
    {
        Phase += TWO_PI32;
    }

    return Phase;
}


//Calculate sine wave samples
f32 polar_dsp_TickSine(POLAR_OSCILLATOR *Oscillator)
{
    f32 SineValue;

#if CUDA
    SineValue = (f32) MiniMax(Oscillator->PhaseCurrent);
#else
    SineValue = (f32) sin(Oscillator->PhaseCurrent);
#endif
    Oscillator->PhaseIncrement = Oscillator->TwoPiOverSampleRate * Oscillator->Frequency.Current; //Load atomic value, multiply to get the phase increment
    Oscillator->PhaseCurrent += Oscillator->PhaseIncrement; //Increase phase by the calculated cycle increment
    
    polar_dsp_PhaseWrap(Oscillator->PhaseCurrent);

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
    polar_dsp_PhaseWrap(Oscillator->PhaseCurrent);

    return SquareValue;
}

//Calculate downward square wave samples
f32 polar_dsp_TickSawDown(POLAR_OSCILLATOR *Oscillator)
{
    f32 SawDownValue;
    
    SawDownValue = ((-1.0) * (Oscillator->PhaseCurrent * (1.0 / TWO_PI32)));

    Oscillator->PhaseIncrement = Oscillator->TwoPiOverSampleRate * Oscillator->Frequency.Current;
    Oscillator->PhaseCurrent += Oscillator->PhaseIncrement;
    
    polar_dsp_PhaseWrap(Oscillator->PhaseCurrent);
    
    return SawDownValue;
}

//Calculate upward square wave samples
f32 polar_dsp_TickSawUp(POLAR_OSCILLATOR *Oscillator)
{
    f32 SawUpValue;
    
    SawUpValue = ((2.0 * (Oscillator->PhaseCurrent * (1.0 / TWO_PI32))) - 1.0);

    Oscillator->PhaseIncrement = Oscillator->TwoPiOverSampleRate * Oscillator->Frequency.Current;
    Oscillator->PhaseCurrent += Oscillator->PhaseIncrement;
    
    polar_dsp_PhaseWrap(Oscillator->PhaseCurrent);
    
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
    
    polar_dsp_PhaseWrap(Oscillator->PhaseCurrent);
    
    return TriangleValue;
}

#endif