#ifndef polar_dsp_cpp
#define polar_dsp_cpp


//Allocation and initialisation functions in one
OSCILLATOR *entropy_wave_OscillatorCreate(u32 SampleRate, WAVEFORM WaveformSelect, f64 InitialFrequency)
{
    //TODO: Look into creating own allocator
    OSCILLATOR *Oscillator = (OSCILLATOR *) malloc(sizeof *Oscillator);

    if(!Oscillator)
    {
        return nullptr;
    }

    if(SampleRate != 0 || WaveformSelect != 0 || InitialFrequency != 0)
    {
        entropy_wave_OscillatorInit(Oscillator, SampleRate, WaveformSelect, InitialFrequency);
    }

    return Oscillator;
}

//Free oscillator struct
void entropy_wave_OscillatorDestroy(OSCILLATOR *Oscillator)
{
    if(Oscillator)
    {
        free(Oscillator);
        Oscillator = nullptr;
    }
}

//Initialise elements of oscillator (can be used to reset)
void entropy_wave_OscillatorInit(OSCILLATOR *Oscillator, u32 SampleRate, WAVEFORM WaveformSelect, f64 InitialFrequency)
{   
    Oscillator->Waveform = WaveformSelect;

    switch(Oscillator->Waveform)
    {
        case SINE:
        {
            Oscillator->Tick = entropy_wave_TickSine;
            break;
        }
        case SQUARE:
        {
            Oscillator->Tick = entropy_wave_TickSquare;
            break;
        }
        case SAWDOWN:
        {
            Oscillator->Tick = entropy_wave_TickSawDown;
            break;
        }
        case SAWUP:
        {
            Oscillator->Tick = entropy_wave_TickSawUp;
            break;
        }
        case TRIANGLE:
        {
            Oscillator->Tick = entropy_wave_TickTriangle;
            break;
        }
        default:
        {
            Oscillator->Tick = entropy_wave_TickSine;
        }
    }

    Oscillator->TwoPiOverSampleRate = TWO_PI32 / SampleRate;
    
    if(InitialFrequency != 0)
    {
        Oscillator->FrequencyCurrent = InitialFrequency;
    }

    else
    {
        Oscillator->FrequencyCurrent = 0;
    }

    Oscillator->PhaseCurrent = 0;
    Oscillator->PhaseIncrement = 0;
}

//Wrap phase 2*Pi as precaution against sin(x) function on different compilers failing to wrap large scale values internally
f64 entropy_wave_PhaseWrap(f64 &Phase)
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
f64 entropy_wave_TickSine(OSCILLATOR *Oscillator)
{
    f64 SineValue;

    //TODO: Replace crt sin function?
    SineValue = (f64) sin(Oscillator->PhaseCurrent); //Input in radians
    
    Oscillator->PhaseIncrement = Oscillator->TwoPiOverSampleRate * Oscillator->FrequencyCurrent; //Load atomic value, multiply to get the phase increment
    Oscillator->PhaseCurrent += Oscillator->PhaseIncrement; //Increase phase by the calculated cycle increment
    
    entropy_wave_PhaseWrap(Oscillator->PhaseCurrent);

    return SineValue;
}

//Calculate square wave samples
f64 entropy_wave_TickSquare(OSCILLATOR *Oscillator)
{
    f64 SquareValue;
    
    Oscillator->PhaseIncrement = Oscillator->TwoPiOverSampleRate * Oscillator->FrequencyCurrent;
    
    if(Oscillator->PhaseCurrent <= PI32)
    {
        SquareValue = 1.0;
    }

    else
    {
        SquareValue = -1.0;
    }
    
    Oscillator->PhaseCurrent += Oscillator->PhaseIncrement;
    entropy_wave_PhaseWrap(Oscillator->PhaseCurrent);

    return SquareValue;
}

//Calculate downward square wave samples
f64 entropy_wave_TickSawDown(OSCILLATOR *Oscillator)
{
    f64 SawDownValue;
    
    SawDownValue = ((-1.0) * (Oscillator->PhaseCurrent * (1.0 / TWO_PI32)));

    Oscillator->PhaseIncrement = Oscillator->TwoPiOverSampleRate * Oscillator->FrequencyCurrent;
    Oscillator->PhaseCurrent += Oscillator->PhaseIncrement;
    
    entropy_wave_PhaseWrap(Oscillator->PhaseCurrent);
    
    return SawDownValue;
}

//Calculate upward square wave samples
f64 entropy_wave_TickSawUp(OSCILLATOR *Oscillator)
{
    f64 SawUpValue;
    
    SawUpValue = ((2.0 * (Oscillator->PhaseCurrent * (1.0 / TWO_PI32))) - 1.0);

    Oscillator->PhaseIncrement = Oscillator->TwoPiOverSampleRate * Oscillator->FrequencyCurrent;
    Oscillator->PhaseCurrent += Oscillator->PhaseIncrement;
    
    entropy_wave_PhaseWrap(Oscillator->PhaseCurrent);
    
    return SawUpValue;
}

//Calculate triangle wave samples
f64 entropy_wave_TickTriangle(OSCILLATOR *Oscillator)
{
    f64 TriangleValue;
    
    TriangleValue = ((2.0 * (Oscillator->PhaseCurrent * (1.0 / TWO_PI32))) - 1.0);

    if(TriangleValue < 0.0)
    {
        TriangleValue = -TriangleValue;
    }
    
    TriangleValue = (2.0 * (TriangleValue - 0.5));

    Oscillator->PhaseIncrement = Oscillator->TwoPiOverSampleRate * Oscillator->FrequencyCurrent;
    Oscillator->PhaseCurrent += Oscillator->PhaseIncrement;
    
    entropy_wave_PhaseWrap(Oscillator->PhaseCurrent);
    
    return TriangleValue;
}

#endif