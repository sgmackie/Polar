

void polar_dsp_PhaseWrap(f64 &Phasor, f64 Limit)
{
    while(Phasor >= Limit)
    {
        Phasor -= Limit;
    }
    while(Phasor < 0)
    {
        Phasor += Limit;
    }  
}