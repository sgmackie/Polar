
void SYS_ENVELOPE_ADSR::Create(MEMORY_ALLOCATOR *Allocator, size_t Size)
{
    SystemVoices = (ID_VOICE *) Allocator->Alloc((sizeof(ID_VOICE) * Size), HEAP_TAG_SYSTEM_ADSR);
    SystemCount = 0;
}

void SYS_ENVELOPE_ADSR::Destroy(MEMORY_ALLOCATOR *Allocator)
{
    Allocator->Free(0, HEAP_TAG_SYSTEM_ADSR);
}

void SYS_ENVELOPE_ADSR::Add(ID_VOICE ID)
{
    SystemVoices[SystemCount] = ID;
    ++SystemCount;
}

bool SYS_ENVELOPE_ADSR::Remove(ID_VOICE ID)
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

void SYS_ENVELOPE_ADSR::Edit(ENTITY_VOICES *Voices, ID_VOICE ID, f64 ControlRate, f64 MaxAmplitude, f64 Attack, f64 Decay, f64 SustainAmplitude, f64 Release, bool IsAligned)
{    
    //Loop through every source that was added to the system
    for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
    {
        //Find active sources in the system
        ID_VOICE Source = SystemVoices[SystemIndex];
        if(Source == ID)
        {
            //Source is valid - get component
            size_t Index                = Voices->RetrieveIndex(Source);
            CMP_ADSR &ADSR              = Voices->ADSRs[Index];

            //Amplitudes
            ADSR.MaxAmplitude           = MaxAmplitude;
            ADSR.SustainAmplitude       = SustainAmplitude;

            //Convert to sample counts for linear ramping
            ADSR.Attack                 = Attack   * ControlRate;
            ADSR.Decay                  = Decay    * ControlRate;
            ADSR.Release                = Release  * ControlRate;

            //Durations
            ADSR.Index                  = 0;
            ADSR.DurationInSamples      = ((Attack + Decay + Release) * ControlRate);
            if(IsAligned)
            {
                ADSR.DurationInSamples  = NearestPowerOf2(ADSR.DurationInSamples);
            }
            ADSR.IsActive               = true;                
        }
    }
}

void SYS_ENVELOPE_ADSR::RenderToBuffer(CMP_ADSR &ADSR, CMP_BUFFER &Buffer, size_t BufferCount)
{
    if(ADSR.IsActive)
    {
        f64 Sample = 0;
        for(size_t i = 0; i < BufferCount; ++i)
        {
            //ADSR finished
            if(ADSR.Index == ADSR.DurationInSamples)
            {
                ADSR.IsActive = false;
                return;
            }

            //Attack
            if(ADSR.Index <= ADSR.Attack)
            {
                Sample = ADSR.Index * (ADSR.MaxAmplitude / ADSR.Attack);
            }

            //Decay
            else if(ADSR.Index <= (ADSR.Attack + ADSR.Decay))
            {
                Sample = ((ADSR.SustainAmplitude - ADSR.MaxAmplitude) / ADSR.Decay) * (ADSR.Index - ADSR.Attack) + ADSR.MaxAmplitude;
            }

            //Sustain
            else if(ADSR.Index <= (ADSR.DurationInSamples - ADSR.Release))
            {
                Sample = ADSR.SustainAmplitude;
            }

            //Release
            else if(ADSR.Index > (ADSR.DurationInSamples - ADSR.Release))
            {
                Sample = -(ADSR.SustainAmplitude / ADSR.Release) * (ADSR.Index - (ADSR.DurationInSamples - ADSR.Release)) + ADSR.SustainAmplitude;
            }

            ADSR.Index++;

            Buffer.Data[i] *= Sample;
        }
    }
}

void SYS_ENVELOPE_ADSR::Update(ENTITY_VOICES *Voices, size_t BufferCount)
{
    //Loop through every source that was added to the system
    for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
    {
        //Find active sources in the system
        ID_VOICE Voice = SystemVoices[SystemIndex];
        if(Voice != 0)
        {
            //Source is valid - get component
            size_t Index = Voices->RetrieveIndex(Voice);
            RenderToBuffer(Voices->ADSRs[Index], Voices->Playbacks[Index].Buffer, BufferCount);
        }
    }
}
