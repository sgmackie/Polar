
void SYS_POSITION::Create(MEMORY_ALLOCATOR *Allocator, size_t Size)
{
    SystemVoices = (ID_VOICE *) Allocator->Alloc((sizeof(ID_VOICE) * Size), HEAP_TAG_SYSTEM_POSITION);
    SystemCount = 0;
}

void SYS_POSITION::Destroy(MEMORY_ALLOCATOR *Allocator)
{
    Allocator->Free(0, HEAP_TAG_SYSTEM_POSITION);
}

void SYS_POSITION::Add(ID_VOICE ID)
{
    SystemVoices[SystemCount] = ID;
    ++SystemCount;
}

bool SYS_POSITION::Remove(ID_VOICE ID)
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


void SYS_POSITION::Edit(ENTITY_VOICES *Voices, ID_VOICE ID, f64 MinDistance, f64 MaxDistance, f64 Rolloff)
{
    //Loop through every source that was added to the system
    for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
    {
        //Find active sources in the system
        ID_VOICE Voice = SystemVoices[SystemIndex];
        if(Voice == ID)
        {
            //Source is valid - get component
            size_t Index            = Voices->RetrieveIndex(Voice);
            CMP_DISTANCE &Distance  = Voices->Distances[Index];

            f64 OldMinDistance      = Distance.MinDistance;
            f64 OldMaxDistance      = Distance.MaxDistance;
            f64 OldRolloff          = Distance.Rolloff;

            if(OldMinDistance != MinDistance || OldMaxDistance != MaxDistance || OldRolloff != Rolloff)
            {
                Distance.RolloffDirty = true;
            }

            Distance.MinDistance    = MinDistance;
            Distance.MaxDistance    = MaxDistance;
            Distance.Rolloff        = Rolloff;
        }
    }
}    

f64 Dist(CMP_POSITION *A, CMP_POSITION *B) 
{
    return sqrt(pow((B->X - A->X), 2) + pow((B->Y - A->Y), 2) + pow((B->Z - A->Z), 2));
}

void SYS_POSITION::Update(ENTITY_VOICES *Voices, LISTENER *GlobalListener, f64 NoiseFloor)
{
    //Loop through every source that was added to the system
    for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
    {
        //Find active sources in the system
        ID_VOICE Voice = SystemVoices[SystemIndex];
        if(Voice != 0)
        {
            //Source is valid - get component
            size_t VoiceIndex = Voices->RetrieveIndex(Voice);
            CMP_POSITION &Position = Voices->Positions[VoiceIndex];
            CMP_FADE &Amplitude = Voices->Amplitudes[VoiceIndex];
            CMP_DISTANCE &Distance = Voices->Distances[VoiceIndex];


            if(Distance.RolloffDirty)
            {
                Distance.RolloffFactor = NoiseFloor / pow((Distance.MaxDistance - Distance.MinDistance), Distance.Rolloff);
                Distance.RolloffDirty = false;
            }

            Distance.FromListener = Dist(&Position, &GlobalListener->Position);

            if(Distance.FromListener > Distance.MinDistance)
            {
                Distance.Attenuation = (Distance.RolloffFactor * pow((Distance.FromListener - Distance.MinDistance), Distance.Rolloff));
                //TODO: Check normalisation
                Distance.Attenuation *= 100000;
                Distance.Attenuation = CLAMP(Distance.Attenuation, 0, 1.0);
            }

            else
            {
                Distance.Attenuation = 0.0;
            }

            Debug("Distance: %f Attenuation: %f", Distance.FromListener, Distance.Attenuation);

        }
    }
}
