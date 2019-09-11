

void SYS_GRAIN::Create(MEMORY_ALLOCATOR *Allocator, size_t Size)
{
    SystemVoices = (ID_VOICE *) Allocator->Alloc((sizeof(ID_VOICE) * Size), HEAP_TAG_SYSTEM_GRAIN);
    SystemCount = 0;
}

void SYS_GRAIN::Destroy(MEMORY_ALLOCATOR *Allocator)
{
    Allocator->Free(0, HEAP_TAG_SYSTEM_GRAIN);
}

void SYS_GRAIN::Add(ID_VOICE ID)
{
    SystemVoices[SystemCount] = ID;
    ++SystemCount;
}

bool SYS_GRAIN::Remove(ID_VOICE ID)
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

// Compute trapeziod envelope to apply to each grain
void SYS_GRAIN::ComputeEnvelope(f32 *Envelope, f32 Attack, f32 Release, i32 ControlRate, size_t Length)
{
    // Compute sample counts
    size_t AttackSamples = round((Attack / 1000.0f) * ControlRate);
    size_t ReleaseSamples = round((Release / 1000.0f) * ControlRate);

    // Sustain
    for(size_t i = 0; i < Length; ++i)
    {
        Envelope[i] = 1.0f;
    }

    // Attack
    for(size_t i = 0; i < AttackSamples; ++i)
    {
        Envelope[i] = i * (1.0f / AttackSamples);
    }

    // Release
    for(size_t i = Length - ReleaseSamples + 1; i < Length; ++i)
    {
        Envelope[i] = -(1.0f / ReleaseSamples) * (i - (Length - ReleaseSamples)) + 1.0f;
    }
}

// Compute the grains from a given WAV file
void SYS_GRAIN::Compute(ENTITY_VOICES *Voices, ID_VOICE ID, i32 SampleRate, RANDOM_PCG *RNG, f32 Density, f32 LengthInMS, f32 Delay, f32 Attack, f32 Release, size_t CrossfadeDuration)
{
    //Loop through every source that was added to the system
    for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
    {
        //Find active sources in the system
        ID_VOICE Voice = SystemVoices[SystemIndex];
        if(Voice == ID)
        {
            //Source is valid - get component
            size_t Index                    = Voices->RetrieveIndex(Voice);
            CMP_GRAIN &Grain                = Voices->Grains[Index];
            CMP_WAV &WAV                    = Voices->Types[Index].WAV;  

            // Calculate the density bounds
            f32 MinimumDensity              = round(SampleRate / 150);
            f32 MaximumDensity              = round(SampleRate / 0.25);

            // Get the density in sample counts
            size_t DensityInSamples         = floor(SampleRate / Density);
            Assert(DensityInSamples <= MaximumDensity, "Grain: Grain density exceeds the maximum bounds!");
            Assert(DensityInSamples >= MinimumDensity, "Grain: Grain density below the minimum bounds!");

            f32 Average                     = (DensityInSamples - MinimumDensity) * 1;
            MinimumDensity                  = MAX(DensityInSamples - Average, MinimumDensity);
            MaximumDensity                  = MIN(DensityInSamples + Average, MaximumDensity);
            Assert(MinimumDensity <= MaximumDensity, "Grain: Grain minimum density exceeds the maximum!");

            // Find the grain length in samples
            u64 StartPoint					= 1;
            size_t LengthInSamples			= round((LengthInMS / 1000) * SampleRate);
            Assert(LengthInSamples < MAX_GRAIN_LENGTH, "Grain: Grain length is too high!");
            
            // Get delay spread samples
            size_t DelayInSamples			= round((Delay / 1000) * SampleRate);

            // Generate envelope to apply to each grain
            ComputeEnvelope(Grain.Envelope, Attack, Release, SampleRate, LengthInSamples);

            // Loop through WAV building each grain
            while((StartPoint + LengthInSamples - 1) <= WAV.Length)
            {
                // Get the index for reading from the WAV buffer
                size_t SourceIndex              = MAX(1, (StartPoint - round(0 * DelayInSamples * RandomF32(RNG))));
				
                // Read rom the WAV buffer
                for(size_t i = 0; i < LengthInSamples - 1; ++i)
                {
                    f32 Sample                  = WAV.Data[i + SourceIndex] * Grain.Envelope[i];
                    Grain.Data[Grain.Count][i]  = Sample;
                }

                // Find the next random grain start point
                StartPoint                      = StartPoint + MinimumDensity + round(RandomF32(RNG) * (MaximumDensity - MinimumDensity));
                ++Grain.Count;
            }

            // Initialise crossfade between each grain
            Grain.Crossfade.Init(0.0f, 1.0f, CrossfadeDuration);
        }
    }
}    

// Randomly shuffle the content of an array
void SYS_GRAIN::ArrayShuffle(size_t *Array, size_t Count)
{
    if(Count > 1) 
    {
        for(size_t i = 0; i < Count - 1; i++) 
        {
            size_t j        = i + rand() / (RAND_MAX / (Count - i) + 1);
            size_t Temp     = Array[j];
            Array[j]        = Array[i];
            Array[i]        = Temp;
        }
    }
}

void SYS_GRAIN::RenderToBuffer(CMP_GRAIN &Grain, CMP_BUFFER &Buffer, size_t BufferCount, size_t GrainSelector)
{
    // Check if building the playlist
    if(Grain.ListReader == 0 || Grain.ListReader)
    {
        // Save the original selection
        size_t MidPoint = GrainSelector;

        // Bounds check
        if(GrainSelector > Grain.Count)
        {
            GrainSelector = Grain.Count;
        }
        if((GrainSelector + 3) >= Grain.Count)
        {
            GrainSelector -= 3;
        }
        if((GrainSelector - 3) <= 0)
        {
            GrainSelector += 3;
        }
        if(GrainSelector == 0)
        {
            GrainSelector += 3;
        }

        // Find the shuffle limits
        size_t LowerBound = GrainSelector - 2;
        size_t UpperBound = GrainSelector + 2;

        // Build the playlist
        Grain.Playlist[0] = LowerBound;
        Grain.Playlist[1] = LowerBound + 1;
        Grain.Playlist[2] = MidPoint;
        Grain.Playlist[3] = UpperBound - 1;
        Grain.Playlist[4] = UpperBound;
        Grain.Playlist[5] = MidPoint;

        // Shuffle
        ArrayShuffle(Grain.Playlist, (MAX_GRAIN_PLAYLIST - 1));
    }

    // Copy grain data to voice buffer
    size_t GrainToPlay = Grain.Playlist[Grain.ListReader];
    for(size_t i = 0; i < BufferCount; ++i)
    {
        f32 Sample      = 0;
        Sample          = Grain.Data[GrainToPlay][i];
        Buffer.Data[i]  = Sample;
    }

    // Crossfade with using a ramp
    for(size_t i = (BufferCount - Grain.Crossfade.DurationInSamples); i < BufferCount; ++i)
    {
        size_t j        = 0;
        f32 Fade        = 0;
        Fade            = Grain.Crossfade.Generate(true, false, false);
        f32 Sample      = 0;
        Sample          = Grain.Data[GrainToPlay][i] * Fade + Grain.Data[GrainToPlay+1][j] * (1 -Fade);
        Buffer.Data[i]  = Sample;
        ++j;
    }        

    // Reset ramp generator
    Grain.Crossfade.Reset();

    // Iterate the playlist
    ++Grain.ListReader;
    if(Grain.ListReader >= (MAX_GRAIN_PLAYLIST - 1))
    {
        Grain.ListReader = 0;
    }
}

void SYS_GRAIN::Update(ENTITY_VOICES *Voices, size_t BufferCount, size_t GrainSelector)
{
    //Loop through every source that was added to the system
    for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
    {
        //Find active sources in the system
        ID_VOICE Voice = SystemVoices[SystemIndex];
        if(Voice != 0)
        {
            //Source is valid - get component
            size_t VoiceIndex       = Voices->RetrieveIndex(Voice);
            RenderToBuffer(Voices->Grains[VoiceIndex], Voices->Playbacks[VoiceIndex].Buffer, BufferCount, GrainSelector);

        }
    }
}