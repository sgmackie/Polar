

void SYS_VOICES::Create(MEMORY_ALLOCATOR *Allocator, size_t Size)
{
    SystemSources = (ID_SOURCE *) Allocator->Alloc((sizeof(ID_SOURCE) * Size), HEAP_TAG_SYSTEM_VOICE);
    SystemCount = 0;
}

void SYS_VOICES::Destroy(MEMORY_ALLOCATOR *Allocator)
{
    Allocator->Free(0, HEAP_TAG_SYSTEM_VOICE);
}

void SYS_VOICES::Add(ID_SOURCE ID)
{
    SystemSources[SystemCount] = ID;
    ++SystemCount;
}

bool SYS_VOICES::Remove(ID_SOURCE ID)
{
    for(size_t i = 0; i <= SystemCount; ++i)
    {
        if(SystemSources[i] == ID)
        {
            SystemSources[i] = 0;
            --SystemCount;
            return true;
        }
    }
    //!Log
    return false;
}

ID_VOICE SYS_VOICES::Spawn(ENTITY_SOURCES *Sources, ENTITY_VOICES *Voices, ID_SOURCE ID, f32 SampleRate, u32 Random, size_t DeferCounter, MDL_SYSTEMS *Systems, POLAR_POOL *Pools)
{
    //Loop through every source that was added to the system
    for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
    {
        if(SystemSources[SystemIndex] == ID)
        {
            HANDLE_SOURCE Source        = {ID, Sources->RetrieveIndex(ID)};
            ID_VOICE Voice              = Voices->Add(Source);
            size_t Index                = Voices->RetrieveIndex(Voice);
            Voices->States[Index].Voice = SPAWNING;

            if(Sources->Voices[Source.Index].Count != MAX_VOICES_PER_SOURCE)
            {
                size_t i = Sources->Voices[Source.Index].Count;
                Sources->Voices[Source.Index].Handles[i].ID    = Voice;
                Sources->Voices[Source.Index].Handles[i].Index = Index;
                ++Sources->Voices[Source.Index].Count;

                //Defering until next update - return now
                if(DeferCounter)
                {
                    return Voice;
                }

                //Not defering - add to systems instantly
                Voices->States[Index].Voice = ACTIVE;
                Voices->Playbacks[Index].Buffer.CreateFromPool(&Pools->Buffers, MAX_BUFFER_SIZE);
                Voices->Playbacks[Index].Format = Sources->Formats[Source.Index];
                Voices->Flags[Index] = Sources->Flags[Source.Index];
                Systems->Play.Add(Voice);
                Systems->Mix.Add(Voice);

                if(Voices->Flags[Index] & ENTITY_SOURCES::POSITION)
                {
                    Voices->Positions[Index] = Sources->Positions[Source.Index];
                    Voices->Distances[Index] = Sources->Distances[Source.Index];
                    Systems->Position.Add(Voice);
                }

                if(Voices->Flags[Index] & ENTITY_SOURCES::AMPLITUDE)
                {
                    Voices->Amplitudes[Index] = Sources->Amplitudes[Source.Index];
                    Systems->Fade.Add(Voice);
                }

                if(Voices->Flags[Index] & ENTITY_SOURCES::CROSSFADE)
                {
                    Voices->Crossfades[Index] = Sources->Crossfades[Source.Index];
                    Systems->Crossfade.Add(Voice);
                }

                if(Voices->Flags[Index] & ENTITY_SOURCES::PAN)
                {
                    Voices->Pans[Index] = Sources->Pans[Source.Index];
                }

                if(Voices->Flags[Index] & ENTITY_SOURCES::BREAKPOINT)
                {
                    Voices->Breakpoints[Index].CreateFromPool(&Pools->Breakpoints, MAX_BREAKPOINTS);
                    Systems->Breakpoint.Add(Voice);
                }

                if(Voices->Flags[Index] & ENTITY_SOURCES::ADSR)
                {
                    Voices->ADSRs[Index] = Sources->ADSRs[Source.Index];
                    Systems->ADSR.Add(Voice);
                }

                if(Voices->Flags[Index] & ENTITY_SOURCES::BIQUAD)
                {
                    Voices->Filters[Index] = Sources->Filters[Source.Index];
                    Systems->Filter.Add(Voice);
                }

                if(Voices->Flags[Index] & ENTITY_SOURCES::CUDA_SINE)
                {
                    Voices->Sines[Index].CreateFromPool(&Pools->Partials);
                    Systems->Cuda.Add(Voice);
                }

                if(Voices->Flags[Index] & ENTITY_SOURCES::CUDA_BUBBLE)
                {
					Voices->Bubbles[Index] = Sources->Bubbles[Source.Index];
                    Voices->Bubbles[Index].CreateFromPool(&Pools->Bubble.Generators, &Pools->Bubble.Radii, &Pools->Bubble.Lambda);

                    for(size_t i = 0; i < Voices->Bubbles[Index].Count; ++i)
                    {
                        Voices->Bubbles[Index].Generators[i].Model.Init();
                        Voices->Bubbles[Index].Generators[i].Pulse.Init(SampleRate, Random);
                    }                    

                    Systems->Bubbles.Add(Voice);
                }

                if(Voices->Flags[Index] & ENTITY_SOURCES::GRAIN)
                {
                    Voices->Grains[Index] = Sources->Grains[Source.Index];
                    Systems->Grain.Add(Voice);
                }

                if(Voices->Flags[Index] & ENTITY_SOURCES::FFT)
                {
                    Voices->FFTs[Index] = Sources->FFTs[Source.Index];
                    Systems->FFT.Add(Voice);
                }

                if(Voices->Flags[Index] & ENTITY_SOURCES::OSCILLATOR)
                {
                    Voices->Types[Index].Oscillator = Sources->Types[Source.Index].Oscillator;
                    switch(Voices->Types[Index].Oscillator.Flag)
                    {
                        case CMP_OSCILLATOR::SINE:
                        {
                            Systems->Oscillator.Sine.Add(Voice);
                            break;
                        }
                        case CMP_OSCILLATOR::SQUARE:
                        {
                            Systems->Oscillator.Square.Add(Voice);
                            break;
                        }       
                        case CMP_OSCILLATOR::TRIANGLE:
                        {
                            Systems->Oscillator.Triangle.Add(Voice);
                            break;
                        }      
                        case CMP_OSCILLATOR::SAWTOOTH:
                        {
                            Systems->Oscillator.Sawtooth.Add(Voice);
                            break;
                        }                                                               
                    }
                }

                if(Voices->Flags[Index] & ENTITY_SOURCES::NOISE)
                {
                    Voices->Types[Index].Noise = Sources->Types[Source.Index].Noise;
                    switch(Voices->Types[Index].Noise.Flag)
                    {
                        case CMP_NOISE::WHITE:
                        {
                            Systems->Noise.White.Add(Voice);
                            break;
                        }
                        case CMP_NOISE::BROWN:
                        {
                            Systems->Noise.Brown.Add(Voice);
                            break;
                        }                                                                   
                    }
                }


                if(Voices->Flags[Index] & ENTITY_SOURCES::WAV)
                {
                    Voices->Types[Index].WAV = Sources->Types[Source.Index].WAV;
                    Systems->WAV.Add(Voice);
                }

                Info("Voice: Source %llu: Spawned voice: %zu of %d", Source.ID, Sources->Voices[Source.Index].Count, MAX_VOICES);

                return Voice;
            }

            Error("Voices: Max count reached! Source: %llu", Source.ID);
        }
    }

    return -1;
}

void SYS_VOICES::Update(ENTITY_SOURCES *Sources, ENTITY_VOICES *Voices, MDL_SYSTEMS *Systems, POLAR_POOL *Pools)
{
    //Loop through every source that was added to the system
    for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
    {
        //Find active sources in the system
        ID_SOURCE Source = SystemSources[SystemIndex];
        if(Source != 0)
        {
            //Source is valid - get voice
            size_t SourceIndex  = Sources->RetrieveIndex(Source);
            CMP_VOICEMAP &Map   = Sources->Voices[SourceIndex];
            for(size_t i = 0; i < Map.Count; ++i)
            {
                HANDLE_VOICE Voice  = Map.Handles[i];
                CMP_STATE &State    = Voices->States[Voice.Index];
                i32 &Flags          = Voices->Flags[Voice.Index];

                switch(State.Voice)
                {
                    case SPAWNING:
                    {
                        State.Voice = ACTIVE;
                        Voices->Playbacks[Voice.Index].Buffer.CreateFromPool(&Pools->Buffers, MAX_BUFFER_SIZE);
                        Systems->Play.Add(Voice.ID);
                        Systems->Mix.Add(Voice.ID);

                        Info("Voice: Source %llu: Spawned voice: %zu of %d", Source, (i + 1), MAX_VOICES);

                        break;
                    }
                    case ACTIVE:
                    {
                        break;
                    }  
                    case INACTIVE:
                    {
                        Systems->Play.Remove(Voice.ID);
                        Systems->Mix.Remove(Voice.ID);

                        if(Flags & ENTITY_SOURCES::POSITION)
                        {
                            Systems->Position.Remove(Voice.ID);
                        }

                        if(Flags & ENTITY_SOURCES::AMPLITUDE)
                        {
                            Systems->Fade.Remove(Voice.ID);
                        }

                        if(Flags & ENTITY_SOURCES::CROSSFADE)
                        {
                            Systems->Crossfade.Remove(Voice.ID);
                        }

                        if(Flags & ENTITY_SOURCES::BREAKPOINT)
                        {
                            Systems->Breakpoint.Remove(Voice.ID);
                            Voices->Breakpoints[Voice.Index].FreeFromPool(&Pools->Breakpoints);
                        }

                        if(Flags & ENTITY_SOURCES::ADSR)
                        {
                            Systems->ADSR.Remove(Voice.ID);
                        }

                        if(Flags & ENTITY_SOURCES::BIQUAD)
                        {
                            Systems->Filter.Remove(Voice.ID);
                        }

                        if(Flags & ENTITY_SOURCES::CUDA_SINE)
                        {
                            Systems->Cuda.Remove(Voice.ID);
                            for(size_t i = 0; i < Voices->Sines[Voice.Index].Count; ++i)
                            {
                                Voices->Sines[Voice.Index].Partials->FreeFromPool(&Pools->Phases);
                            }
                            Voices->Sines[Voice.Index].Count = 0;
                            Voices->Sines[Voice.Index].FreeFromPool(&Pools->Partials);
                        }

                        if(Flags & ENTITY_SOURCES::CUDA_BUBBLE)
                        {
                            Systems->Bubbles.Remove(Voice.ID);

                            Voices->Bubbles[Voice.Index].FreeFromPool(&Pools->Bubble.Generators, &Pools->Bubble.Radii, &Pools->Bubble.Lambda);
                        }

                        if(Flags & ENTITY_SOURCES::GRAIN)
                        {
                            Systems->Grain.Remove(Voice.ID);
                        }

                        if(Flags & ENTITY_SOURCES::FFT)
                        {
                            Systems->FFT.Remove(Voice.ID);
                        }

                        if(Flags & ENTITY_SOURCES::OSCILLATOR)
                        {
                            switch(Voices->Types[Voice.Index].Oscillator.Flag)
                            {
                                case CMP_OSCILLATOR::SINE:
                                {
                                    Systems->Oscillator.Sine.Remove(Voice.ID);
                                    break;
                                }
                                case CMP_OSCILLATOR::SQUARE:
                                {
                                    Systems->Oscillator.Square.Remove(Voice.ID);
                                    break;
                                }       
                                case CMP_OSCILLATOR::TRIANGLE:
                                {
                                    Systems->Oscillator.Triangle.Remove(Voice.ID);
                                    break;
                                }      
                                case CMP_OSCILLATOR::SAWTOOTH:
                                {
                                    Systems->Oscillator.Sawtooth.Remove(Voice.ID);
                                    break;
                                }                                                               
                            }
                        }

                        if(Flags & ENTITY_SOURCES::NOISE)
                        {
                            switch(Voices->Types[Voice.Index].Noise.Flag)
                            {
                                case CMP_NOISE::WHITE:
                                {
                                    Systems->Noise.White.Remove(Voice.ID);
                                    break;
                                }
                                case CMP_NOISE::BROWN:
                                {
                                    Systems->Noise.Brown.Remove(Voice.ID);
                                    break;
                                }                                                                   
                            }
                        }

                        if(Flags & ENTITY_SOURCES::WAV)
                        {
                            Systems->WAV.Remove(Voice.ID);
                        }

                        // Pools
                        Voices->Playbacks[Voice.Index].Buffer.FreeFromPool(&Pools->Buffers);

                        --Map.Count;
                        Voices->Remove(Voice.ID);

                        break;
                    }
                }
            }
        }
    }
}
