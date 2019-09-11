
void SYS_PLAY::Create(MEMORY_ALLOCATOR *Allocator, size_t Size)
{
    SystemVoices = (ID_VOICE *) Allocator->Alloc((sizeof(ID_VOICE) * Size), HEAP_TAG_SYSTEM_PLAY);
    SystemCount  = 0;
}

void SYS_PLAY::Destroy(MEMORY_ALLOCATOR *Allocator)
{
    Allocator->Free(0, HEAP_TAG_SYSTEM_PLAY);
}

void SYS_PLAY::Add(ID_VOICE ID)
{
    SystemVoices[SystemCount] = ID;
    ++SystemCount;        
}

bool SYS_PLAY::Remove(ID_VOICE ID)
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

bool SYS_PLAY::Start(ENTITY_VOICES *Voices, ID_VOICE ID, f64 InputDuration, i32 LoopCount, u32 Delay, bool IsAligned)
{
    for(size_t i = 0; i <= SystemCount; ++i)
    {
        if(SystemVoices[i] == ID)
        {
            size_t Index                    = Voices->RetrieveIndex(ID);
            CMP_DURATION &Duration          = Voices->Playbacks[Index].Duration;
            CMP_FORMAT &Format              = Voices->Playbacks[Index].Format;
            CMP_STATE &State                = Voices->States[Index];

            //If stopped then start playback
            if(State.Play == STOPPED)
            {
                Duration.SampleCount        = InputDuration * (Format.SampleRate);
                Duration.FrameDelay         = Delay;
                State.Play                  = PLAYING;
                State.LoopCount             = LoopCount;

                if(IsAligned)
                {
                    Duration.SampleCount    = NearestPowerOf2(Duration.SampleCount);
                }

                Duration.OriginalCount      = Duration.SampleCount;

                HANDLE_SOURCE Source        = Voices->Sources[Index];
                Info("Play: Source %llu | Voice %llu:\tDuration: %f | SampleCount: %llu | LoopCount: %d | FrameDelay: %lu", Source.ID, SystemVoices[i], InputDuration, Duration.SampleCount, State.LoopCount, Duration.FrameDelay);

                return true;
            }
        }
    }

    //!Log
    return false;        
}

void SYS_PLAY::Update(ENTITY_VOICES *Voices, f64 Time, u32 PreviousSamplesWritten, u32 SamplesToWrite)
{
    //Loop through every source that was added to the system
    for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
    {
        //Find active sources in the system
        ID_VOICE Voice = SystemVoices[SystemIndex];
        if(Voice != 0)
        {
            size_t Index			    = Voices->RetrieveIndex(Voice);
            CMP_STATE &State            = Voices->States[Index];
            CMP_DURATION &Duration      = Voices->Playbacks[Index].Duration;
            CMP_BUFFER &Buffer          = Voices->Playbacks[Index].Buffer;

            if(!Duration.FrameDelay)
            {
                if(State.Play != STOPPED)
                {
					State.Play = PLAYING;
                    if(Duration.SampleCount <= 0)
                    {
                        if(State.LoopCount == 0)
                        {
                            Duration.SampleCount = 0;
                            State.Play = STOPPED;
                            State.Voice = INACTIVE;
                            break;
                        }

                        if(State.LoopCount < 0)
                        {
                            //Infinite
                            Duration.SampleCount = Duration.OriginalCount;
                            break;
                        }

                        else
                        {
                            State.LoopCount -= 1;
                            if(State.LoopCount < 0)
                            {
                                State.LoopCount = 0;
                            }
                            Duration.SampleCount = Duration.OriginalCount;
                            break;
                        }
                    }

                    if(Duration.SampleCount <= PreviousSamplesWritten)
                    {
                        if(State.LoopCount == 0)
                        {
                            Duration.SampleCount = 0;
                            State.Play = STOPPED;
                            State.Voice = INACTIVE;
                            break;
                        }

						if(State.LoopCount < 0)
						{
							//Infinite
							Duration.SampleCount = Duration.OriginalCount;
							break;
						}

                        else
                        {
                            State.LoopCount -= 1;
                            if(State.LoopCount < 0)
                            {
                                State.LoopCount = 0;
                            }
                            Duration.SampleCount = Duration.OriginalCount;
                            break;
                        }

						break;
                    }

                    //If ending in the next callback, mark as stopping (fade out)
					if(Duration.SampleCount <= SamplesToWrite)
                    {
                        Duration.SampleCount = SamplesToWrite;
                        State.Play = STOPPING;
						break;
                    }

			    	else
                    {
                        Duration.SampleCount -= PreviousSamplesWritten;

                        HANDLE_SOURCE Source = Voices->Sources[Index];
                        Debug("Play: Source %llu | Voice %llu:\tSampleCount: %llu\tLoopCount: %d", Source.ID, Voice, Duration.SampleCount, State.LoopCount);
                    }
                }
            }
            else if(Duration.FrameDelay < 0)
            {
				State.Play = PLAYING;
                Duration.FrameDelay = 0;
            }
            else
            {
                State.Play = PAUSED;
                --Duration.FrameDelay;
            }
        }
    }      
}