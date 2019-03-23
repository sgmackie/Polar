#ifndef polar_render_cpp
#define polar_render_cpp


void polar_render_Source(u32 &SampleRate, u64 &SampleCount, u32 SamplesToWrite, POLAR_SOURCE_TYPE &Type, u32 &FX, f32 *Buffer)
{
    while(SampleCount != 0 && Type.Flag != SO_NONE)
    {
        switch(Type.Flag)
        {
            case SO_FILE:
            {
                u64 Position = 0;
                u64 SamplesRemaining = (Type.File->FrameCount - Type.File->ReadIndex);

                if(SamplesRemaining < SamplesToWrite)
                {
                    SamplesToWrite = SamplesRemaining;
                }

                for(u32 FrameIndex = 0; FrameIndex < SamplesToWrite; ++FrameIndex)
	            {
                    Position = (Type.File->ReadIndex += Type.File->Channels);
	            	Buffer[FrameIndex] = Type.File->Samples[Position];
                }

                break;
            }

            case SO_OSCILLATOR:
            {
                double SecondsPerSample = (1.0f / (double) SampleRate);
        
                float CurrentFreq = Type.Oscillator->Frequency.Current;
                float FreqDelta = (SecondsPerSample * Type.Oscillator->Frequency.Delta);
                bool FreqEnded = false;
        
                if(FreqDelta != 0.0f)
                {
                    double NewDeltaFreq = (Type.Oscillator->Frequency.Target - CurrentFreq);
                    int FreqSampleCount = (int)((NewDeltaFreq / FreqDelta + 0.5f));
                        
                    if(SamplesToWrite > FreqSampleCount)
                    {
                        SamplesToWrite = FreqSampleCount;
                        FreqEnded = true;
                    }
        
                    if(FreqDelta == 0)
                    {
                        FreqEnded = true;
                    }
                }
        
                for(u32 FrameIndex = 0; FrameIndex < SamplesToWrite; ++FrameIndex)
	            {
                    Buffer[FrameIndex] = Type.Oscillator->Tick(Type.Oscillator);
                    CurrentFreq += FreqDelta;
                    Type.Oscillator->Frequency.Current = CurrentFreq;
                    // printf("Current: %f\tTarget: %f\t: Interp: %f\n", Current, Target, CurrentFreq);
                }
        
                Type.Oscillator->Frequency.Current = CurrentFreq;
        
                if(FreqEnded)
                {
                    Type.Oscillator->Frequency.Current = Type.Oscillator->Frequency.Target;
                    Type.Oscillator->Frequency.Delta = 0.0f;
                }                

                break;
            }
            
            default:
            {
                break;
            }
        }

        //FX is a bitmask, &'ing with the FX defines to find which ones to process
        if(FX != FX_DRY)
        {
            if(FX & FX_AM)
            {
                EffectAM(SamplesToWrite, SampleRate, Buffer, 100);
            }

            if(FX & FX_ECHO)
            {
                EffectEcho(SamplesToWrite, SampleRate, Buffer, 3);
            }
        }

        SampleCount -= SamplesToWrite;
        
        return;  
    }
}



void polar_render_MixSources(f32 &AmplitudeCurrent, f32 &AmplitudePrevious, f32 *Buffer, POLAR_BUFFER *MixBuffer)
{
    f32 TargetAmplitude = AmplitudeCurrent;

    if(fabsf(AmplitudePrevious - TargetAmplitude) < 0.1e-5)
    {
        for(u32 FrameIndex = 0; FrameIndex < MixBuffer->SampleCount; ++FrameIndex)
        {
            MixBuffer->Data[FrameIndex] += ((Buffer[FrameIndex]) * TargetAmplitude);
        }
    }

    else
    {
        f32 Current = AmplitudePrevious;
        f32 Step = (TargetAmplitude - Current) * 1.0f / MixBuffer->SampleCount;

        for(u32 FrameIndex = 0; FrameIndex < MixBuffer->SampleCount; ++FrameIndex)
        {
            Current += Step;
            MixBuffer->Data[FrameIndex] += ((Buffer[FrameIndex]) * Current);
        }

        AmplitudePrevious = TargetAmplitude;
    }
}


void polar_render_Container(POLAR_SOURCE &ContainerSources, f64 ContainerAmplitude, POLAR_BUFFER *MixBuffer)
{
    for(u8 i = 0; i <= ContainerSources.CurrentSources; ++i)
    {
        if(ContainerSources.SampleCount[i] <= 0)
        {
            ContainerSources.PlayState[i] = Stopped;
        }

        //Set the source to stop if it ends within the next two callbacks
        else if(ContainerSources.SampleCount[i] <= (MixBuffer->SampleCount) * 2)
        {
            ContainerSources.PlayState[i] = Stopping;
        }

        switch(ContainerSources.PlayState[i])
        {
            case Playing:
            {
                polar_render_Source(ContainerSources.SampleRate[i], ContainerSources.SampleCount[i], MixBuffer->SampleCount, ContainerSources.Type[i], ContainerSources.FX[i], ContainerSources.Buffer[i]);
                polar_render_MixSources(ContainerSources.States[i].Amplitude.Current, ContainerSources.States[i].Amplitude.Previous, ContainerSources.Buffer[i], MixBuffer);
                break;
            }

            case Stopped:
            {
                break;
            }

            case Stopping:
            {
                ContainerSources.States[i].Amplitude.Current = 0;
                ContainerSources.States[i].Amplitude.IsFading = true;
                
                polar_render_Source(ContainerSources.SampleRate[i], ContainerSources.SampleCount[i], MixBuffer->SampleCount, ContainerSources.Type[i], ContainerSources.FX[i], ContainerSources.Buffer[i]);
                polar_render_MixSources(ContainerSources.States[i].Amplitude.Current, ContainerSources.States[i].Amplitude.Previous, ContainerSources.Buffer[i], MixBuffer);
                
                break;
            }

            default:
            {
                break;
            }
        }
    }

    for(u32 FrameIndex = 0; FrameIndex < MixBuffer->SampleCount; ++FrameIndex)
    {
        MixBuffer->Data[FrameIndex] *= ContainerAmplitude;	
    }
}

void polar_render_Submix(POLAR_SUBMIX *Submix, POLAR_BUFFER *MixBuffer)
{
    for(u8 i = 0; i < Submix->Containers.CurrentContainers; ++i)
    {
        polar_render_Container(Submix->Containers.Sources[i], Submix->Containers.Amplitude[i], MixBuffer);
    }

    for(u32 FrameIndex = 0; FrameIndex < MixBuffer->SampleCount; ++FrameIndex)
    {
        MixBuffer->Data[FrameIndex] *= Submix->Amplitude;	
    }
}


void polar_render_ConvertToInt16(POLAR_ENGINE *Engine, POLAR_BUFFER *MixBuffer, i16 *OutputBuffer)
{
    i16 *ConvertedSamples = OutputBuffer;
    
    for(u32 SampleIndex = 0; SampleIndex < MixBuffer->SampleCount; ++SampleIndex)
    {
        f32 FloatSample = MixBuffer->Data[SampleIndex];
        i16 IntSample = FloatToInt16(FloatSample);

        for(u8 ChannelIndex = 0; ChannelIndex < Engine->Channels; ++ChannelIndex)
        {
            *ConvertedSamples++ = IntSample;
        }
    }
}

void polar_render_Callback(POLAR_ENGINE *Engine, POLAR_MIXER *Mixer, POLAR_BUFFER *MixBuffer, i16 *OutputBuffer)
{
    //Clear mixer
    for(u32 FrameIndex = 0; FrameIndex < (MixBuffer->SampleCount * Engine->Channels); ++FrameIndex)
	{
        MixBuffer->Data[FrameIndex] = 0.0f;
    }

    //Render submixes
    for(POLAR_SUBMIX *SubmixIndex = Mixer->FirstInList; SubmixIndex; SubmixIndex = SubmixIndex->NextSubmix)
    {
        for(POLAR_SUBMIX *ChildSubmixIndex = SubmixIndex->ChildSubmix; ChildSubmixIndex; ChildSubmixIndex = ChildSubmixIndex->ChildSubmix)
        {
            polar_render_Submix(ChildSubmixIndex, MixBuffer);
        }

        polar_render_Submix(SubmixIndex, MixBuffer);
    }

    //Mix to master amplitude
    for(u32 FrameIndex = 0; FrameIndex < MixBuffer->SampleCount; ++FrameIndex)
	{
        MixBuffer->Data[FrameIndex] *= Mixer->Amplitude;
    }

    //Convert to int16 samples
    polar_render_ConvertToInt16(Engine, MixBuffer, OutputBuffer);;
}

#endif