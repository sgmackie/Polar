#ifndef polar_render_cpp
#define polar_render_cpp

void polar_render_Source(u32 &SampleRate, u64 &SampleCount, f64 &Amplitude, f64 &AmplitudeTarget, f64 &AmplitudeDelta, u32 Samples, POLAR_SOURCE_TYPE &Type, u32 &FX, f32 *Buffer)
{
    while(SampleCount != 0 && Type.Flag != SO_NONE)
    {
        f64 SecondsPerSample = (1.0f / (f64) SampleRate);

        f64 CurrentAmp = (Amplitude);
        f64 AmpDelta = (SecondsPerSample * AmplitudeDelta);

        bool AmpEnded;

        if(AmpDelta != 0.0f)
        {
            f64 NewDeltaVolume = (AmplitudeTarget - CurrentAmp);
    
            u32 VolumeSampleCount = (u32)((NewDeltaVolume / AmpDelta + 0.5f));
            if(Samples > VolumeSampleCount)
            {
                Samples = VolumeSampleCount;
                AmpEnded = true;
            }

            if(AmpDelta == 0)
            {
                AmpEnded = true;
            }
        }

        switch(Type.Flag)
        {
            case SO_FILE:
            {
                u64 Position = 0;
                u64 SamplesRemaining = (Type.File->FrameCount - Type.File->ReadIndex);

                if(SamplesRemaining < Samples)
                {
                    Samples = SamplesRemaining;
                }

                for(u32 FrameIndex = 0; FrameIndex < Samples; ++FrameIndex)
	            {
                    Position = (Type.File->ReadIndex += Type.File->Channels);
	            	Buffer[FrameIndex] = Type.File->Samples[Position];

                    CurrentAmp += AmpDelta;
                }

                break;
            }

            case SO_OSCILLATOR:
            {
                for(u32 FrameIndex = 0; FrameIndex < Samples; ++FrameIndex)
	            {
	            	Buffer[FrameIndex] = (f32) Type.Oscillator->Tick(Type.Oscillator);

                    CurrentAmp += AmpDelta;
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
                EffectAM(Samples, SampleRate, Buffer, 100);
            }

            if(FX & FX_ECHO)
            {
                EffectEcho(Samples, SampleRate, Buffer, 3);
            }
        }

        Amplitude = CurrentAmp;
        if(AmpEnded)
        {
            Amplitude = AmplitudeTarget;
            AmplitudeDelta = 0.0f;
        }

        SampleCount -= Samples;
        return;  
    }
}

void polar_render_SumStereo(POLAR_ENGINE PolarEngine, u8 &Channels, f32 *PanPositions, f64 &Amplitude, f32 *Buffer, f32 *SourceOutput)
{
    for(u32 FrameIndex = 0; FrameIndex < PolarEngine.BufferFrames; ++FrameIndex)
    {
        for(u32 ChannelIndex = 0; ChannelIndex < Channels; ++ChannelIndex)
        {
            f32 LeftPhase = 0.25f * PI32 * (PanPositions[ChannelIndex] + 1.0f);
            f32 RightPhase = 0.5f * PI32 * (0.5f * (PanPositions[ChannelIndex] + 1.0f) + 1.0);

            SourceOutput[FrameIndex * PolarEngine.Channels] += ((sinf(LeftPhase) * Buffer[FrameIndex]) * Amplitude);
            SourceOutput[FrameIndex * PolarEngine.Channels + 1] += ((sinf(RightPhase) * Buffer[FrameIndex]) * Amplitude);
        }
    }
}

void polar_render_Container(POLAR_ENGINE PolarEngine, POLAR_SOURCE &ContainerSources, f64 ContainerAmplitude, f32 *ContainerOutput)
{
    for(u8 i = 0; i <= ContainerSources.CurrentSources; ++i)
    {
        if(ContainerSources.SampleCount[i] <= 0)
        {
            ContainerSources.PlayState[i] = Stopped;
        }

        if(ContainerSources.SampleCount[i] == ContainerSources.BufferSize[i])
        {
            ContainerSources.PlayState[i] = Stopping;
        }

        switch(ContainerSources.PlayState[i])
        {
            case Playing:
            {
                ContainerSources.BufferSize[i] = (PolarEngine.BufferFrames / PolarEngine.Channels);
                polar_render_Source(ContainerSources.SampleRate[i], ContainerSources.SampleCount[i], ContainerSources.States[i].AmplitudeCurrent, ContainerSources.States[i].AmplitudeTarget, ContainerSources.States[i].AmplitudeDelta, ContainerSources.BufferSize[i], ContainerSources.Type[i], ContainerSources.FX[i], ContainerSources.Buffer[i]);
                Summing(PolarEngine, ContainerSources.Channels[i], ContainerSources.States[i].PanPositions, ContainerSources.States[i].AmplitudeCurrent, ContainerSources.Buffer[i], ContainerOutput);

                break;
            }

            case Stopped:
            {
                break;
            }

            case Stopping:
            {
                ContainerSources.BufferSize[i] = (PolarEngine.BufferFrames / PolarEngine.Channels);
                ContainerSources.States[i].AmplitudeTarget = 0;
                polar_render_Source(ContainerSources.SampleRate[i], ContainerSources.SampleCount[i], ContainerSources.States[i].AmplitudeCurrent, ContainerSources.States[i].AmplitudeTarget, ContainerSources.States[i].AmplitudeDelta, ContainerSources.BufferSize[i], ContainerSources.Type[i], ContainerSources.FX[i], ContainerSources.Buffer[i]);
                Summing(PolarEngine, ContainerSources.Channels[i], ContainerSources.States[i].PanPositions, ContainerSources.States[i].AmplitudeCurrent, ContainerSources.Buffer[i], ContainerOutput);

                break;
            }

            default:
            {
                break;
            }
        }
    }

    for(u32 FrameIndex = 0; FrameIndex < PolarEngine.BufferFrames; ++FrameIndex)
    {
        ContainerOutput[FrameIndex] *= ContainerAmplitude;	
    }
}



void polar_render_Submix(POLAR_ENGINE PolarEngine, POLAR_SUBMIX *Submix, f32 *SubmixOutput)
{
    for(u8 i = 0; i < Submix->Containers.CurrentContainers; ++i)
    {
        polar_render_Container(PolarEngine, Submix->Containers.Sources[i], Submix->Containers.Amplitude[i], SubmixOutput);
    }

    for(u32 FrameIndex = 0; FrameIndex < PolarEngine.BufferFrames; ++FrameIndex)
    {
        SubmixOutput[FrameIndex] *= Submix->Amplitude;	
    }
}

void polar_render_Callback(POLAR_ENGINE PolarEngine, POLAR_MIXER *Mixer, f32 *MasterOutput)
{
    for(POLAR_SUBMIX *SubmixIndex = Mixer->FirstInList; SubmixIndex; SubmixIndex = SubmixIndex->NextSubmix)
    {
        for(POLAR_SUBMIX *ChildSubmixIndex = SubmixIndex->ChildSubmix; ChildSubmixIndex; ChildSubmixIndex = ChildSubmixIndex->ChildSubmix)
        {
            polar_render_Submix(PolarEngine, ChildSubmixIndex, MasterOutput);
        }

        polar_render_Submix(PolarEngine, SubmixIndex, MasterOutput);
    }

    for(u32 FrameIndex = 0; FrameIndex < PolarEngine.BufferFrames; ++FrameIndex)
    {
        MasterOutput[FrameIndex] *= Mixer->Amplitude;	
    }
}

#endif