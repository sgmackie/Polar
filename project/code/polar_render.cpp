#ifndef polar_render_cpp
#define polar_render_cpp

internal f32 polar_render_PanPositionGet(u16 Position, f32 Amplitude, f32 PanFactor)
{
	f32 PanPosition = Amplitude; 

	//Left panning
	if(Position % 2 == 0)
	{
		PanPosition = Amplitude * (f32) sqrt(2.0) * (1 - PanFactor) / (2* (f32) sqrt(1 + PanFactor * PanFactor));
	}

	//Right panning
	if(Position % 2 != 0)
	{
		PanPosition = Amplitude * (f32) sqrt(2.0) * (1 + PanFactor) / (2* (f32) sqrt(1 + PanFactor * PanFactor));
	}

	return PanPosition;
}

internal void polar_render_BufferFill(u16 ChannelCount, u32 FramesToWrite, f32 *SampleBuffer, void *DeviceBuffer, f32 *FileSamples, OSCILLATOR *Osc, POLAR_OBJECT_STATE *State)
{	
	f32 CurrentSample = 0;
	f32 PanAmp = 0;

	//Increase frame counter by the number of channels
	for(u32 FrameIndex = 0; FrameIndex < FramesToWrite; FrameIndex += ChannelCount)
	{
		//TODO: Taylor series GPU for sine call, matrix additive function
		Osc->FrequencyCurrent = State->Frequency;
		CurrentSample = (f32) Osc->Tick(Osc);

		for(u16 ChannelIndex = 0; ChannelIndex < ChannelCount; ++ChannelIndex)
		{
			PanAmp = polar_render_PanPositionGet(ChannelIndex, State->Amplitude, State->Pan);

			SampleBuffer[FrameIndex + ChannelIndex] = CurrentSample * PanAmp;

			if(FileSamples != nullptr)
			{
				FileSamples[FrameIndex + ChannelIndex] = CurrentSample * PanAmp;
			}
		}		
	}

	memcpy(DeviceBuffer, SampleBuffer, (sizeof(* SampleBuffer) * FramesToWrite));
}

extern "C" POLAR_RENDER_CALLBACK(RenderUpdate)
{
	//Memory checking	
	Assert(sizeof(POLAR_OBJECT_STATE) <= Memory->PermanentDataSize);
	Array->Objects[0]->State = (POLAR_OBJECT_STATE *) Memory->PermanentData;

	//Initial object initialisation
	if(!Memory->IsInitialized)
    {
		//!Can't loop through array because each state needs to be allocated like above; divide PermanentData into chunks to hand out in a loop
		// for(u32 i = 0; i < Array->Count; ++i)
        // {
        //     Array->Objects[i]->State->Frequency = 440;
        //     Array->Objects[i]->State->Amplitude = 0.35f;
        //     Array->Objects[i]->State->Pan = 0;
		// }

        Array->Objects[0]->State->Frequency = 440;
        Array->Objects[0]->State->Amplitude = 0.2f;
        Array->Objects[0]->State->Pan = 0;
        Array->Objects[0]->State->Waveform = SINE;

        Memory->IsInitialized = true;
    }

	//Use input to change states
    for(u32 ControllerIndex = 0; ControllerIndex < ArrayCount(Input->Controllers); ++ControllerIndex)
    {
		POLAR_INPUT_CONTROLLER *Controller = ControllerGet(Input, ControllerIndex);

		if(Controller->State.Press.MoveUp.EndedDown)
        {
            Array->Objects[0]->State->Amplitude += 0.1f;
        }

        if(Controller->State.Press.MoveDown.EndedDown)
        {
            Array->Objects[0]->State->Amplitude -= 0.1f;
        }

		if(Controller->State.Press.MoveRight.EndedDown)
        {
            Array->Objects[0]->State->Frequency += 10.0f;
        }
            
        if(Controller->State.Press.MoveLeft.EndedDown)
        {
            Array->Objects[0]->State->Frequency -= 10.0f;
        }

		if(Controller->State.Press.ActionRight.EndedDown)
        {
            Array->Objects[0]->State->Pan += 0.1f;
        }
            
        if(Controller->State.Press.ActionLeft.EndedDown)
        {
            Array->Objects[0]->State->Pan -= 0.1f;
        }

        if(Controller->State.Press.LeftShoulder.EndedDown)
		{
			Array->Objects[0]->State->Waveform = SINE;
		}

		if(Controller->State.Press.RightShoulder.EndedDown)
		{
			Array->Objects[0]->State->Waveform = TRIANGLE;
		}
	}

	//Write sound buffer
	if(File != nullptr)
	{
        polar_render_BufferFill(Engine.Channels, (Engine.Buffer.FramesAvailable * Engine.Channels), Engine.Buffer.SampleBuffer, Engine.Buffer.DeviceBuffer, File->Data, Array->Objects[0]->Oscillator,  Array->Objects[0]->State);
	}

	else
	{
		polar_render_BufferFill(Engine.Channels, (Engine.Buffer.FramesAvailable * Engine.Channels), Engine.Buffer.SampleBuffer, Engine.Buffer.DeviceBuffer, nullptr, Array->Objects[0]->Oscillator,  Array->Objects[0]->State);
	}
}

#endif