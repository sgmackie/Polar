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
	//Cast from float to BYTE
	SampleBuffer = reinterpret_cast<f32 *>(DeviceBuffer);
	
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


// internal void polar_render_Update(POLAR_DATA &Engine, POLAR_WAV *File, OSCILLATOR *Osc, POLAR_MEMORY *Memory, POLAR_INPUT *Input)
extern "C" POLAR_RENDER_CALLBACK(RenderUpdate)
{
	//Memory checking
	Assert(sizeof(POLAR_OBJECT_STATE) <= Memory->PermanentDataSize);
	POLAR_OBJECT_STATE *ObjectState = (POLAR_OBJECT_STATE *)Memory->PermanentData;

	//Initial object initialisation
	if(!Memory->IsInitialized)
    {       
        ObjectState->Frequency = 440;
        ObjectState->Amplitude = 0.35f;
        ObjectState->Pan = 0;

        Memory->IsInitialized = true;
    }

	//Use input to change states
    for(u32 ControllerIndex = 0; ControllerIndex < ArrayCount(Input->Controllers); ++ControllerIndex)
    {
		POLAR_INPUT_CONTROLLER *Controller = ControllerGet(Input, ControllerIndex);

		if(Controller->State.ButtonPress.MoveUp.EndedDown)
        {
            ObjectState->Amplitude += 0.1f;
        }

        if(Controller->State.ButtonPress.MoveDown.EndedDown)
        {
            ObjectState->Amplitude -= 0.1f;
        }

		if(Controller->State.ButtonPress.MoveRight.EndedDown)
        {
            ObjectState->Frequency += 10.0f;
        }
            
        if(Controller->State.ButtonPress.MoveLeft.EndedDown)
        {
            ObjectState->Frequency -= 10.0f;
        }

		if(Controller->State.ButtonPress.ActionRight.EndedDown)
        {
            ObjectState->Pan += 0.1f;
        }
            
        if(Controller->State.ButtonPress.ActionLeft.EndedDown)
        {
            ObjectState->Pan -= 0.1f;
        }
	}

	//Write sound buffer
	if(File != nullptr)
	{
        polar_render_BufferFill(Engine.Channels, (Engine.Buffer.FramesAvailable * Engine.Channels), Engine.Buffer.SampleBuffer, Engine.Buffer.DeviceBuffer, File->Data, Osc, ObjectState);
	}

	else
	{
		polar_render_BufferFill(Engine.Channels, (Engine.Buffer.FramesAvailable * Engine.Channels), Engine.Buffer.SampleBuffer, Engine.Buffer.DeviceBuffer, nullptr, Osc, ObjectState);
	}
}

#endif