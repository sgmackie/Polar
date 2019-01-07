#ifndef polar_render_cpp
#define polar_render_cpp

//Bit conversion functions
//TODO: Create 24 bit function (shift 32bit result << 8 bits?)
//TODO: Create dithering function!
internal f32 polar_render_Int32ToFloat(i32 Input)
{
	f32 Result = 0;

	//Convert
	Result = ((f32) Input) / (f32) 2147483648;

	//Rounding
	if(Result > 1)
	{
		Result = 1;
	}
	if(Result < -1)
	{
		Result = -1;
	}

	return Result;
}

internal i32 polar_render_FloatToInt32(f32 Input)
{
	i32 Result = 0;

	//Multiply by max integer value
	Input = Input * 2147483648;
	
	if(Input > 2147483647)
	{
		Input = 2147483647;
	}
	if(Input < -2147483648) 
	{
		Input = -2147483648;
	}

	//Convert
	Result = (i32) Input;

	return Result;
}

internal f32 polar_render_Int16ToFloat(i16 Input)
{
	f32 Result = 0;

	//Convert
	Result = ((f32) Input) / (f32) 32768;

	//Rounding
	if(Result > 1)
	{
		Result = 1;
	}
	if(Result < -1)
	{
		Result = -1;
	}

	return Result;
}

internal i16 polar_render_FloatToInt16(f32 Input)
{
	i16 Result = 0;

	//Multiply by max integer value
	Input = Input * 32768;
	
	if(Input > 32767)
	{
		Input = 32767;
	}
	if(Input < -32768) 
	{
		Input = -32768;
	}

	//Convert
	Result = (i16) Input;

	return Result;
}

internal f32 polar_render_Int8ToFloat(i8 Input)
{
	f32 Result = 0;

	//Convert
	Result = ((f32) Input) / (f32) 128;

	//Rounding
	if(Result > 1)
	{
		Result = 1;
	}
	if(Result < -1)
	{
		Result = -1;
	}

	return Result;
}

internal i8 polar_render_FloatToInt8(f32 Input)
{
	i8 Result = 0;

	//Multiply by max integer value
	Input = Input * 128;
	
	if(Input > 127)
	{
		Input = 127;
	}
	if(Input < -128) 
	{
		Input = -128;
	}

	//Convert
	Result = (i8) Input;

	return Result;
}

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


void polar_render_ObjectRender(u16 ChannelCount, u32 FramesToWrite, f32 *SampleBuffer, POLAR_OBJECT *Object)
{
    f32 CurrentSample = 0;
    f32 PanAmp = 0;

  	for(u32 FrameIndex = 0; FrameIndex < FramesToWrite; FrameIndex += ChannelCount)
	{
		//TODO: Taylor series GPU for sine call, matrix additive function
		Object->Oscillator->FrequencyCurrent = Object->State->Frequency;
		CurrentSample = (f32) Object->Oscillator->Tick(Object->Oscillator);

		for(u16 ChannelIndex = 0; ChannelIndex < ChannelCount; ++ChannelIndex)
		{
			PanAmp = polar_render_PanPositionGet(ChannelIndex, Object->State->Amplitude, Object->State->Pan);

			SampleBuffer[FrameIndex + ChannelIndex] = CurrentSample * PanAmp;
		}		
	}
}

internal POLAR_OBJECT *polar_update_ObjectPlay(POLAR_ENGINE *Engine, POLAR_OBJECT *InputSound, u32 Duration, u16 Channels)
{
	//If the linked list is null, allocate space from the arena
    if(!Engine->State->FirstInFreeList)
    {
        Engine->State->FirstInFreeList = polar_PushStruct(&Engine->State->Arena, POLAR_OBJECT);
        Engine->State->FirstInFreeList->Next = 0;
    }

    POLAR_OBJECT *CurrentObject = Engine->State->FirstInFreeList;
    Engine->State->FirstInFreeList = CurrentObject->Next;

    CurrentObject->SamplesPlayed = 0;
    CurrentObject->SampleCount = ((Engine->SampleRate * Duration) * Channels);

    CurrentObject->UID = InputSound->UID;

    CurrentObject->Oscillator = InputSound->Oscillator;
    CurrentObject->State = InputSound->State;

    CurrentObject->Next = Engine->State->FirstInList;
    Engine->State->FirstInList = CurrentObject;

    return CurrentObject;
}


internal void polar_render_BufferFill(u16 ChannelCount, u32 FramesToWrite, f32 *SampleBuffer, void *DeviceBuffer, f32 *MixChannel01, f32 *FileSamples)
{	
	//Increase frame counter by the number of channels
	for(u32 FrameIndex = 0; FrameIndex < FramesToWrite; FrameIndex += ChannelCount)
	{
		for(u16 ChannelIndex = 0; ChannelIndex < ChannelCount; ++ChannelIndex)
		{
			SampleBuffer[FrameIndex + ChannelIndex] = MixChannel01[FrameIndex + ChannelIndex];

			if(FileSamples != nullptr)
			{
				FileSamples[FrameIndex + ChannelIndex] = MixChannel01[FrameIndex + ChannelIndex];
			}
		}		
	}

	memcpy(DeviceBuffer, SampleBuffer, (sizeof(* SampleBuffer) * FramesToWrite));
}


extern "C" POLAR_UPDATE_CALLBACK(Update)
{
    Assert((sizeof (POLAR_ENGINE_STATE)) <= Memory->PermanentDataSize);
   	Engine->State = (POLAR_ENGINE_STATE *) Memory->PermanentData;

    if(!Engine->State->IsInitialised)
    {   
        polar_memory_ArenaInitialise(&Engine->State->Arena, (Memory->PermanentDataSize - sizeof(POLAR_ENGINE_STATE)), ((uint8 *) Memory->PermanentData + sizeof(POLAR_ENGINE_STATE)));
        Engine->State->FirstInList = 0;
        Engine->State->FirstInFreeList = 0;
        Engine->State->MasterAmplitude = 0.5;
        Engine->State->IsInitialised = true;
    }

    //Create temporary arena for mixing memory
	Assert(sizeof(POLAR_MEMORY_TRANSIENT) <= Memory->TemporaryDataSize);   
	POLAR_MEMORY_TRANSIENT *TransientState = (POLAR_MEMORY_TRANSIENT *) Memory->TemporaryData;
    
	if(!TransientState->IsInitialized)
    {
        polar_memory_ArenaInitialise(&TransientState->TransientArena, (Memory->TemporaryDataSize - sizeof(POLAR_MEMORY_TRANSIENT)), ((uint8 *) Memory->TemporaryData + sizeof(POLAR_MEMORY_TRANSIENT)));
        TransientState->IsInitialized = true;
    }

    //States
	//!Pass the ID or name of an object, not the array index!
    polar_update_ObjectPlay(Engine, &Objects[0], 1, STEREO);
    polar_update_ObjectPlay(Engine, &Objects[1], 3, STEREO);
    polar_update_ObjectPlay(Engine, &Objects[2], 5, STEREO);
    polar_update_ObjectPlay(Engine, &Objects[3], 2, STEREO);

    for(u32 ControllerIndex = 0; ControllerIndex < ArrayCount(Input->Controllers); ++ControllerIndex)
    {
		POLAR_INPUT_CONTROLLER *Controller = ControllerGet(Input, ControllerIndex);

		for(POLAR_OBJECT **ObjectIndex = &Engine->State->FirstInList; *ObjectIndex;)
    	{
			POLAR_OBJECT *CurrentObject = *ObjectIndex;

			if(Controller->State.Press.MoveUp.EndedDown && Controller->State.Press.LeftTrigger.EndedDown)
        	{
        	    Engine->State->MasterAmplitude += 0.010;
				if(Engine->State->MasterAmplitude > 1.0)
				{
					Engine->State->MasterAmplitude = 1.0;
				}

				if(Engine->State->MasterAmplitude < 0.0)
				{
					Engine->State->MasterAmplitude = 0.0;
				}
        	}

			if(Controller->State.Press.MoveDown.EndedDown && Controller->State.Press.LeftTrigger.EndedDown)
        	{
        	    Engine->State->MasterAmplitude -= 0.010;
				if(Engine->State->MasterAmplitude > 1.0)
				{
					Engine->State->MasterAmplitude = 1.0;
				}

				if(Engine->State->MasterAmplitude < 0.0)
				{
					Engine->State->MasterAmplitude = 0.0;
				}
        	}

			if(Controller->State.Press.MoveUp.EndedDown)
        	{
        	    CurrentObject->State->Amplitude += 0.010;
				if(CurrentObject->State->Amplitude > 1.0)
				{
					CurrentObject->State->Amplitude = 1.0;
				}

				if(CurrentObject->State->Amplitude < 0.0)
				{
					CurrentObject->State->Amplitude = 0.0;
				}
        	}

        	if(Controller->State.Press.MoveDown.EndedDown)
        	{
        	    CurrentObject->State->Amplitude -= 0.010;
				if(CurrentObject->State->Amplitude > 1.0)
				{
					CurrentObject->State->Amplitude = 1.0;
				}

				if(CurrentObject->State->Amplitude < 0.0)
				{
					CurrentObject->State->Amplitude = 0.0;
				}
        	}

			if(Controller->State.Press.MoveRight.EndedDown)
        	{
        	    CurrentObject->State->Frequency += 10.0f;
        	}
            
        	if(Controller->State.Press.MoveLeft.EndedDown)
        	{
        	    CurrentObject->State->Frequency -= 10.0f;
        	}

			if(Controller->State.Press.ActionRight.EndedDown)
        	{
        	    CurrentObject->State->Pan += 0.010;
				if(CurrentObject->State->Pan > 1.0)
				{
					CurrentObject->State->Pan = 1.0;
				}
        	}
            
        	if(Controller->State.Press.ActionLeft.EndedDown)
        	{
        	    CurrentObject->State->Pan -= 0.010;
				if(CurrentObject->State->Pan < -1.0)
				{
					CurrentObject->State->Pan = -1.0;
				}
        	}

        	if(Controller->State.Press.LeftShoulder.EndedDown)
			{
				CurrentObject->State->Waveform = SINE;
			}

			if(Controller->State.Press.RightShoulder.EndedDown)
			{
				CurrentObject->State->Waveform = TRIANGLE;
			}
				
			ObjectIndex = &CurrentObject->Next;
		}
	}
}


extern "C" POLAR_RENDER_CALLBACK(Render)
{
	//Get current states allocated in the update callback
	Engine->State = (POLAR_ENGINE_STATE *) Memory->PermanentData;
	POLAR_MEMORY_TRANSIENT *TransientState = (POLAR_MEMORY_TRANSIENT *) Memory->TemporaryData;
    
	//Allocate temporary mixing memory
    POLAR_MEMORY_TEMPORARY MixerMemory = polar_memory_TemporaryArenaCreate(&TransientState->TransientArena);
	f32 *MixChannel01 = polar_PushArray(&TransientState->TransientArena, (Engine->Buffer.FramesAvailable * Engine->Channels), f32);
        
    //Create temporary render buffer and clear to 0
    f32 *RenderChannel01 = MixChannel01;
    for(u32 SampleIndex = 0; SampleIndex < (Engine->Buffer.FramesAvailable * Engine->Channels); ++SampleIndex)
    {
        *RenderChannel01++ = 0.0f;
    }

    //Iterate through the list of current objects
    for(POLAR_OBJECT **ObjectIndex = &Engine->State->FirstInList; *ObjectIndex;)
    {
        POLAR_OBJECT *CurrentObject = *ObjectIndex;
        bool SoundFinished = false;
            
        u32 TotalSamplesToMix = (Engine->Buffer.FramesAvailable * Engine->Channels);

        while(TotalSamplesToMix && !SoundFinished)
        {
            Assert(CurrentObject->SamplesPlayed >= 0);
            u32 SamplesRemainingInSound = (CurrentObject->SampleCount - CurrentObject->SamplesPlayed);
            u32 SamplesToMix = TotalSamplesToMix;

            if(SamplesToMix > SamplesRemainingInSound)
            {
                SamplesToMix = SamplesRemainingInSound;
            }

            polar_render_ObjectRender(Engine->Channels, SamplesToMix, RenderChannel01, CurrentObject);

            for(u32 i = 0; i < SamplesToMix; ++i)
            {
                MixChannel01[i] += RenderChannel01[i] * Engine->State->MasterAmplitude;
            }

            Assert(TotalSamplesToMix >= SamplesToMix);
            CurrentObject->SamplesPlayed += SamplesToMix;
            TotalSamplesToMix -= SamplesToMix;

            if((u32) CurrentObject->SamplesPlayed == CurrentObject->SampleCount)
            {
                // SoundFinished = true;
                CurrentObject->SamplesPlayed = 0;
            }

            else
            {
                Assert(TotalSamplesToMix == 0);
            }

            SoundFinished = true;
        }

        if(SoundFinished)
        {
            *ObjectIndex = CurrentObject->Next;
            CurrentObject->Next = Engine->State->FirstInFreeList;
            Engine->State->FirstInFreeList = CurrentObject;
        }
        else
        {
            ObjectIndex = &CurrentObject->Next;
        }
    }

	//Write sound buffer
	if(File != nullptr)
	{
		polar_render_BufferFill(Engine->Channels, (Engine->Buffer.FramesAvailable * Engine->Channels), Engine->Buffer.SampleBuffer, Engine->Buffer.DeviceBuffer, MixChannel01, File->Data);
	}

	else
	{
		polar_render_BufferFill(Engine->Channels, (Engine->Buffer.FramesAvailable * Engine->Channels), Engine->Buffer.SampleBuffer, Engine->Buffer.DeviceBuffer, MixChannel01, nullptr);
	}

	//Free mixer memory
	polar_memory_TemporaryArenaDestroy(MixerMemory);

}

#endif
