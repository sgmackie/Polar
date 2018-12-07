#ifndef polar_platform_cpp
#define polar_platform_cpp

#include "polar_WASAPI.cpp"

//Create and initialise WASAPI struct
WASAPI_DATA *polar_WASAPI_Create(WASAPI_BUFFER &Buffer)
{	
	WASAPI_DATA *WASAPI = wasapi_CreateDataInterface();
	wasapi_InitDataInterface(*WASAPI);

	if((WASAPI->DeviceReady = wasapi_InitDevice(WASAPI->HR, *WASAPI)) == false)
	{
		debug_PrintLine("WASAPI: Failed to configure output device! Exiting...");
		wasapi_DeinitDevice(*WASAPI);
		wasapi_DestroyDataInterface(WASAPI);
		return nullptr;
	}

	debug_PrintLine("\t\t\t\t\tWASAPI: Configured output device! Playing...");

	//TODO: Create function for this
	Buffer.FramePadding = 0;
	Buffer.FramesAvailable = 0;
	Buffer.SampleBuffer = (f32 *) VirtualAlloc(0, ((sizeof *Buffer.SampleBuffer) * ((WASAPI->OutputBufferFrames * WASAPI->OutputWaveFormat->Format.nChannels))), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
	Buffer.ByteBuffer = nullptr;

	// Initial zero fill	
	WASAPI->HR = WASAPI->AudioRenderClient->GetBuffer(WASAPI->OutputBufferFrames, &Buffer.ByteBuffer);
	HR_TO_RETURN(WASAPI->HR, "Couldn't get WASAPI buffer for zero fill", nullptr);

	WASAPI->HR = WASAPI->AudioRenderClient->ReleaseBuffer(WASAPI->OutputBufferFrames, AUDCLNT_BUFFERFLAGS_SILENT);
	HR_TO_RETURN(WASAPI->HR, "Couldn't release WASAPI buffer for zero fill", nullptr);

	return WASAPI;
}

//Remove WASAPI struct
void polar_WASAPI_Destroy(WASAPI_DATA *WASAPI)
{
	wasapi_DeinitDevice(*WASAPI);
	wasapi_DestroyDataInterface(WASAPI);
}

//Get WASAPI buffer and release after filling with specified amount of samples
void polar_WASAPI_PrepareBuffer(WASAPI_DATA *WASAPI, WASAPI_BUFFER &Buffer)
{
	if(WASAPI->DeviceState == Playing)
	{
		WaitForSingleObject(WASAPI->RenderEvent, INFINITE);

		WASAPI->HR = WASAPI->AudioClient->GetCurrentPadding(&Buffer.FramePadding);
		HR_TO_RETURN(WASAPI->HR, "Couldn't get current padding", NONE);

		Buffer.FramesAvailable = WASAPI->OutputBufferFrames - Buffer.FramePadding;

		if(Buffer.FramesAvailable != 0)
		{
			WASAPI->HR = WASAPI->AudioRenderClient->GetBuffer(Buffer.FramesAvailable, &Buffer.ByteBuffer);
			HR_TO_RETURN(WASAPI->HR, "Couldn't get WASAPI buffer", NONE);
		}
	}
}

void polar_WASAPI_ReleaseBuffer(WASAPI_DATA *WASAPI, WASAPI_BUFFER &Buffer)
{
	if(WASAPI->DeviceState == Playing)
	{
		//?Because the BYTE buffer is filled outside this function in the Polar render (and stored as a struct), it doesn't need to be called here and the buffer can still be released as expected

		WASAPI->HR = WASAPI->AudioRenderClient->ReleaseBuffer(Buffer.FramesAvailable, 0);
		HR_TO_RETURN(WASAPI->HR, "Couldn't release WASAPI buffer", NONE);	
	}
}


//Update the audio clock's position in the current stream
void polar_WASAPI_UpdateClock(WASAPI_DATA &Interface, WASAPI_CLOCK Clock)
{
    Interface.AudioClock->GetFrequency(&Clock.PositionFrequency);
    Interface.AudioClock->GetPosition(&Clock.PositionUnits, 0);
}


#endif