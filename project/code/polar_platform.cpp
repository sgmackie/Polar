#ifndef polar_platform_cpp
#define polar_platform_cpp

#include "polar_WASAPI.cpp"

//Create and initialise WASAPI struct
WASAPI_DATA *polar_WASAPI_Create(WASAPI_BUFFER &Buffer)
{
	//Windows error result
	HRESULT HR = S_FALSE;
	
	WASAPI_DATA *WASAPI = wasapi_CreateDataInterface();
	wasapi_InitDataInterface(*WASAPI);

	if((WASAPI->DeviceReady = wasapi_InitDevice(HR, *WASAPI)) == false)
	{
		debug_PrintLine(Console, "WASAPI: Failed to configure output device! Exiting...");
		wasapi_DeinitDevice(*WASAPI);
		wasapi_DestroyDataInterface(WASAPI);
		return nullptr;
	}

	debug_PrintLine(Console, "\t\t\t\t\tWASAPI: Configured output device! Playing...");

	Buffer.FramePadding = 0;
	Buffer.FramesAvailable = 0;
	Buffer.Data = nullptr;

	// Initial zero fill	
	HR = WASAPI->AudioRenderClient->GetBuffer(WASAPI->OutputBufferFrames, &Buffer.Data);
	HR_TO_RETURN(HR, "Couldn't get WASAPI buffer for zero fill", nullptr);

	HR = WASAPI->AudioRenderClient->ReleaseBuffer(WASAPI->OutputBufferFrames, AUDCLNT_BUFFERFLAGS_SILENT);
	HR_TO_RETURN(HR, "Couldn't release WASAPI buffer for zero fill", nullptr);
	
	return WASAPI;
}

//Remove WASAPI struct
void polar_WASAPI_Destroy(WASAPI_DATA *WASAPI)
{
	wasapi_DeinitDevice(*WASAPI);
	wasapi_DestroyDataInterface(WASAPI);
}

//Get WASAPI buffer and release after filling with specified amount of samples
void polar_WASAPI_Render(WASAPI_DATA *WASAPI, WASAPI_BUFFER &Buffer, OSCILLATOR *Osc)
{
	HRESULT HR = S_FALSE;

	WaitForSingleObject(WASAPI->RenderEvent, INFINITE);

	HR = WASAPI->AudioClient->GetCurrentPadding(&Buffer.FramePadding);
	HR_TO_RETURN(HR, "Couldn't get current padding", NONE);

	Buffer.FramesAvailable = WASAPI->OutputBufferFrames - Buffer.FramePadding;

	HR = WASAPI->AudioRenderClient->GetBuffer(Buffer.FramesAvailable, &Buffer.Data);
	HR_TO_RETURN(HR, "Couldn't get WASAPI buffer", NONE);

	wasapi_FillBuffer(*WASAPI, Buffer.FramesAvailable, Buffer.Data, Osc, 0.25);

	HR = WASAPI->AudioRenderClient->ReleaseBuffer(Buffer.FramesAvailable, 0);
	HR_TO_RETURN(HR, "Couldn't release WASAPI buffer", NONE);	
}


#endif