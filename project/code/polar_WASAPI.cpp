#ifndef polar_WASAPI_cpp
#define polar_WASAPI_cpp

#include "polar_WASAPI.h"

//Create struct for WASAPI properties including output/rendering device info
WASAPI_DATA *wasapi_CreateDataInterface()
{
	WASAPI_DATA *Result = (WASAPI_DATA *) VirtualAlloc(0, (sizeof *Result), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
	return Result;
}

//Destroy WASAPI data struct
void wasapi_DestroyDataInterface(WASAPI_DATA *WASAPI)
{
	VirtualFree(WASAPI, 0, MEM_RELEASE);
}

//Set/reset WASAPI data struct
void wasapi_InitDataInterface(WASAPI_DATA &Interface)
{
	Interface.DeviceEnumerator = nullptr;
	Interface.AudioDevice = nullptr;
	Interface.AudioClient = nullptr;
	Interface.AudioRenderClient = nullptr;
	Interface.UsingDefaultDevice = false;
	Interface.DeviceState = Stopped;
	Interface.DeviceReady = false;
	Interface.DeviceFlags = 0;
	Interface.UsingDefaultWaveFormat = false;
	Interface.OutputBufferFrames = 0;
	Interface.OutputBufferPeriod = 0;
	Interface.OutputLatency = 0;
	Interface.RenderEvent = 0;

	//TODO: Windows macro to RtlZeroMemory, worth using?
	ZeroMemory(&Interface.OutputWaveFormat, sizeof(WAVEFORMATEXTENSIBLE));
}

//Print default audio endpoint
void wasapi_PrintDefaultDevice(HRESULT &HR, IMMDevice *Device)
{
	IPropertyStore* DevicePropertyStore = nullptr;
	PROPVARIANT DeviceName;
	LPWSTR DeviceID = nullptr;

	HR = Device->GetId(&DeviceID);
	HR_TO_RETURN(HR, "Failed to get device ID for printing", NONE);

	HR = Device->OpenPropertyStore(STGM_READ, &DevicePropertyStore);
	HR_TO_RETURN(HR, "Failed to get device for printing", NONE);

	HR = DevicePropertyStore->GetValue(PKEY_Device_FriendlyName, &DeviceName);
	HR_TO_RETURN(HR, "Failed to get device value for printing", NONE);

	debug_PrintLine(Console, "Audio Endpoint\tUsing default endpoint:\t\t\t%S\t[%S]", DeviceName.pwszVal, DeviceID);
}

//Print waveformat information
void wasapi_PrintWaveFormat(WAVEFORMATEXTENSIBLE &WaveFormat)
{   
	debug_PrintLine(Console, "\tCurrent Format\tFormat:\t\t\t\t\t%s", wasapi_FormatTagString(WaveFormat.Format.wFormatTag, WaveFormat.SubFormat));
	debug_PrintLine(Console, "\tCurrent Format\tChannels:\t\t\t\t%u", WaveFormat.Format.nChannels);
	debug_PrintLine(Console, "\tCurrent Format\tChannel Mask:\t\t\t\t%s", wasapi_ChannelMaskTagString(WaveFormat.dwChannelMask));
	debug_PrintLine(Console, "\tCurrent Format\tSample Rate:\t\t\t\t%u", WaveFormat.Format.nSamplesPerSec);
	debug_PrintLine(Console, "\tCurrent Format\tAverage bytes per second:\t\t%li bytes\t%f kilobytes\t%f megabytes", WaveFormat.Format.nAvgBytesPerSec, Kilobytes(WaveFormat.Format.nAvgBytesPerSec), Megabytes(WaveFormat.Format.nAvgBytesPerSec));
	debug_PrintLine(Console, "\tCurrent Format\tBlock alignment:\t\t\t%u", WaveFormat.Format.nBlockAlign);
	debug_PrintLine(Console, "\tCurrent Format\tBit Depth:\t\t\t\t%u", WaveFormat.Format.wBitsPerSample);
	debug_PrintLine(Console, "\tCurrent Format\tBits per sample:\t\t\t%u", WaveFormat.Samples.wValidBitsPerSample);
	debug_PrintLine(Console, "\tCurrent Format\tSize for additional data:\t\t%u", WaveFormat.Format.cbSize);
}

//Get default audio endpoint
bool wasapi_GetDefaultDevice(HRESULT &HR, WASAPI_DATA &Interface, bool PrintDefaultDevice)
{
	HR = Interface.DeviceEnumerator->GetDefaultAudioEndpoint(eRender, eConsole, &Interface.AudioDevice);
	HR_TO_RETURN(HR, "Failed to get default audio endpoint", false);
	
	if(PrintDefaultDevice)
	{
	    wasapi_PrintDefaultDevice(HR, Interface.AudioDevice);
	}

	HR = Interface.AudioDevice->Activate(__uuidof(IAudioClient), CLSCTX_ALL, NULL, (void**) &Interface.AudioClient);
	HR_TO_RETURN(HR, "Failed to activate audio endpoint", false);
	
	return true;
}

//Get default waveformat or use user defined one
bool wasapi_GetWaveFormat(HRESULT &HR, WASAPI_DATA &Interface, WAVEFORMATEXTENSIBLE *Custom, bool PrintDefaultWaveFormat)
{
	if(Custom == nullptr)
	{
		WAVEFORMATEX *DeviceWaveFormat = nullptr;

		HR = Interface.AudioClient->GetMixFormat(&DeviceWaveFormat);
    	HR_TO_RETURN(HR, "Failed to get default wave format for audio client", false);

		Interface.OutputWaveFormat = (WAVEFORMATEXTENSIBLE *) DeviceWaveFormat;

		if(PrintDefaultWaveFormat)
		{
			wasapi_PrintWaveFormat(*Interface.OutputWaveFormat);
		}

		return true;
	}

	//If user format is wrong then call IsFormatSupported to get the closest match
	WAVEFORMATEXTENSIBLE *Adjusted;

	//TODO: Finish this whole block (what to do when AUDCLNT_E_UNSUPPORTED_FORMAT)
	HR = Interface.AudioClient->IsFormatSupported(AUDCLNT_SHAREMODE_SHARED, (WAVEFORMATEX *) &Custom, (WAVEFORMATEX **) &Adjusted);
	//HR_TO_RETURN(HR, "Failed to check user format", NONE);
	
	if(HR == AUDCLNT_E_UNSUPPORTED_FORMAT)
	{
		debug_PrintLine(Console, "\tCurrent Format\tCannot use this format in exclusive mode");
	}

	else if(HR == S_FALSE)
	{
		debug_PrintLine(Console, "\tCurrent Format\tUsing adjusted format");
	}

	else
	{
		debug_PrintLine(Console, "\tCurrent Format\tUser format is valid");
	}

	Interface.OutputWaveFormat = (WAVEFORMATEXTENSIBLE *) Adjusted;

	if(PrintDefaultWaveFormat)
	{
		wasapi_PrintWaveFormat(*Interface.OutputWaveFormat);
	}

	return false;
}

//TODO: Add verbosity controls to print certain properties
//Initialise WASAPI device 
bool wasapi_InitDevice(HRESULT &HR, WASAPI_DATA &Interface)
{
	//TODO: COINIT_MULTITHREADED or COINIT_SPEED_OVER_MEMORY?
	HR = CoInitializeEx(0, COINIT_MULTITHREADED);
	HR_TO_RETURN(HR, "Failed to initialise COM", false);

	HR = CoCreateInstance(__uuidof(MMDeviceEnumerator), NULL, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator), (void**) &Interface.DeviceEnumerator);
	HR_TO_RETURN(HR, "Failed to create device COM", false);

	if((Interface.UsingDefaultDevice = wasapi_GetDefaultDevice(HR, Interface, true)) == false)
	{
		return false;
	}

	WAVEFORMATEXTENSIBLE *UserWaveFormat = (WAVEFORMATEXTENSIBLE *) VirtualAlloc(0, (sizeof *UserWaveFormat), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);	
	
	UserWaveFormat->Format.cbSize = sizeof(WAVEFORMATEXTENSIBLE);
	UserWaveFormat->Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE;
	UserWaveFormat->Format.nChannels = 2;
	UserWaveFormat->Format.wBitsPerSample = 8 * sizeof(f32);
	UserWaveFormat->Format.nSamplesPerSec = 48000;
	UserWaveFormat->Samples.wValidBitsPerSample = 8 * sizeof(f32);
	UserWaveFormat->Format.nBlockAlign = (UserWaveFormat->Format.wBitsPerSample / 8) * UserWaveFormat->Format.nChannels;
	UserWaveFormat->Format.nAvgBytesPerSec = UserWaveFormat->Format.nSamplesPerSec * UserWaveFormat->Format.nBlockAlign;
	UserWaveFormat->dwChannelMask = KSAUDIO_SPEAKER_STEREO;
	UserWaveFormat->SubFormat = KSDATAFORMAT_SUBTYPE_IEEE_FLOAT;

	if((Interface.UsingDefaultWaveFormat = wasapi_GetWaveFormat(HR, Interface, nullptr, true)) == true)
	{
		debug_PrintLine(Console, "\tWASAPI: Using default output wave format");
	}
	else
	{
		debug_PrintLine(Console, "\tWASAPI: Using user defined output wave format");
	}

	VirtualFree(UserWaveFormat, 0, MEM_RELEASE);
	
	//Get device period
	REFERENCE_TIME DevicePeriod = 0;
	REFERENCE_TIME DevicePeriodMin = 0;

	HR = Interface.AudioClient->GetDevicePeriod(&DevicePeriod, &DevicePeriodMin);
	HR_TO_RETURN(HR, "Failed to get device period for callback buffer", false);

	debug_PrintLine(Console, "\tDevice\t\tBuffer period:\t\t\t\t%lli hns", DevicePeriod);
	debug_PrintLine(Console, "\tDevice\t\tMinimum buffer period:\t\t\t%lli hns", DevicePeriodMin);
	
	//Outbut buffer device period
	f64 DevicePeriodInSeconds;
	DevicePeriodInSeconds = DevicePeriod / (10000.0 * 1000.0);
	Interface.OutputBufferPeriod = (Interface.OutputWaveFormat->Format.nSamplesPerSec * DevicePeriodInSeconds + 0.5);
	
	debug_PrintLine(Console, "\tDevice\t\tOutput buffer period:\t\t\t%u s", Interface.OutputBufferPeriod);
	Interface.DeviceFlags = AUDCLNT_STREAMFLAGS_EVENTCALLBACK;

	HR = Interface.AudioClient->Initialize(AUDCLNT_SHAREMODE_SHARED, Interface.DeviceFlags, DevicePeriod, 0, &Interface.OutputWaveFormat->Format, nullptr);
	HR_TO_RETURN(HR, "Failed to initialise audio client", false);
	
	Interface.RenderEvent = CreateEvent(nullptr, false, false, nullptr);
	
	if(!Interface.RenderEvent)
	{
		HR_TO_RETURN(HR, "Failed to create event", false);
	}

	HR = Interface.AudioClient->SetEventHandle(Interface.RenderEvent);
	HR_TO_RETURN(HR, "Failed to set rendering event", false);

	HR = Interface.AudioClient->GetBufferSize(&Interface.OutputBufferFrames);
	HR_TO_RETURN(HR, "Failed to get maximum read buffer size for audio client", false);

	debug_PrintLine(Console, "\tDevice\t\tBuffer size:\t\t\t\t%u frames", Interface.OutputBufferFrames);

	REFERENCE_TIME StreamLatency = 0;
	HR = Interface.AudioClient->GetStreamLatency(&StreamLatency);
	Interface.OutputLatency = (1000 * StreamLatency) / REF_TIMES_PER_SECOND;

	debug_PrintLine(Console, "\tDevice\t\tOutput latency:\t\t\t\t%f frames", Interface.OutputLatency);

	HR = Interface.AudioClient->GetService(__uuidof(IAudioRenderClient), (void**) &Interface.AudioRenderClient);
	HR_TO_RETURN(HR, "Failed to assign client to render client", false);	

	HR = Interface.AudioClient->Reset();
	HR_TO_RETURN(HR, "Failed to reset audio client before playback", false);

	HR = Interface.AudioClient->Start();
	HR_TO_RETURN(HR, "Failed to start audio client", false);

	Interface.DeviceState = Playing;
	debug_PrintLine(Console, "\tDevice\t\tState:\t\t\t\t\tPlaying");

	return true;
}

//Remove WASAPI device
void wasapi_DeinitDevice(WASAPI_DATA &Interface)
{
	if(Interface.AudioRenderClient)
	{
		Interface.AudioRenderClient->Release();
	}

	if(Interface.AudioClient)
	{
		Interface.AudioClient->Reset();
		Interface.AudioClient->Stop();
		Interface.AudioClient->Release();
	}

	if(Interface.AudioDevice)
	{
		Interface.AudioDevice->Release();
	}

	if(Interface.RenderEvent)
	{
		CloseHandle(Interface.RenderEvent);
		Interface.RenderEvent = nullptr;
	}

	wasapi_InitDataInterface(Interface);
	CoUninitialize();
}

//Fill WASAPI buffer with callback to audio renderer
void wasapi_FillBuffer(WASAPI_DATA &Interface, u32 FramesToWrite, BYTE *Data, OSCILLATOR * Osc, f32 Amplitude)
{
	f32 *FloatData = reinterpret_cast<f32 *>(Data);

	for(u32 FrameIndex = 0; FrameIndex < FramesToWrite; ++FrameIndex) 
	{
    	f32 CurrentSample = Osc->Tick(Osc) * Amplitude;

    	for(int ChannelIndex = 0; ChannelIndex < Interface.OutputWaveFormat->Format.nChannels; ++ChannelIndex) 
		{
    	    *FloatData++ = CurrentSample;
    	}      
	}
}


#endif