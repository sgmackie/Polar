#ifndef win32_internal_WASAPI_cpp
#define win32_internal_WASAPI_cpp

//Create struct for WASAPI properties including output/rendering device info
WASAPI_DATA *wasapi_InterfaceCreate()
{
	WASAPI_DATA *Result = (WASAPI_DATA *) VirtualAlloc(0, (sizeof *Result), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
	return Result;
}

//Destroy WASAPI data struct
void wasapi_InterfaceDestroy(WASAPI_DATA *WASAPI)
{
	VirtualFree(WASAPI, 0, MEM_RELEASE);
}

//Set/reset WASAPI data struct
void wasapi_InterfaceInit(WASAPI_DATA &Interface)
{
	Interface.HR = S_FALSE;
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

	//?Windows macro to RtlZeroMemory
	ZeroMemory(&Interface.OutputWaveFormat, sizeof(WAVEFORMATEXTENSIBLE));
}

//Print default audio endpoint
internal void wasapi_DevicePrint(HRESULT &HR, IMMDevice *Device)
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

    char DeviceBuffer[256];
    sprintf_s(DeviceBuffer, "WASAPI: Using default endpoint:\t\t%S\t[%S]\n", DeviceName.pwszVal, DeviceID);
    OutputDebugString(DeviceBuffer);
}


//Get default audio endpoint
internal bool wasapi_DeviceGetDefault(HRESULT &HR, WASAPI_DATA &Interface)
{
	HR = Interface.DeviceEnumerator->GetDefaultAudioEndpoint(eRender, eConsole, &Interface.AudioDevice);
	HR_TO_RETURN(HR, "Failed to get default audio endpoint", false);

#if WASAPI_INFO
	wasapi_DevicePrint(HR, Interface.AudioDevice);
#endif

	HR = Interface.AudioDevice->Activate(__uuidof(IAudioClient), CLSCTX_ALL, NULL, (void**) &Interface.AudioClient);
	HR_TO_RETURN(HR, "Failed to activate audio endpoint", false);
	
	return true;
}

//Get default waveformat or use user defined one
internal bool wasapi_FormatGet(HRESULT &HR, WASAPI_DATA &Interface, WAVEFORMATEXTENSIBLE *Custom)
{
	if(Custom == nullptr)
	{
		WAVEFORMATEX *DeviceWaveFormat = nullptr;

		HR = Interface.AudioClient->GetMixFormat(&DeviceWaveFormat);
    	HR_TO_RETURN(HR, "Failed to get default wave format for audio client", false);

		Interface.OutputWaveFormat = (WAVEFORMATEXTENSIBLE *) DeviceWaveFormat;

		return true;
	}

	//If user format is wrong then call IsFormatSupported to get the closest match
	// WAVEFORMATEXTENSIBLE **Adjusted;

	WAVEFORMATEX *Adjusted = {};

	//TODO: Finish this whole block (what to do when AUDCLNT_E_UNSUPPORTED_FORMAT)
	HR = Interface.AudioClient->IsFormatSupported(AUDCLNT_SHAREMODE_SHARED, (WAVEFORMATEX *) &Custom, &Adjusted);
	//HR_TO_RETURN(HR, "Failed to check user format", NONE);

	if(HR == AUDCLNT_E_UNSUPPORTED_FORMAT)
	{
#if WASAPI_INFO		
		char FormatBuffer[256];
    	sprintf_s(FormatBuffer, "Current Format\tCannot use this format in exclusive mode");
    	OutputDebugString(FormatBuffer);
#endif
	}

	else if(HR == S_FALSE)
	{
#if WASAPI_INFO		
		char FormatBuffer[256];
    	sprintf_s(FormatBuffer, "Current Format\tUsing adjusted format");
    	OutputDebugString(FormatBuffer);
#endif
	}

	else
	{
#if WASAPI_INFO		
		char FormatBuffer[256];
    	sprintf_s(FormatBuffer, "Current Format\tUser format is valid");
    	OutputDebugString(FormatBuffer);
#endif
	}

	Interface.OutputWaveFormat = (WAVEFORMATEXTENSIBLE *) Adjusted;


	return false;
}

//TODO: Add verbosity controls to print certain properties
//Initialise WASAPI device 
bool wasapi_DeviceInit(HRESULT &HR, WASAPI_DATA &Interface, u32 UserSampleRate, u16 UserBitRate, u16 UserChannels)
{
	HR = CoInitializeEx(0, COINIT_MULTITHREADED);
	HR_TO_RETURN(HR, "Failed to initialise COM", false);

	HR = CoCreateInstance(__uuidof(MMDeviceEnumerator), NULL, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator), (void**) &Interface.DeviceEnumerator);
	HR_TO_RETURN(HR, "Failed to create device COM", false);

	if((Interface.UsingDefaultDevice = wasapi_DeviceGetDefault(HR, Interface)) == false)
	{
		return false;
	}

	if(UserSampleRate > 0 || UserBitRate > 0|| UserChannels > 0)
	{
		WAVEFORMATEXTENSIBLE *UserWaveFormat = (WAVEFORMATEXTENSIBLE *) VirtualAlloc(0, (sizeof *UserWaveFormat), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);	
	
		UserWaveFormat->Format.cbSize = sizeof(WAVEFORMATEXTENSIBLE);
		UserWaveFormat->Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE;
		UserWaveFormat->Format.nChannels = UserChannels;
		UserWaveFormat->Format.wBitsPerSample = UserBitRate * sizeof(f32);
		UserWaveFormat->Format.nSamplesPerSec = UserSampleRate;
		UserWaveFormat->Samples.wValidBitsPerSample = UserBitRate * sizeof(f32);
		UserWaveFormat->Format.nBlockAlign = (UserWaveFormat->Format.wBitsPerSample / 8) * UserWaveFormat->Format.nChannels;
		UserWaveFormat->Format.nAvgBytesPerSec = UserWaveFormat->Format.nSamplesPerSec * UserWaveFormat->Format.nBlockAlign;
		UserWaveFormat->dwChannelMask = KSAUDIO_SPEAKER_STEREO;
		UserWaveFormat->SubFormat = KSDATAFORMAT_SUBTYPE_IEEE_FLOAT;

		if((Interface.UsingDefaultWaveFormat = wasapi_FormatGet(HR, Interface, UserWaveFormat)) == true)
		{
		}
		else
		{
		}

		VirtualFree(UserWaveFormat, 0, MEM_RELEASE);
	}
	else
	{
		if((Interface.UsingDefaultWaveFormat = wasapi_FormatGet(HR, Interface, nullptr)) == true)
		{
		}	
	}


	//Get device period
	REFERENCE_TIME DevicePeriod = 0;
	REFERENCE_TIME DevicePeriodMin = 0;

	HR = Interface.AudioClient->GetDevicePeriod(&DevicePeriod, &DevicePeriodMin);
	HR_TO_RETURN(HR, "Failed to get device period for callback buffer", false);
	
	//Outbut buffer device period
	f64 DevicePeriodInSeconds;
	DevicePeriodInSeconds = DevicePeriod / (10000.0 * 1000.0);
	Interface.OutputBufferPeriod = (Interface.OutputWaveFormat->Format.nSamplesPerSec * DevicePeriodInSeconds + 0.5);
	
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

	REFERENCE_TIME StreamLatency = 0;
	HR = Interface.AudioClient->GetStreamLatency(&StreamLatency);
	Interface.OutputLatency = (1000 * StreamLatency) / REF_TIMES_PER_SECOND;

	HR = Interface.AudioClient->GetService(__uuidof(IAudioRenderClient), (void**) &Interface.AudioRenderClient);
	HR_TO_RETURN(HR, "Failed to assign client to render client", false);	

	HR = Interface.AudioClient->GetService(__uuidof(IAudioClock), (void**) &Interface.AudioClock);
	HR_TO_RETURN(HR, "Failed to assign clock to render client", false);	

	HR = Interface.AudioClient->Reset();
	HR_TO_RETURN(HR, "Failed to reset audio client before playback", false);

	HR = Interface.AudioClient->Start();
	HR_TO_RETURN(HR, "Failed to start audio client", false);

	Interface.DeviceState = Playing;

	return true;
}

//Remove WASAPI device
void wasapi_DeviceDeInit(WASAPI_DATA &Interface)
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

	if(Interface.AudioClock)
	{
		Interface.AudioClock->Release();
	}

	if(Interface.RenderEvent)
	{
		CloseHandle(Interface.RenderEvent);
		Interface.RenderEvent = nullptr;
	}

	wasapi_InterfaceInit(Interface);
	CoUninitialize();
}

#endif