#ifndef polar_WASAPI_h
#define polar_WASAPI_h

//WASAPI
#include <audioclient.h>                    //WASAPI
#include <mmdeviceapi.h>                    //Audio endpoints
#include <Functiondiscoverykeys_devpkey.h>  //Used for getting "FriendlyNames" from audio endpoints
#include <avrt.h>

//Reference times as variable
global const u64 REF_TIMES_PER_SECOND = 10000000;

//Convert WASAPI HRESULT to printable string
global const TCHAR *wasapi_HRString(HRESULT Result)
{
	switch(Result)
	{
		case S_OK:										return TEXT("S_OK");
		case S_FALSE:									return TEXT("S_FALSE");
		case AUDCLNT_E_NOT_INITIALIZED:					return TEXT("AUDCLNT_E_NOT_INITIALIZED");
		case AUDCLNT_E_ALREADY_INITIALIZED:				return TEXT("AUDCLNT_E_ALREADY_INITIALIZED");
		case AUDCLNT_E_WRONG_ENDPOINT_TYPE:				return TEXT("AUDCLNT_E_WRONG_ENDPOINT_TYPE");
		case AUDCLNT_E_DEVICE_INVALIDATED:				return TEXT("AUDCLNT_E_DEVICE_INVALIDATED");
		case AUDCLNT_E_NOT_STOPPED:						return TEXT("AUDCLNT_E_NOT_STOPPED");
		case AUDCLNT_E_BUFFER_TOO_LARGE:				return TEXT("AUDCLNT_E_BUFFER_TOO_LARGE");
		case AUDCLNT_E_OUT_OF_ORDER:					return TEXT("AUDCLNT_E_OUT_OF_ORDER");
		case AUDCLNT_E_UNSUPPORTED_FORMAT:				return TEXT("AUDCLNT_E_UNSUPPORTED_FORMAT");
		case AUDCLNT_E_INVALID_SIZE:					return TEXT("AUDCLNT_E_INVALID_SIZE");
		case AUDCLNT_E_DEVICE_IN_USE:					return TEXT("AUDCLNT_E_DEVICE_IN_USE");
		case AUDCLNT_E_BUFFER_OPERATION_PENDING:		return TEXT("AUDCLNT_E_BUFFER_OPERATION_PENDING");
		case AUDCLNT_E_THREAD_NOT_REGISTERED:			return TEXT("AUDCLNT_E_THREAD_NOT_REGISTERED");
		case AUDCLNT_E_EXCLUSIVE_MODE_NOT_ALLOWED:		return TEXT("AUDCLNT_E_EXCLUSIVE_MODE_NOT_ALLOWED");
		case AUDCLNT_E_ENDPOINT_CREATE_FAILED:			return TEXT("AUDCLNT_E_ENDPOINT_CREATE_FAILED");
		case AUDCLNT_E_SERVICE_NOT_RUNNING:				return TEXT("AUDCLNT_E_SERVICE_NOT_RUNNING");
		case AUDCLNT_E_EVENTHANDLE_NOT_EXPECTED:		return TEXT("AUDCLNT_E_EVENTHANDLE_NOT_EXPECTED");
		case AUDCLNT_E_EXCLUSIVE_MODE_ONLY:				return TEXT("AUDCLNT_E_EXCLUSIVE_MODE_ONLY");
		case AUDCLNT_E_BUFDURATION_PERIOD_NOT_EQUAL:	return TEXT("AUDCLNT_E_BUFDURATION_PERIOD_NOT_EQUAL");
		case AUDCLNT_E_EVENTHANDLE_NOT_SET:				return TEXT("AUDCLNT_E_EVENTHANDLE_NOT_SET");
		case AUDCLNT_E_INCORRECT_BUFFER_SIZE:			return TEXT("AUDCLNT_E_INCORRECT_BUFFER_SIZE");
		case AUDCLNT_E_BUFFER_SIZE_ERROR:				return TEXT("AUDCLNT_E_BUFFER_SIZE_ERROR");
		case AUDCLNT_E_CPUUSAGE_EXCEEDED:				return TEXT("AUDCLNT_E_CPUUSAGE_EXCEEDED");
		case AUDCLNT_E_BUFFER_ERROR:					return TEXT("AUDCLNT_E_BUFFER_ERROR");
		case AUDCLNT_E_BUFFER_SIZE_NOT_ALIGNED:			return TEXT("AUDCLNT_E_BUFFER_SIZE_NOT_ALIGNED");
		case AUDCLNT_E_INVALID_DEVICE_PERIOD:			return TEXT("AUDCLNT_E_INVALID_DEVICE_PERIOD");
		case AUDCLNT_E_INVALID_STREAM_FLAG:				return TEXT("AUDCLNT_E_INVALID_STREAM_FLAG");
		case AUDCLNT_E_ENDPOINT_OFFLOAD_NOT_CAPABLE:	return TEXT("AUDCLNT_E_ENDPOINT_OFFLOAD_NOT_CAPABLE");
		case AUDCLNT_E_OUT_OF_OFFLOAD_RESOURCES:		return TEXT("AUDCLNT_E_OUT_OF_OFFLOAD_RESOURCES");
		case AUDCLNT_E_OFFLOAD_MODE_ONLY:				return TEXT("AUDCLNT_E_OFFLOAD_MODE_ONLY");
		case AUDCLNT_E_NONOFFLOAD_MODE_ONLY:			return TEXT("AUDCLNT_E_NONOFFLOAD_MODE_ONLY");
		case AUDCLNT_E_RESOURCES_INVALIDATED:			return TEXT("AUDCLNT_E_RESOURCES_INVALIDATED");
		case AUDCLNT_E_RAW_MODE_UNSUPPORTED:			return TEXT("AUDCLNT_E_RAW_MODE_UNSUPPORTED");
		case REGDB_E_CLASSNOTREG:						return TEXT("REGDB_E_CLASSNOTREG");
		case CLASS_E_NOAGGREGATION:						return TEXT("CLASS_E_NOAGGREGATION");
		case E_NOINTERFACE:								return TEXT("E_NOINTERFACE");
		case E_POINTER:									return TEXT("E_POINTER");
		case E_INVALIDARG:								return TEXT("E_INVALIDARG");
		case E_OUTOFMEMORY:								return TEXT("E_OUTOFMEMORY");
		default:										return TEXT("UNKNOWN");
	}
}

#define NONE    //Blank space for returning nothing in void functions

//Use print and return on HRESULT codes
#define HR_TO_RETURN(Result, Text, Type)				                    \
	if(FAILED(Result))								                        \
	{												                        \
		debug_PrintLine(Text "\t[%s]", wasapi_HRString(Result));   \
		return Type;								                        \
	}

//Convert waveformat flags to text
global const TCHAR *wasapi_FormatTagString(WORD Format, GUID SubFormat)
{
    switch(Format)
    {
		case WAVE_FORMAT_UNKNOWN:						return TEXT("WAVE_FORMAT_UNKNOWN");
    	case WAVE_FORMAT_IEEE_FLOAT:					return TEXT("WAVE_FORMAT_IEEE_FLOAT");
    	case WAVE_FORMAT_PCM:							return TEXT("WAVE_FORMAT_PCM");
		case WAVE_FORMAT_ADPCM:							return TEXT("WAVE_FORMAT_ADPCM");
		case WAVE_FORMAT_FLAC:							return TEXT("WAVE_FORMAT_FLAC");
		case WAVE_FORMAT_EXTENSIBLE:
        {
            if(SubFormat == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT)
            {
            											return TEXT("WAVE_FORMAT_EXTENSIBLE || KSDATAFORMAT_SUBTYPE_IEEE_FLOAT");
            }
            else if(SubFormat == KSDATAFORMAT_SUBTYPE_PCM)
            {
            											return TEXT("WAVE_FORMAT_EXTENSIBLE || KSDATAFORMAT_SUBTYPE_PCM");
            }
        } 
        default:										return TEXT("UNKNOWN");
    }
}


//Convert channel configurations to text
global const TCHAR *wasapi_ChannelMaskTagString(WORD ChannelMask)
{
    switch(ChannelMask)
    {
        case KSAUDIO_SPEAKER_MONO:						return TEXT("KSAUDIO_SPEAKER_MONO");
        case KSAUDIO_SPEAKER_STEREO:					return TEXT("KSAUDIO_SPEAKER_STEREO");
        case KSAUDIO_SPEAKER_QUAD:						return TEXT("KSAUDIO_SPEAKER_QUAD");
        case KSAUDIO_SPEAKER_SURROUND:					return TEXT("KSAUDIO_SPEAKER_SURROUND");
        case KSAUDIO_SPEAKER_5POINT1:					return TEXT("KSAUDIO_SPEAKER_5POINT1");
        case KSAUDIO_SPEAKER_7POINT1:					return TEXT("KSAUDIO_SPEAKER_7POINT1");
        case KSAUDIO_SPEAKER_5POINT1_SURROUND:			return TEXT("KSAUDIO_SPEAKER_5POINT1_SURROUND");
        case KSAUDIO_SPEAKER_7POINT1_SURROUND:			return TEXT("KSAUDIO_SPEAKER_7POINT1_SURROUND");
        default:										return TEXT("UNKNOWN");
    }
}

//Store current state of WASAPI
enum WASAPI_STATE
{
	Stopped,
	Playing,
	Paused,
};

//TODO: Clean this up; if variables are being calculated, do they need to be stored? What is actually being used multiple times here?
//Structure containing common properties defined by WASAPI
typedef struct WASAPI_DATA
{
	//Windows error reporting
	HRESULT HR;

	//Device endpoints
	IMMDeviceEnumerator *DeviceEnumerator;
	IMMDevice *AudioDevice;

	//Rendering clients
	IAudioClient *AudioClient;
	IAudioRenderClient *AudioRenderClient;

	//Audio clock
	IAudioClock* AudioClock;

	//Device state
	bool UsingDefaultDevice;
	//TODO: Swap Atomic for InterlockedExchange
	std::atomic<WASAPI_STATE> DeviceState;
	bool DeviceReady;
	DWORD DeviceFlags;

	//Output properties
	WAVEFORMATEXTENSIBLE *OutputWaveFormat;
	bool UsingDefaultWaveFormat;
	u32 OutputBufferFrames;
	u32 OutputBufferPeriod;
	f32 OutputLatency;

	//Rendering state
	HANDLE RenderEvent;  
} WASAPI_DATA;

//BYTE buffer passed to WASAPI for rendering
typedef struct WASAPI_BUFFER
{
	u32 FramePadding;
	u32 FramesAvailable;
	f32 *SampleBuffer;
	BYTE *ByteBuffer;
} WASAPI_BUFFER;



typedef struct WASAPI_CLOCK
{
	UINT64 PositionFrequency;
	UINT64 PositionUnits;
} WASAPI_CLOCK;

//Prototypes
//Create struct for WASAPI properties including output/rendering device info
WASAPI_DATA *wasapi_CreateDataInterface();

//Destroy WASAPI data struct
void wasapi_DestroyDataInterface(WASAPI_DATA *WASAPI);

//Set/reset WASAPI data struct
void wasapi_InitDataInterface(WASAPI_DATA &Interface);

//Print default audio endpoint
void wasapi_PrintDefaultDevice(HRESULT &HR, IMMDevice *Device);

//Print waveformat information
void wasapi_PrintWaveFormat(WAVEFORMATEXTENSIBLE &WaveFormat);

//Get default audio endpoint
bool wasapi_GetDefaultDevice(HRESULT &HR, WASAPI_DATA &Interface, bool PrintDefaultDevice);

//Get default waveformat or use user defined one
bool wasapi_GetWaveFormat(HRESULT &HR, WASAPI_DATA &Interface, WAVEFORMATEXTENSIBLE *Custom, bool PrintDefaultWaveFormat);

//Initialise WASAPI device 
bool wasapi_InitDevice(HRESULT &HR, WASAPI_DATA &Interface);

//Remove WASAPI device
void wasapi_DeinitDevice(WASAPI_DATA &Interface);

#endif