#ifndef polar_h
#define polar_h

//TODO: Finish rest of the comments

/*                  */
/*  WASAPI code     */
/*                  */

//TODO: Move this to an "internal" header, not the polar.h (ie, private)
//WASAPI includes
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
	f64 OutputBufferPeriod;
	u64 OutputLatency;

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


//Prototypes
WASAPI_DATA *wasapi_InterfaceCreate();                                                                                  //Create struct for WASAPI properties including output/rendering device info
void wasapi_InterfaceDestroy(WASAPI_DATA *WASAPI);                                                                      //Destroy WASAPI data struct
void wasapi_InterfaceInit(WASAPI_DATA &Interface);                                                                      //Set/reset WASAPI data struct
void wasapi_DevicePrint(HRESULT &HR, IMMDevice *Device);                                                                //Print default audio endpoint
void wasapi_FormatPrint(WAVEFORMATEXTENSIBLE &WaveFormat);                                                              //Print waveformat information
bool wasapi_DeviceGetDefault(HRESULT &HR, WASAPI_DATA &Interface, bool PrintDefaultDevice);                             //Get default audio endpoint
bool wasapi_FormatGet(HRESULT &HR, WASAPI_DATA &Interface, WAVEFORMATEXTENSIBLE *Custom, bool PrintDefaultWaveFormat);  //Get default waveformat or use user defined one
bool wasapi_DeviceInit(HRESULT &HR, WASAPI_DATA &Interface);                                                            //Initialise WASAPI device
void wasapi_DeviceDeInit(WASAPI_DATA &Interface);                                                                       //Release WASAPI devices


/*                  */
/*  Platform code   */
/*                  */

//Prototypes
//WASAPI
WASAPI_DATA *polar_WASAPI_Create(WASAPI_BUFFER &Buffer);                        //Create and initialise WASAPI struct
void polar_WASAPI_Destroy(WASAPI_DATA *WASAPI);                                 //Remove WASAPI struct
void polar_WASAPI_BufferGet(WASAPI_DATA *WASAPI, WASAPI_BUFFER &Buffer);        //Get WASAPI buffer and the maxium samples to fill
void polar_WASAPI_BufferRelease(WASAPI_DATA *WASAPI, WASAPI_BUFFER &Buffer);    //Release byte buffer after the rendering loop


/*                  */
/*  Rendering code  */
/*                  */

//Defines
//64 bit max size
//TODO: Check on x86 builds
#define WAV_FILE_MAX_SIZE  ((u64)0xFFFFFFFFFFFFFFFF)

//Structs
typedef struct POLAR_DATA       //Struct to hold platform specific audio API important engine properties
{
	//TODO: Create union for different audio API (CoreAudio)
	WASAPI_DATA *WASAPI;        //WASAPI data
	WASAPI_BUFFER Buffer;       //Float and BYTE buffers for rendering
	u32 BufferFrames;			//Frame count for output buffer
	u16 Channels;               //Engine current channels
	u32 SampleRate;             //Engine current sampling rate
	u16 BitRate;                //Engine current bitrate
} POLAR_DATA;

typedef struct POLAR_WAV_HEADER //WAV file specification "http://www-mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html"
{
	u16 AudioFormat;		    //1 for WAVE_FORMAT_PCM, 3 for WAVE_FORMAT_IEEE_FLOAT
	u16 NumChannels;		    //2
	u32 SampleRate;			    //192000
	u32 ByteRate;			    //SampleRate * NumChannels * BitsPerSample/8
	u16 BlockAlign;			    //NumChannels * BitsPerSample/8
	u16 BitsPerSample;		    //32
	u64 DataChunkDataSize;	    //Overall size of the "data" chunk
	u64 DataChunkDataStart;	    //Starting byte of the data chunk
} POLAR_WAV_HEADER;

typedef struct POLAR_WAV
{
	FILE *WAVFile;              //Handle to a file
	const char *Path;			//Path to a file
	POLAR_WAV_HEADER WAVHeader; //Struct to store WAV header properties
    //TODO: Support i16/i32 data
	f32 *Data;                  //Floating point sample buffer
	u64 TotalSampleCount;       //Total samples in a file when read
} POLAR_WAV;

//Protypes
//File writing
POLAR_WAV *polar_render_WAVWriteCreate(const char *FilePath, POLAR_DATA *Engine);
void polar_render_WAVWriteDestroy(POLAR_WAV *File);
bool polar_render_WAVWriteHeader(POLAR_WAV *File, POLAR_DATA *Engine);
size_t polar_render_WAVWriteRaw(POLAR_WAV *File, size_t BytesToWrite, const void *FileData);
u64 polar_render_WAVWriteFloat(POLAR_WAV *File, u64 SamplesToWrite, const void *FileData);
u32 polar_render_RIFFChunkRound(u64 RIFFChunkSize);
u32 polar_render_DataChunkRound(u64 DataChunkSize);

//Rendering
f32 polar_render_PanPositionGet(u16 Position, f32 Amplitude, f32 PanFactor);    //Calculate stereo pan position
void polar_render_BufferFill(u16 ChannelCount, u32 FramesToWrite, f32 *SampleBuffer, BYTE *ByteBuffer, f32 *FileSamples, OSCILLATOR *Osc, f32 Amplitude, f32 PanValue);
void polar_render_BufferCopy(POLAR_DATA &Engine, POLAR_WAV *File, OSCILLATOR *Osc, f32 Amplitude, f32 PanValue);

#endif