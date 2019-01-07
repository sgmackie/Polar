#ifndef win32_polar_h
#define win32_polar_h

//TODO: Remove atomic for InterlockedExchanges
#include <Windows.h>
#include <atomic>

/*                   */
/*  Windows code     */
/*                   */

#define WIN32_MAX_FILE_PATH MAX_PATH	//Max file path length

//Structs
typedef struct WIN32_REPLAY_BUFFER
{
    HANDLE File;
    HANDLE MemoryMap;
    char FileName[WIN32_MAX_FILE_PATH];
    void *MemoryBlock;
} WIN32_REPLAY_BUFFER;


typedef struct WIN32_ENGINE_CODE
{
    HMODULE EngineDLL;
    FILETIME DLLLastWriteTime;
	bool IsDLLValid;
	polar_render_Update *UpdateCallback;
	polar_render_Render *RenderCallback;
} WIN32_ENGINE_CODE;

typedef struct WIN32_STATE
{
	//State data
	u64 TotalSize;
	void *EngineMemoryBlock;
	WIN32_REPLAY_BUFFER ReplayBuffers[4];	//Switched from 4

	//Store .exe
    char EXEPath[WIN32_MAX_FILE_PATH];
    char *EXEFileName;

	//Store code .dlls
	char EngineSourceCodePath[WIN32_MAX_FILE_PATH];
	char TempEngineSourceCodePath[WIN32_MAX_FILE_PATH];

	//File handle for state recording
	HANDLE RecordingHandle;
    i32 InputRecordingIndex;

	//File handle for state playback
    HANDLE PlaybackHandle;
    i32 InputPlayingIndex;
} WIN32_STATE;

typedef struct WIN32_WINDOW_DIMENSIONS
{
    i32 Width;
    i32 Height;
} WIN32_WINDOW_DIMENSIONS;

typedef struct WIN32_OFFSCREEN_BUFFER
{
    BITMAPINFO BitmapInfo;
    void *Data;
    i32 Width;
    i32 Height;
    i32 Pitch;
    i32 BytesPerPixel;
} WIN32_OFFSCREEN_BUFFER;

//Prototypes
//File handling
internal void win32_EXEFileNameGet(WIN32_STATE *State);														//Find file name of current application
internal void win32_BuildEXEPathGet(WIN32_STATE *State, const char *FileName, char *Path);					//Get file path
internal void win32_InputFilePathGet(WIN32_STATE *State, bool InputStream, i32 SlotIndex, char *Path);		//"Print" to a custom text file for looping editss
internal FILETIME win32_LastWriteTimeGet(char *Filename);													//Find the last time a file was written to
internal WIN32_ENGINE_CODE win32_EngineCodeLoad(char *SourceDLLName, char *TempDLLName);					//Load dll for dynamic render code
internal void win32_EngineCodeUnload(WIN32_ENGINE_CODE *EngineCode);										//Unload .dll

//State recording
internal WIN32_REPLAY_BUFFER *win32_ReplayBufferGet(WIN32_STATE *State, u32 Index);							//Get the current replay buffer
internal void win32_StateRecordingStart(WIN32_STATE *State, i32 InputRecordingIndex);						//Start recording the engine state
internal void win32_StateRecordingStop(WIN32_STATE *State);													//Stop recording the engine state
internal void win32_StatePlaybackStart(WIN32_STATE *State, i32 InputPlayingIndex);							//Start playback of the recorded state
internal void win32_StatePlaybackStop(WIN32_STATE *State);													//Stop playback of the recorded state

//Input state handlings
internal void win32_InputMessageProcess(POLAR_INPUT_STATE *NewState, bool IsDown);							//Process inputs when released
internal void win32_WindowMessageProcess(WIN32_STATE *State, POLAR_INPUT_CONTROLLER *KeyboardController);	//Process the window message queue
internal void win32_InputRecord(WIN32_STATE *State, POLAR_INPUT *NewInput);									//Start recording input parameters						
internal void win32_InputPlayback(WIN32_STATE *State, POLAR_INPUT *NewInput);								//Start playback of recorded inputs

//Display rendering
internal WIN32_WINDOW_DIMENSIONS win32_WindowDimensionsGet(HWND Window);									//Get the current window dimensions
internal void win32_BitmapBufferResize(WIN32_OFFSCREEN_BUFFER *Buffer, i32 TargetWidth, i32 TargetHeight);	//Resize input buffer to a specific width and height
internal void win32_DisplayBufferInWindow(WIN32_OFFSCREEN_BUFFER *Buffer, HDC DeviceContext);				//Copy buffer to a specified device

//Performance timing
internal LARGE_INTEGER win32_WallClockGet();																//Get the current position of the perfomance counter (https://msdn.microsoft.com/en-us/library/windows/desktop/ms644904(v=vs.85).aspx)
internal f32 win32_SecondsElapsedGet(LARGE_INTEGER Start, LARGE_INTEGER End);								//Determine the amount of seconfs elapised against the perfomance counter

/*                  */
/*  WASAPI code     */
/*                  */

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
		char HRBuffer[256];													\
		OutputDebugString(HRBuffer);										\
		sprintf_s(HRBuffer, Text "\t[%s]\n", wasapi_HRString(Result));   	\
		return Type;								                        \
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


//Prototypes
WASAPI_DATA *wasapi_InterfaceCreate();                                                                                  			//Create struct for WASAPI properties including output/rendering device info
void wasapi_InterfaceDestroy(WASAPI_DATA *WASAPI);                                                                      			//Destroy WASAPI data struct
void wasapi_InterfaceInit(WASAPI_DATA &Interface);                                                                      			//Set/reset WASAPI data struct
bool wasapi_DeviceInit(HRESULT &HR, WASAPI_DATA &Interface);                                                            			//Initialise WASAPI device
void wasapi_DeviceDeInit(WASAPI_DATA &Interface);                                                                       			//Release WASAPI devices
internal void wasapi_DevicePrint(HRESULT &HR, IMMDevice *Device);                                                                	//Print default audio endpoint
internal bool wasapi_DeviceGetDefault(HRESULT &HR, WASAPI_DATA &Interface);                             							//Get default audio endpoint
internal bool wasapi_FormatGet(HRESULT &HR, WASAPI_DATA &Interface, WAVEFORMATEXTENSIBLE *Custom);  								//Get default waveformat or use user defined one


#include "win32_internal_WASAPI.cpp"



/*                  */
/*  Platform code   */
/*                  */

//Prototypes
//WASAPI
internal WASAPI_DATA *win32_WASAPI_Create(POLAR_BUFFER &Buffer, u32 UserSampleRate, u16 UserBitRate, u16 UserChannels);				//Create and initialise WASAPI struct
internal void win32_WASAPI_Destroy(WASAPI_DATA *WASAPI);                                 											//Remove WASAPI struct
internal void win32_WASAPI_BufferGet(WASAPI_DATA *WASAPI, POLAR_BUFFER &Buffer);        											//Get WASAPI buffer and the maxium samples to fill
internal void win32_WASAPI_BufferRelease(WASAPI_DATA *WASAPI, POLAR_BUFFER &Buffer);    											//Release byte buffer after the rendering loop




#endif