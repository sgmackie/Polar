#ifndef linux_polar_h
#define linux_polar_h

/*                  */
/*  Linux code      */
/*                  */

//System
#include <sys/mman.h>   //Memory allocations (mmap, munmap)
#include <sys/stat.h>   //Stat struct for file loading
#include <dlfcn.h>      //Dynanamic linking (dlopen)

//X11
#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <X11/Xutil.h>

//Defines
//Keycodes found using xev command to print keycode
#define KEYCODE_W           25
#define KEYCODE_A           38
#define KEYCODE_S           39
#define KEYCODE_D           40
#define KEYCODE_Q           24
#define KEYCODE_E           26
#define KEYCODE_UP          111
#define KEYCODE_DOWN        116
#define KEYCODE_LEFT        113
#define KEYCODE_RIGHT       114
#define KEYCODE_ESCAPE      9
#define KEYCODE_ENTER       36
#define KEYCODE_SPACE       65
#define KEYCODE_P           33
#define KEYCODE_L           46

#define KEYCODE_SHIFT_L     50
#define KEYCODE_SHIFT_R     62
#define KEYCODE_CTRL_L      37
#define KEYCODE_CTRL_R      105
#define KEYCODE_ALT_L       64
#define KEYCODE_ALT_R       108
#define KEYCODE_SUPER       133

#define KEYCODE_PLUS        21
#define KEYCODE_MINUS       20

#define KEYCODE_F1          67
#define KEYCODE_F10         76
#define KEYCODE_F11         95
#define KEYCODE_F12         96

#define KEYCODE_SHIFT_MASK  0x01
#define KEYCODE_CTRL_MASK   0x04
#define KEYCODE_ALT_MASK    0x08


//Structs
typedef struct LINUX_OFFSCREEN_BUFFER
{
    void *Data;
    i32 Width;
    i32 Height;
    i32 Pitch;
    i32 BytesPerPixel;
} LINUX_OFFSCREEN_BUFFER;

typedef struct LINUX_ENGINE_CODE
{
    void *EngineHandle;
    ino_t EngineID;

	bool IsDLValid;
	polar_render_Update *UpdateAndRender;
} LINUX_ENGINE_CODE;


#define LINUX_MAX_FILE_PATH 512

typedef struct LINUX_STATE
{
	//State data
	u64 TotalSize;
	void *EngineMemoryBlock;

	//Store .exe
    char EXEPath[LINUX_MAX_FILE_PATH];
    char *EXEFileName;

	//Store code .so
	char EngineSourceCodePath[LINUX_MAX_FILE_PATH];

	//File handle for state recording
	i32 RecordingHandle;
    i32 InputRecordingIndex;

	//File handle for state playback
    i32 PlaybackHandle;
    i32 InputPlayingIndex;
} LINUX_STATE;


//Prototypes
internal void linux_EXEFileNameGet(LINUX_STATE *State);                                             //Find file name of current application
internal void linux_BuildEXEPathGet(LINUX_STATE *State, const char *FileName, char *Path);          //Get file path
internal ino_t linux_FileIDGet(char *FileName);                                                     //Record file attributes using stat ("http://pubs.opengroup.org/onlinepubs/000095399/basedefs/sys/stat.h.html") 
internal void *linux_LibraryOpen(const char *Library);                                              //Wrap dlopen with error handling
internal void linux_LibraryClose(void *Handle);                                                     //Wrap dlclose
internal void *linux_ExternalFunctionLoad(void *Library, const char *Name);                         //Wrap dlsym with error handling
internal bool linux_EngineCodeLoad(LINUX_ENGINE_CODE *EngineCode, char *DLName, ino_t FileID);      //Check if file ID's match and load engine code if not
internal void linux_EngineCodeUnload(LINUX_ENGINE_CODE *EngineCode);                                //Unload engine code

internal LINUX_OFFSCREEN_BUFFER linux_WindowDimensionsGet(u32 Width, u32 Height);					//Create display buffer
internal void linux_InputMessageProcess(POLAR_INPUT_STATE *NewState, bool IsDown);					//Process inputs when released
internal void linux_WindowMessageProcess(LINUX_STATE *State, Display *Display, Window Window, Atom WmDeleteWindow, POLAR_INPUT_CONTROLLER *KeyboardController); //Process the window message queue

/*                  */
/*  ALSA code       */
/*                  */

//Includes
#include <alsa/asoundlib.h>
#include <alsa/pcm.h>

//Macros
#define NONE    //Blank space for returning nothing in void functions

//ALSA Error code print and return
#define ERR_TO_RETURN(Result, Text, Type)				                    \
	if(Result < 0)								                            \
	{												                        \
		printf(Text "\t[%s]\n", snd_strerror(Result));   	                \
		return Type;								                        \
	}

//Structs
typedef struct ALSA_DATA
{
    snd_pcm_sframes_t FramesWritten;
    snd_pcm_t *Device;


    u32 SampleRate;
    u8 ALSAResample;
    u16 Channels;
    u32 LatencyInMS;
    u64 BufferSize;
    u64 PeriodSize;
} ALSA_DATA;

//Prototypes
ALSA_DATA *linux_ALSA_Create(POLAR_BUFFER &Buffer, u32 UserSampleRate, u16 UserChannels, u32 UserLatency);      //ALSA setup
void linux_ALSA_Destroy(ALSA_DATA *Result, POLAR_BUFFER &Buffer);                                               //ALSA destroy

#endif