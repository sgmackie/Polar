#include <stdio.h>
#include <string.h>

#include <sys/mman.h>   //mmap
#include <sys/stat.h>   //stat struct
#include <dlfcn.h>      //dynamic library

#include <alsa/asoundlib.h>
#include <alsa/pcm.h>


//Polar
#include "polar.h"

#include "../external/entropy/entropy.h"


#define NONE    //Blank space for returning nothing in void functions

//ALSA Error code print and return
#define ERR_TO_RETURN(Result, Text, Type)				                    \
	if(Result < 0)								                            \
	{												                        \
		printf(Text "\t[%s]\n", snd_strerror(Result));   	                \
		return Type;								                        \
	}




typedef struct ALSA_PROPERTIES
{
    u32 SampleRate;
    u8 ALSAResample;
    u16 Channels;
    u32 LatencyInMS;
    u64 BufferSize;
    u64 PeriodSize;
} ALSA_PROPERTIES;


//? Test function
internal void FILL(u16 ChannelCount, u32 FramesToWrite, f32 *SampleBuffer, void *DeviceBuffer, OSCILLATOR *Osc)
{	
	f32 CurrentSample = 0;
	f32 PanAmp = 0;

	//Increase frame counter by the number of channels
	for(u32 FrameIndex = 0; FrameIndex < FramesToWrite; FrameIndex += ChannelCount)
	{
		CurrentSample = (f32) Osc->Tick(Osc);

    	for(u16 ChannelIndex = 0; ChannelIndex < ChannelCount; ++ChannelIndex)
		{
            // PanAmp = polar_render_PanPositionGet(ChannelIndex, 0.4, 1);
            PanAmp = 0.3;

			SampleBuffer[FrameIndex + ChannelIndex] = CurrentSample * PanAmp;
		}
	}

	memcpy(DeviceBuffer, SampleBuffer, (sizeof(* SampleBuffer) * FramesToWrite));
}


typedef struct LINUX_ENGINE_CODE
{
    void *GameLibHandle;
    ino_t GameLibID;

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

	//Store code .dlls
	char EngineSourceCodePath[LINUX_MAX_FILE_PATH];
	char TempEngineSourceCodePath[LINUX_MAX_FILE_PATH];

	//File handle for state recording
	i32 RecordingHandle;
    i32 InputRecordingIndex;

	//File handle for state playback
    i32 PlaybackHandle;
    i32 InputPlayingIndex;
} LINUX_STATE;

//Linux file handling
//Find file name of current application
internal void linux_EXEFileNameGet(LINUX_STATE *State)
{

    ssize_t NumRead = readlink("/proc/self/exe", State->EXEPath, ArrayCount(State->EXEPath) - 1);
    if (NumRead > 0)
    {
        State->EXEFileName = State->EXEPath;

        //Scan through the full path and remove until the final "\\"
        for(char *Scan = State->EXEPath; *Scan; ++Scan)
        {
            if(*Scan == '\\')
            {
                State->EXEFileName = Scan + 1;
            }
        }
    }
}

//Get file path
internal void linux_BuildEXEPathGet(LINUX_STATE *State, const char *FileName, char *Path)
{
    polar_StringConcatenate(State->EXEFileName - State->EXEPath, State->EXEPath, polar_StringLengthGet(FileName), FileName, Path);
}


internal ino_t linux_FileIDGet(char *FileName)
{
    struct stat Attr = {};
    if (stat(FileName, &Attr))
    {
        Attr.st_ino = 0;
    }

    return Attr.st_ino;
}

internal LINUX_ENGINE_CODE linux_EngineCodeLoad(char *SourceDLLName, ino_t FileID)
{
    LINUX_ENGINE_CODE Result = {};

    Result.GameLibID = FileID;
    Result.IsDLValid = false;
    Result.GameLibHandle = dlopen(SourceDLLName, RTLD_NOW | RTLD_LOCAL);

    if(Result.GameLibHandle)
    {
        dlsym(Result.GameLibHandle, "RenderUpdate");
        Result.IsDLValid = (Result.UpdateAndRender);
    }

    if(!Result.IsDLValid)
    {
        Result.UpdateAndRender = 0;
    }

    return Result;
}



int main(int argc, char *argv[])
{
    //Error handling code passed to snd_strerror()
    i32 ALSAError;

    snd_pcm_t *Device;
    snd_pcm_sframes_t FramesWritten;

    ALSA_PROPERTIES DeviceSettings = {};
    DeviceSettings.SampleRate = 48000;
    DeviceSettings.ALSAResample = 1;
    DeviceSettings.Channels = 2;
    DeviceSettings.LatencyInMS = 32;
    // DeviceSettings.LatencyInMS = 500;


    ALSAError = snd_pcm_open(&Device, "default", SND_PCM_STREAM_PLAYBACK, 0);   
    ERR_TO_RETURN(ALSAError, "Failed to open default audio device", -1);

    ALSAError = snd_pcm_set_params(Device, SND_PCM_FORMAT_FLOAT, SND_PCM_ACCESS_RW_INTERLEAVED, DeviceSettings.Channels, DeviceSettings.SampleRate, DeviceSettings.ALSAResample, (DeviceSettings.LatencyInMS * 1000));
    ERR_TO_RETURN(ALSAError, "Failed to set default device parameters", -1);


    ALSAError = snd_pcm_get_params(Device, &DeviceSettings.BufferSize, &DeviceSettings.PeriodSize);
    ERR_TO_RETURN(ALSAError, "Failed to get default device parameters", -1);

    POLAR_BUFFER TestBuffer = {};
    TestBuffer.SampleBuffer = (f32 *) mmap(nullptr, ((sizeof *TestBuffer.SampleBuffer) * (DeviceSettings.SampleRate * DeviceSettings.Channels)), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    TestBuffer.DeviceBuffer = (f32 *) mmap(nullptr, ((sizeof *TestBuffer.SampleBuffer) * (DeviceSettings.SampleRate * DeviceSettings.Channels)), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);



    OSCILLATOR *SineOsc = entropy_wave_OscillatorCreate(DeviceSettings.SampleRate, SINE, 440);





    LINUX_STATE LinuxState = {};
    linux_EXEFileNameGet(&LinuxState);
    linux_BuildEXEPathGet(&LinuxState, "linux_polar", LinuxState.EngineSourceCodePath);
    linux_BuildEXEPathGet(&LinuxState, "polar.so", LinuxState.TempEngineSourceCodePath);



    POLAR_MEMORY EngineMemory = {};
    EngineMemory.PermanentDataSize = Megabytes(64);
    EngineMemory.TemporaryDataSize = Megabytes(32);


    LinuxState.TotalSize = EngineMemory.PermanentDataSize + EngineMemory.TemporaryDataSize;
    LinuxState.EngineMemoryBlock = mmap(nullptr, ((size_t) LinuxState.TotalSize), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);


    EngineMemory.PermanentData = LinuxState.EngineMemoryBlock;
    EngineMemory.TemporaryData = ((uint8 *) EngineMemory.PermanentData + EngineMemory.PermanentDataSize);





    if(EngineMemory.PermanentData && EngineMemory.TemporaryData)
    {
        //! Not finding the so file! Debug this
        LINUX_ENGINE_CODE PolarState = linux_EngineCodeLoad(LinuxState.EngineSourceCodePath, linux_FileIDGet(LinuxState.TempEngineSourceCodePath));
        
        
        //Extern rendering function
        if(PolarState.UpdateAndRender)
        {

            FILL(DeviceSettings.Channels, (DeviceSettings.SampleRate), TestBuffer.SampleBuffer, TestBuffer.DeviceBuffer, SineOsc);

            //TODO: Fill PolarEngine and Input structs
            // PolarState.UpdateAndRender(PolarEngine, nullptr, SineOsc, &EngineMemory, NewInput);
        }

        for(int i = 0; i < 5; i++)
        {
            //Send frames to the device
            FramesWritten = snd_pcm_writei(Device, TestBuffer.SampleBuffer, (DeviceSettings.SampleRate / 2));

            //If no frames are written then try to recover the output stream
            if(FramesWritten < 0)
            {
                FramesWritten = snd_pcm_recover(Device, FramesWritten, 0);
            }

            //If recovery fails then quit
            if(FramesWritten < 0) 
            {
                ERR_TO_RETURN(FramesWritten, "Failed to write any output frames! snd_pcm_writei()", -1);
            }

            //Wrote less frames than the total buffer length
            if(FramesWritten > 0 && FramesWritten < (DeviceSettings.SampleRate / 2))
            {
                printf("Short write (expected %i, wrote %li)\n", (DeviceSettings.SampleRate / 2), FramesWritten);
            }

            printf("Frames written:\t%ld\n", FramesWritten);
        }

    }


    munmap(TestBuffer.SampleBuffer, (sizeof *TestBuffer.SampleBuffer) * (DeviceSettings.SampleRate * DeviceSettings.Channels));
    munmap(TestBuffer.DeviceBuffer, (sizeof *TestBuffer.SampleBuffer) * (DeviceSettings.SampleRate * DeviceSettings.Channels));


    entropy_wave_OscillatorDestroy(SineOsc);
    snd_pcm_close(Device);
    
    
    return 0;
}