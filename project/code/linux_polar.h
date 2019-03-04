#ifndef linux_polar_h
#define linux_polar_h

/*                  */
/*  Linux code      */
/*                  */

//System
#include <sys/mman.h>   //Memory allocations (mmap, munmap)
#include <time.h>
#include <unistd.h>

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
    i32 ALSAError;

    snd_pcm_t *Device;
    snd_pcm_hw_params_t *HardwareParameters;
    snd_pcm_sw_params_t *SoftwareParameters;
    snd_pcm_sframes_t FramesWritten;

    u32 SampleRate;
    u8 ALSAResample;
    u16 Channels;
    u32 LatencyInMS;
    u32 Frames;

    u32 Periods;
    snd_pcm_uframes_t PeriodSizeMin;
    snd_pcm_uframes_t PeriodSizeMax;
    snd_pcm_uframes_t BufferSizeMin;
    snd_pcm_uframes_t BufferSizeMax;
} ALSA_DATA;


ALSA_DATA *linux_ALSA_Create(MEMORY_ARENA *Arena, i32 &FramesAvailable, u32 UserSampleRate, u16 UserChannels, u32 UserFrames);
void linux_ALSA_Destroy(MEMORY_ARENA *Arena, ALSA_DATA *ALSA);
void linux_ALSA_Callback(ALSA_DATA *ALSA, POLAR_ENGINE PolarEngine, POLAR_MIXER *Mixer, POLAR_RINGBUFFER *CallbackBuffer);

#endif