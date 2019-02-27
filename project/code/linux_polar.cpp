//Perfomance defines (will change stack allocation sizes for things like max sources per container)
#define MAX_STRING_LENGTH 64
#define MAX_CHANNELS 4
#define MAX_CONTAINERS 4
#define MAX_SOURCES 128
#define MAX_BREAKPOINTS 64
#define MAX_ENVELOPES 4
#define DEFAULT_AMPLITUDE 0.8
#define DEFAULT_SAMPLERATE 48000

#include "polar.h"
#include "linux_polar.h"
#include "../external/external_code.h"

global char AssetPath[MAX_STRING_LENGTH] = {"../../data/"};

#include "polar.cpp"

//ALSA setup
ALSA_DATA *linux_ALSA_Create(MEMORY_ARENA *Arena, i32 &FramesAvailable, u32 UserSampleRate, u16 UserChannels, u32 UserFrames)
{
    ALSA_DATA *Result = 0;
    Result = (ALSA_DATA *) memory_arena_Push(Arena, Result, (sizeof (ALSA_DATA)));
    Result->SampleRate = UserSampleRate;
    Result->ALSAResample = 1;
    Result->Channels = UserChannels;
    Result->Frames = UserFrames;

    //Error handling code passed to snd_strerror()
    Result->ALSAError = 0;

    Result->ALSAError = snd_pcm_open(&Result->Device, "default", SND_PCM_STREAM_PLAYBACK, SND_PCM_NONBLOCK | SND_PCM_ASYNC);   
    ERR_TO_RETURN(Result->ALSAError, "Failed to open default audio device", nullptr);

    Result->HardwareParameters = (snd_pcm_hw_params_t *) memory_arena_Push(Arena, Result->HardwareParameters, (sizeof (Result->HardwareParameters)));
    Result->SoftwareParameters = (snd_pcm_sw_params_t *) memory_arena_Push(Arena, Result->SoftwareParameters, (sizeof (Result->SoftwareParameters)));

    Result->ALSAError = snd_pcm_hw_params_any(Result->Device, Result->HardwareParameters);
    ERR_TO_RETURN(Result->ALSAError, "Failed to initialise hardware parameters", nullptr);

    Result->ALSAError = snd_pcm_hw_params_set_access(Result->Device, Result->HardwareParameters, SND_PCM_ACCESS_RW_INTERLEAVED);
    ERR_TO_RETURN(Result->ALSAError, "Failed to set PCM read and write access", nullptr);

    Result->ALSAError = snd_pcm_hw_params_set_format(Result->Device, Result->HardwareParameters, SND_PCM_FORMAT_FLOAT);
    ERR_TO_RETURN(Result->ALSAError, "Failed to set PCM output format", nullptr);

    Result->ALSAError = snd_pcm_hw_params_set_rate(Result->Device, Result->HardwareParameters, Result->SampleRate, 0);
    ERR_TO_RETURN(Result->ALSAError, "Failed to set sample rate", nullptr);

    Result->ALSAError = snd_pcm_hw_params_set_rate_resample(Result->Device, Result->HardwareParameters, Result->ALSAResample);
    ERR_TO_RETURN(Result->ALSAError, "Failed to set resampling", nullptr);

    Result->ALSAError = snd_pcm_hw_params_set_channels(Result->Device, Result->HardwareParameters, Result->Channels);
    ERR_TO_RETURN(Result->ALSAError, "Failed to set channels", nullptr);

    Result->ALSAError = snd_pcm_hw_params(Result->Device, Result->HardwareParameters);
    ERR_TO_RETURN(Result->ALSAError, "Failed to set period", nullptr);

    Result->ALSAError = snd_pcm_sw_params_current(Result->Device, Result->SoftwareParameters);
    ERR_TO_RETURN(Result->ALSAError, "Failed to get current software parameters", nullptr);

    Result->ALSAError = snd_pcm_sw_params_set_avail_min(Result->Device, Result->SoftwareParameters, Result->Frames);
    ERR_TO_RETURN(Result->ALSAError, "Failed to set software available frames", nullptr);

    Result->ALSAError = snd_pcm_sw_params_set_start_threshold(Result->Device, Result->SoftwareParameters, 0);
    ERR_TO_RETURN(Result->ALSAError, "Failed to set software available frames", nullptr);
    
    Result->ALSAError = snd_pcm_sw_params(Result->Device, Result->SoftwareParameters);
    ERR_TO_RETURN(Result->ALSAError, "Failed to set software parameters", nullptr);
    
    Result->ALSAError = snd_pcm_prepare(Result->Device);
    ERR_TO_RETURN(Result->ALSAError, "Failed to start PCM device", nullptr);

    Result->ALSAError = snd_pcm_hw_params_get_period_size_max(Result->HardwareParameters, &Result->PeriodSizeMin, 0);
    ERR_TO_RETURN(Result->ALSAError, "Failed to get minimum period size", nullptr);

    Result->ALSAError = snd_pcm_hw_params_get_period_size_max(Result->HardwareParameters, &Result->PeriodSizeMax, 0);
    ERR_TO_RETURN(Result->ALSAError, "Failed to get maximum period size", nullptr);

    Result->ALSAError = snd_pcm_hw_params_get_buffer_size_min(Result->HardwareParameters, &Result->BufferSizeMin);
    ERR_TO_RETURN(Result->ALSAError, "Failed to get minimum buffer size", nullptr);

    Result->ALSAError = snd_pcm_hw_params_get_buffer_size_max(Result->HardwareParameters, &Result->BufferSizeMax);
    ERR_TO_RETURN(Result->ALSAError, "Failed to get maximum buffer size", nullptr);

    Result->ALSAError = snd_pcm_hw_params_get_periods(Result->HardwareParameters, &Result->Periods, 0);
    ERR_TO_RETURN(Result->ALSAError, "Failed to get period count", nullptr);

    FramesAvailable = (Result->PeriodSizeMin * Result->Channels);

    return Result;
}


//ALSA destroy
void linux_ALSA_Destroy(MEMORY_ARENA *Arena, ALSA_DATA *ALSA)
{
    snd_pcm_close(ALSA->Device);
    memory_arena_Reset(Arena);
    memory_arena_Pull(Arena);
}

void linux_ALSA_Callback(ALSA_DATA *ALSA, POLAR_ENGINE PolarEngine, POLAR_MIXER *Mixer, POLAR_RINGBUFFER *CallbackBuffer)
{
    snd_pcm_wait(ALSA->Device, -1);

    if(polar_ringbuffer_WriteCheck(CallbackBuffer))
    {
        Callback(PolarEngine, Mixer, polar_ringbuffer_WriteData(CallbackBuffer));
        polar_ringbuffer_WriteFinish(CallbackBuffer);
    }

    if(polar_ringbuffer_ReadCheck(CallbackBuffer))
    {
        ALSA->FramesWritten = snd_pcm_writei(ALSA->Device, polar_ringbuffer_ReadData(CallbackBuffer), (PolarEngine.BufferFrames / PolarEngine.Channels));
        if(ALSA->FramesWritten < 0)
        {
            ALSA->FramesWritten = snd_pcm_recover(ALSA->Device, ALSA->FramesWritten, 0);
        }
        if(ALSA->FramesWritten < 0) 
        {
            ERR_TO_RETURN(ALSA->FramesWritten, "ALSA: Failed to write any output frames! snd_pcm_writei()", NONE);
        }
        if(ALSA->FramesWritten > 0 && ALSA->FramesWritten < (PolarEngine.BufferFrames / PolarEngine.Channels))
        {
            printf("ALSA: Short write!\tExpected %i, wrote %li\n", (PolarEngine.BufferFrames / PolarEngine.Channels), ALSA->FramesWritten);
        }

        polar_ringbuffer_ReadFinish(CallbackBuffer);
    }
}

int main(int argc, char *argv[])
{
    //Allocate memory
    MEMORY_ARENA *EngineArena = memory_arena_Create(Kilobytes(500));
    MEMORY_ARENA *SourceArena = memory_arena_Create(Megabytes(100));

    POLAR_ENGINE Engine = {};
    i32 FramesAvailable = 0;
    ALSA_DATA *ALSA =      linux_ALSA_Create(EngineArena, FramesAvailable, 48000, 2, 32);
    Engine.BufferFrames =  FramesAvailable;
    Engine.Channels =      ALSA->Channels;
    Engine.SampleRate =    ALSA->SampleRate;

    //Define the callback and channel summing functions
    Callback = &polar_render_Callback;
    switch(Engine.Channels)
    {
        case 2:
        {
            Summing = &polar_render_SumStereo;
            break;
        }
        default:
        {
            Summing = &polar_render_SumStereo;
            break;
        }
    }

    //PCG Random Setup
    i32 Rounds = 5;
    pcg32_srandom(time(NULL) ^ (intptr_t) &printf, (intptr_t) &Rounds);

    //Create ringbuffer with a specified block count (default is 3)
    POLAR_RINGBUFFER *CallbackBuffer = polar_ringbuffer_Create(EngineArena, Engine.BufferFrames, 3);

    //Create mixer object that holds all submixes and their containers
    POLAR_MIXER *MasterOutput = polar_mixer_Create(SourceArena, -1);
    
    //Sine sources
    polar_mixer_SubmixCreate(SourceArena, MasterOutput, 0, "SM_SineChordMix", -1);
    polar_mixer_ContainerCreate(MasterOutput, "SM_SineChordMix", "CO_ChordContainer", -1);
    polar_source_CreateFromFile(SourceArena, MasterOutput, Engine, "asset_lists/Source_Import.txt");

    //File sources
    polar_mixer_SubmixCreate(SourceArena, MasterOutput, 0, "SM_FileMix", -1);
    polar_mixer_ContainerCreate(MasterOutput, "SM_FileMix", "CO_FileContainer", -1);
    polar_source_Create(SourceArena, MasterOutput, Engine, "CO_FileContainer", "SO_WPN_Phasor", Stereo, SO_FILE, "audio/wpn_phasor.wav");
    polar_source_Create(SourceArena, MasterOutput, Engine, "CO_FileContainer", "SO_AMB_Forest_01", Stereo, SO_FILE, "audio/amb_river.wav");
    polar_source_Create(SourceArena, MasterOutput, Engine, "CO_FileContainer", "SO_Whiterun", Stereo, SO_FILE, "audio/Whiterun48.wav");

    //Silent first loop
    printf("Polar: Pre-roll silence\n");
    MasterOutput->Amplitude = DB(-99);
    for(u32 i = 0; i < 60; ++i)
    {
        linux_ALSA_Callback(ALSA, Engine, MasterOutput, CallbackBuffer);
    }

    printf("Polar: Playback\n");
    MasterOutput->Amplitude = DB(-6);
    for(u32 i = 0; i < 2000; ++i)
    {
        polar_source_UpdatePlaying(MasterOutput);

        if(i == 100)
        {
            f32 StackPositions[MAX_CHANNELS] = {0.0};
            // polar_source_Play(MasterOutput, "SO_SineChord_Segment_A", 6, StackPositions, FX_DRY, EN_BREAKPOINT, "../../data/breakpoints/breaks.txt");
            // polar_source_Play(MasterOutput, "SO_SineChord_Segment_C", 6, StackPositions, FX_DRY, EN_BREAKPOINT, "breaks.txt");
            // polar_source_Play(MasterOutput, "SO_SineChord_Segment_D", 6, StackPositions, FX_DRY, EN_BREAKPOINT, "breaks.txt");

            polar_source_Play(MasterOutput, "SO_SineChord_Segment_B", 9, StackPositions, FX_DRY, EN_BREAKPOINT, "breakpoints/breaks2.txt");

            polar_source_Play(MasterOutput, "SO_Whiterun", 8, StackPositions, FX_DRY, EN_ADSR);
        }

        linux_ALSA_Callback(ALSA, Engine, MasterOutput, CallbackBuffer);
        // printf("ALSA: Frames written:\t%ld\n", ALSA->FramesWritten);
    }

    polar_mixer_Destroy(EngineArena, MasterOutput);

    polar_ringbuffer_Destroy(EngineArena, CallbackBuffer);
    linux_ALSA_Destroy(EngineArena, ALSA);

    memory_arena_Destroy(EngineArena);
    memory_arena_Destroy(SourceArena);

    return 0;
}