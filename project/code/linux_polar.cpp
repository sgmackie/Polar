//Polar
#include "polar.h"
#include "polar_file.cpp"


//Linux
#include "linux_polar.h"

//Linux globals
// global bool GlobalRunning;
// global bool GlobalPause;

//!Test variables!
global WAVEFORM Waveform = SINE;

//ALSA setup
ALSA_DATA *linux_ALSA_Create(POLAR_BUFFER &Buffer, u32 UserSampleRate, u16 UserChannels, u32 UserLatency)
{
    //Error handling code passed to snd_strerror()
    i32 ALSAError;

    ALSA_DATA *Result = (ALSA_DATA *) mmap(nullptr, (sizeof (ALSA_DATA)), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);;
    Result->SampleRate = UserSampleRate;
    Result->ALSAResample = 1;
    Result->Channels = UserChannels;
    Result->LatencyInMS = UserLatency;

    ALSAError = snd_pcm_open(&Result->Device, "default", SND_PCM_STREAM_PLAYBACK, 0);   
    ERR_TO_RETURN(ALSAError, "Failed to open default audio device", nullptr);

    ALSAError = snd_pcm_set_params(Result->Device, SND_PCM_FORMAT_FLOAT, SND_PCM_ACCESS_RW_INTERLEAVED, Result->Channels, Result->SampleRate, Result->ALSAResample, (Result->LatencyInMS * 1000));
    ERR_TO_RETURN(ALSAError, "Failed to set default device parameters", nullptr);

    ALSAError = snd_pcm_get_params(Result->Device, &Result->BufferSize, &Result->PeriodSize);
    ERR_TO_RETURN(ALSAError, "Failed to get default device parameters", nullptr);

    Buffer.SampleBuffer = (f32 *) mmap(nullptr, ((sizeof *Buffer.SampleBuffer) * (Result->SampleRate * Result->Channels)), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    Buffer.DeviceBuffer = (f32 *) mmap(nullptr, ((sizeof *Buffer.SampleBuffer) * (Result->SampleRate * Result->Channels)), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    Buffer.FramesAvailable = ((Result->BufferSize + Result->PeriodSize) * Result->Channels);

    return Result;
}

//ALSA destroy
void linux_ALSA_Destroy(ALSA_DATA *Result, POLAR_BUFFER &Buffer)
{
    munmap(Buffer.SampleBuffer, (sizeof *Buffer.SampleBuffer) * (Result->SampleRate * Result->Channels));
    munmap(Buffer.DeviceBuffer, (sizeof *Buffer.SampleBuffer) * (Result->SampleRate * Result->Channels));
    snd_pcm_close(Result->Device);
    munmap(Result, (sizeof (Result)));
}


//Linux file handling
//Find file name of current application
internal void linux_EXEFileNameGet(LINUX_STATE *State)
{
    //Read value of a symbolic link and record size
    ssize_t PathSize = readlink("/proc/self/exe", State->EXEPath, ArrayCount(State->EXEPath) - 1);
    if(PathSize > 0)
    {
        State->EXEFileName = State->EXEPath;

        //Scan through the full path and record
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

//Record file attributes using stat ("http://pubs.opengroup.org/onlinepubs/000095399/basedefs/sys/stat.h.html") 
internal ino_t linux_FileIDGet(char *FileName)
{
    struct stat FileAttributes = {};

    if(stat(FileName, &FileAttributes))
    {
        FileAttributes.st_ino = 0;
    }

    return FileAttributes.st_ino;
}

//Wrap dlopen with error handling
internal void *linux_LibraryOpen(const char *Library)
{
    void *Handle = nullptr;

    Handle = dlopen(Library, RTLD_NOW | RTLD_LOCAL);
    
    //Record error using dlerror
    if(!Handle)
    {
        printf("Linux: dlopen failed!\t%s\n", dlerror());
    }

    return Handle;
}

//Wrap dlclose
internal void linux_LibraryClose(void *Handle)
{
    if(Handle != nullptr)
    {
        dlclose(Handle);
        Handle = nullptr;
    }
}

//Wrap dlsym with error handling
internal void *linux_ExternalFunctionLoad(void *Library, const char *Name)
{
    void *FunctionSymbol = dlsym(Library, Name);

    if(!FunctionSymbol)
    {
        printf("Linux: dlsym failed!\t%s\n", dlerror());
    }

    return FunctionSymbol;
}

//Check if file ID's match and load engine code if not
internal bool linux_EngineCodeLoad(LINUX_ENGINE_CODE *EngineCode, char *DLName, ino_t FileID)
{
    if(EngineCode->EngineID != FileID)
    {
        linux_LibraryClose(EngineCode->EngineHandle);
        EngineCode->EngineID = FileID;
        EngineCode->IsDLValid = false;

        //TODO: Can't actually pass DLName here because Linux want's "./" prefixed, create function to prefix strings
        EngineCode->EngineHandle = linux_LibraryOpen("./polar.so");
        if (EngineCode->EngineHandle)
        {
            *(void **)(&EngineCode->UpdateAndRender) = linux_ExternalFunctionLoad(EngineCode->EngineHandle, "RenderUpdate");

            EngineCode->IsDLValid = (EngineCode->UpdateAndRender);
        }
    }

    if(!EngineCode->IsDLValid)
    {
        linux_LibraryClose(EngineCode->EngineHandle);
        EngineCode->EngineID = 0;
        EngineCode->UpdateAndRender = 0;
    }

    return EngineCode->IsDLValid;
}

//Unload engine code
internal void linux_EngineCodeUnload(LINUX_ENGINE_CODE *EngineCode)
{
    linux_LibraryClose(EngineCode->EngineHandle);
    EngineCode->EngineID = 0;
    EngineCode->IsDLValid = false;
    EngineCode->UpdateAndRender = 0;
}


int main(int argc, char *argv[])
{
    POLAR_DATA PolarEngine = {};


    ALSA_DATA *ALSA =           linux_ALSA_Create(PolarEngine.Buffer, 48000, 2, 32);
    PolarEngine.BufferFrames =  PolarEngine.Buffer.FramesAvailable;
    PolarEngine.Channels =      ALSA->Channels;
    PolarEngine.SampleRate =    ALSA->SampleRate;
    //TODO: Convert flags like SND_PCM_FORMAT_FLOAT to numbers
    PolarEngine.BitRate =       32;


    POLAR_WAV *OutputRenderFile = polar_render_WAVWriteCreate("Polar_Output.wav", &PolarEngine);
    OSCILLATOR *SineOsc = entropy_wave_OscillatorCreate(PolarEngine.SampleRate, Waveform, 440);



    LINUX_STATE LinuxState = {};
    linux_EXEFileNameGet(&LinuxState);
    linux_BuildEXEPathGet(&LinuxState, "polar.so", LinuxState.EngineSourceCodePath);


    POLAR_MEMORY EngineMemory = {};
    EngineMemory.PermanentDataSize = Megabytes(64);
    EngineMemory.TemporaryDataSize = Megabytes(32);


    LinuxState.TotalSize = EngineMemory.PermanentDataSize + EngineMemory.TemporaryDataSize;
    LinuxState.EngineMemoryBlock = mmap(nullptr, ((size_t) LinuxState.TotalSize), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);


    EngineMemory.PermanentData = LinuxState.EngineMemoryBlock;
    EngineMemory.TemporaryData = ((uint8 *) EngineMemory.PermanentData + EngineMemory.PermanentDataSize);



    if(EngineMemory.PermanentData && EngineMemory.TemporaryData)
    {
        
        LINUX_ENGINE_CODE PolarState = {};
        linux_EngineCodeLoad(&PolarState, LinuxState.EngineSourceCodePath, linux_FileIDGet(LinuxState.EngineSourceCodePath));

        POLAR_INPUT Input[2] = {};
        POLAR_INPUT *NewInput = &Input[0];
        POLAR_INPUT *OldInput = &Input[1];
    

        for(int i = 0; i < 5; i++)
        {
            //Extern rendering function
            if(PolarState.UpdateAndRender)
            {
                //Update objects and fill the buffer
                if(OutputRenderFile != nullptr)
                {
                    PolarState.UpdateAndRender(PolarEngine, OutputRenderFile, SineOsc, &EngineMemory, NewInput);
                    OutputRenderFile->TotalSampleCount += polar_render_WAVWriteFloat(OutputRenderFile, (PolarEngine.Buffer.FramesAvailable * PolarEngine.Channels), OutputRenderFile->Data);
                }

                else
                {
                    PolarState.UpdateAndRender(PolarEngine, nullptr, SineOsc, &EngineMemory, NewInput);
                }

                ALSA->FramesWritten = snd_pcm_writei(ALSA->Device, PolarEngine.Buffer.SampleBuffer, (PolarEngine.BufferFrames));

                //If no frames are written then try to recover the output stream
                if(ALSA->FramesWritten < 0)
                {
                    ALSA->FramesWritten = snd_pcm_recover(ALSA->Device, ALSA->FramesWritten, 0);
                }

                //If recovery fails then quit
                if(ALSA->FramesWritten < 0) 
                {
                    ERR_TO_RETURN(ALSA->FramesWritten, "Failed to write any output frames! snd_pcm_writei()", -1);
                }

                //Wrote less frames than the total buffer length
                if(ALSA->FramesWritten > 0 && ALSA->FramesWritten < (PolarEngine.BufferFrames))
                {
                    printf("Short write (expected %i, wrote %li)\n", (PolarEngine.BufferFrames), ALSA->FramesWritten);
                }
            }        
        }

    

        //Reset input for next loop
        POLAR_INPUT *Temp = NewInput;
        NewInput = OldInput;
        OldInput = Temp;

    }

    printf("Frames written:\t%ld\n", ALSA->FramesWritten);
    printf("Polar: %lu frames written to %s\n", OutputRenderFile->TotalSampleCount, OutputRenderFile->Path);

    polar_render_WAVWriteDestroy(OutputRenderFile);

    entropy_wave_OscillatorDestroy(SineOsc);
    linux_ALSA_Destroy(ALSA, PolarEngine.Buffer);    
    
    return 0;
}