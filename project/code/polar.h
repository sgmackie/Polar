//Headers
#include "core/core.h"
#include "core/log.cpp"

//Define name of logger here for macro functions
static LOGGER Logger = {};
bool core_CreateLogger(char const *Path, int Level, bool IsQuiet)
{
    FILE *LogFile = 0;
    fopen_s(&LogFile, Path, "w+");

    if(!LogFile)
    {
        printf("Failed to open file: %s\n", Path);
        return false;
    }

    Logger.Init(LogFile, Level, IsQuiet);
    return true;
}

void core_DestroyLogger()
{
    fclose(Logger.File);
}

//Assertion with logging
static bool AssertQuit = true;
static i64 AssertFailuresCounter = 0;

//Assertion function with optional logging
//If AssertQuit is set, exists program
#define Assert(Expression, ...) do {                            \
  if(!(Expression))                                             \
  {                                                             \
    ++AssertFailuresCounter;                                    \
    Logger.Log(LOG_FATAL,    __FILE__, __LINE__, __VA_ARGS__);  \
    if(AssertQuit)                                              \
    {                                                           \
        abort();                                                \
    }                                                           \
  }                                                             \
} while(0);                                                     \

//CUDA
#if CUDA
#include "polar_cuda.h"
#include "cuda/polar_cuda_error.h"
#endif

//External functions
#include "../external/xxhash.c"     //Fast hasing function
#define FastHash(Name) XXH64(Name, strlen(Name), 0)
#include "../external/csv_read.c"
#include "../external/split.c"
#include "../external/pcg_basic.c"  //Random number generator
#define DR_WAV_IMPLEMENTATION
#include "../external/dr_wav.h"

//IMGUI
#include "../external/imgui/imgui.cpp"
#include "../external/imgui/imgui_widgets.cpp"
#include "../external/imgui/imgui_draw.cpp"
#include "../external/imgui/imgui_demo.cpp"

//Core functions
#include "core/functions.cpp"       //General functions (alignment checks)
#include "core/memory/arena.cpp"    //Arena creation (alloc, resize, freeall)
#include "core/memory/pool.cpp"     //Pool creation (alloc, free, freeall)
#include "core/math.cpp"            //Math functions
// #include "core/string.cpp"          //String handling

//Max sizes
#define MAX_SOURCES         256
#define MAX_STRING_LENGTH   128
#define MAX_BUFFER_SIZE     8192
#define MAX_BREAKPOINTS     64


typedef enum STATE_PLAY
{
    STOPPED,
    STOPPING,
    PLAYING,
    PAUSED,
} STATE_PLAY;

typedef struct CMP_DURATION
{
    //Data    
    STATE_PLAY  States;
    u64         SampleCount;
} CMP_DURATION;

typedef struct CMP_FORMAT
{
    //Data    
    u32 SampleRate;
    u32 Channels;

    void Init(u32 InputRate, u32 InputChannels)
    {
        SampleRate  = 0;
        Channels    = 0;

        if(InputRate)
        {
            SampleRate = InputRate;
        }
        if(InputChannels)
        {
            Channels = InputChannels;
        }
    }

} CMP_FORMAT;

typedef struct CMP_POSITION
{
    //Data    
    f32 X;
    f32 Y;
    f32 Z;
    f32 W;
} CMP_POSITION;


typedef struct CMP_WAV
{
    f32     *Data;
    u8      BitRate;
    u32     SampleRate;
    u32     Channels;
    u64     Length;
    u64     ReadIndex;

    void Init(char const *Name)
    {
        Data = 0;
        BitRate = 0;
        SampleRate = 0;
        Channels = 0;
        Length = 0;
        ReadIndex = 0;
        Data = drwav_open_file_and_read_pcm_frames_f32(Name, &Channels, &SampleRate, &Length);  
    }

} CMP_WAV;




typedef struct CMP_BUFFER
{
    //Data
    size_t  Count;
    f32     *Data;

    void CreateFromArena(MEMORY_ARENA *Arena, size_t Type, size_t InputCount)
    {
        Count   = InputCount;
        Data    = (f32 *) Arena->Alloc((Type * Count), MEMORY_ARENA_ALIGNMENT);
    }

    void CreateFromPool(MEMORY_POOL *Pool, size_t InputCount)
    {
        Count   = InputCount;
        Data    = (f32 *) Pool->Alloc();
    }

    void Destroy()
    {
        Count = 0;
        Data = 0;
    }

    f32 *Write()
    {
        return Data;
    }

    f32 *Read()
    {
        return Data;
    }    

} CMP_BUFFER;


#define RINGBUFFER_DEFAULT_BLOCK_COUNT 3
//TODO: Create generic void * buffers to support any type
typedef struct CMP_RINGBUFFER
{
    //Data
    size_t  Count;
    size_t  ReadAddress;
    size_t  WriteAddress;
    size_t  TotalBlocks;
    size_t  CurrentBlocks;
    i16     *Data;

    //Functions
    void Create(MEMORY_ARENA *Arena, size_t Type, size_t InputCount, size_t Blocks)
    {
        if(Blocks <= 0)
        {
            Blocks = RINGBUFFER_DEFAULT_BLOCK_COUNT;
            Info("Ringbuffer: Using default block count: %d", RINGBUFFER_DEFAULT_BLOCK_COUNT);
        }

        Count           = InputCount;
        TotalBlocks     = Blocks;    
        CurrentBlocks   = 0;
        ReadAddress     = 0;
        WriteAddress    = 0;

        Data = (i16 *)  Arena->Alloc(((Type * Count) * TotalBlocks), MEMORY_ARENA_ALIGNMENT);
    }

    void Destroy()
    {
        Count           = 0;
        TotalBlocks     = 0;    
        CurrentBlocks   = 0;
        ReadAddress     = 0;
        WriteAddress    = 0;
        Data            = 0;
    }

    i16 *Write()
    {
        return Data + (WriteAddress * Count);
    }

    bool CanWrite()
    {
        return CurrentBlocks != TotalBlocks;
    }

    void FinishWrite()
    {
        WriteAddress = ((WriteAddress + 1) % TotalBlocks);
        CurrentBlocks += 1;
    }

    i16 *Read()
    {
        return Data + (ReadAddress * Count);
    }

    bool CanRead()
    {
        return CurrentBlocks != 0;
    }

    void FinishRead()
    {
        ReadAddress = ((ReadAddress + 1) % TotalBlocks);
        CurrentBlocks -= 1;        
    }    

} CMP_RINGBUFFER;

typedef struct CMP_FADE
{
    //Data
    f64     Current;
    f64     Previous;
    f64     StartValue;
    f64     EndValue;
    f64     StartTime;
    f64     Duration;
    bool    IsFading;

    //Functions
    void Init(double Amplitude);

} CMP_FADE;


typedef struct CMP_OSCILLATOR
{
    //Flags
    typedef enum TYPE
    {
        NOISE   = 1 << 0,
        SQUARE  = 1 << 1,
        SINE    = 1 << 2,
    } TYPE;

    //Data
    i32         Flag;
    f64         Phasor;
    f64         PhaseIncrement;
    f64         SizeOverSampleRate;
    f64         Frequency;

    //Functions
    void Init(i32 Type, u32 SampleRate, f64 InputFrequency, f64 Limit = TWO_PI32)
    {
        Flag |= Type;
        Phasor = 0;
        PhaseIncrement = 0;
        SizeOverSampleRate = Limit / SampleRate;
        Frequency = InputFrequency;
    }


} CMP_OSCILLATOR;



typedef struct CMP_ADSR
{
    bool IsActive;
    f64 MaxAmplitude;
    f64 Attack;
    f64 Decay;
    f64 SustainAmplitude;
    f64 Release;
    u64 Index;
    u64 DurationInSamples;
} CMP_ADSR;

typedef struct CMP_BREAKPOINT_POINT
{
    f64 Value;
    f64 Time;
} CMP_BREAKPOINT_POINT;

typedef struct CMP_BREAKPOINT
{
    u64 Index;
    size_t Count;
    CMP_BREAKPOINT_POINT *Points;

    void Init(size_t PointIndex, f64 InputValue, f64 InputTime)
    {
        Index   = 0;

        if(PointIndex)
        {
            Points[PointIndex].Value = InputValue;
            Points[PointIndex].Time = InputTime;
        }
    }

    void CreateFromArena(MEMORY_ARENA *Arena, size_t Type, size_t InputCount)
    {
        Count   = 0;
        Points  = (CMP_BREAKPOINT_POINT *) Arena->Alloc((Type * InputCount), MEMORY_ARENA_ALIGNMENT);
    }

    void CreateFromPool(MEMORY_POOL *Pool, size_t InputCount)
    {
        Count   = 0;
        Points  = (CMP_BREAKPOINT_POINT *) Pool->Alloc();
    }

    void Destroy()
    {
        Index = 0;
        Count = 0;
        Points = 0;
    }

} CMP_BREAKPOINT;


typedef struct CMP_MODULATOR
{
    //Flags
    typedef enum ASSIGNMENT
    {
        AMPLITUDE       = 1 << 0,
        FREQUENCY       = 1 << 1,
    } ASSIGNMENT;

    typedef enum TYPE
    {
        LFO_OSCILLATOR  = 1 << 2,
        ENV_ADSR        = 1 << 3,
        ENV_BREAKPOINT  = 1 << 4,
    } TYPE;

    i32 Flag;
    union
    {
        CMP_OSCILLATOR  Oscillator;
        CMP_ADSR        ADSR;
        CMP_BREAKPOINT  Breakpoint;
    };

    void Init(i32 Type, i32 Assignment)
    {
        Flag |= Type;
        Flag |= Assignment;
    }

} CMP_MODULATOR;


typedef struct POLAR_ENGINE
{
    //Data
    f32                         UpdateRate;
    f64                         NoiseFloor;
    size_t                      BytesPerSample;
    u32                         BufferFrames;
    u32                         LatencyFrames;
    CMP_FORMAT     Format;
    CMP_RINGBUFFER        CallbackBuffer;
    CMP_BUFFER            MixBuffer;

} POLAR_ENGINE;

typedef struct TPL_PLAYBACK
{
    //Data
    CMP_FORMAT      Format;
    CMP_BUFFER      Buffer;
    CMP_DURATION    Duration;
} TPL_PLAYBACK;


typedef u64 ID_SOURCE;
typedef struct ENTITY_SOURCES
{
    //Flags
    enum FLAG_COMPONENTS
    {
        PLAYBACK                = 1 << 0,
        AMPLITUDE               = 1 << 1,
        POSITION                = 1 << 2,
        OSCILLATOR              = 1 << 3,
        WAV                     = 1 << 4,
        ADSR                    = 1 << 5,
        BREAKPOINT              = 1 << 6,
        MODULATOR               = 1 << 7,
    };


    //Data
    size_t                      Count;
    char                        **Names;
    ID_SOURCE                   *IDs;
    TPL_PLAYBACK                *Playbacks;
    CMP_FADE                    *Amplitudes;
    CMP_POSITION                *Positions;
    CMP_OSCILLATOR              *Oscillators;
    CMP_WAV                     *WAVs;
    CMP_ADSR                    *ADSRs;
    CMP_BREAKPOINT              *Breakpoints;
    CMP_MODULATOR               *Modulators;
    i32                         *Flags;

    //Functions
    void Create                 (MEMORY_ARENA *Arena, size_t Size);
    void Destroy                (MEMORY_ARENA *Arena);
    void Init                   (size_t Index);
    ID_SOURCE AddByName         (MEMORY_POOL *Pool, char *Name);
    ID_SOURCE AddByHash         (u64 Hash);
    bool Remove                 (MEMORY_POOL *Pool, ID_SOURCE ID);
    size_t RetrieveIndex        (ID_SOURCE ID);
    ID_SOURCE RetrieveID        (size_t Index);

} ENTITY_SOURCES;



typedef struct TUPLE_MIX
{
    //Data
    CMP_FORMAT      Format;
    CMP_BUFFER      Buffer;
    CMP_FADE        Amplitude;
} TUPLE_MIX;






typedef struct SYS_PLAY
{
    //Data
    size_t          SystemCount;    
    ID_SOURCE       *SystemSources;
    
    //Functions
    void Create     (MEMORY_ARENA *Arena, size_t Size);
    void Destroy    (MEMORY_ARENA *Arena);
    void Add        (ID_SOURCE ID);
    bool Remove     (ID_SOURCE ID);
    bool Start      (ENTITY_SOURCES *Sources, ID_SOURCE ID, f64 InputDuration, bool IsAligned = true);
    bool Pause      (ENTITY_SOURCES *Sources, ID_SOURCE ID);
    bool Resume     (ENTITY_SOURCES *Sources, ID_SOURCE ID);
    void Update     (ENTITY_SOURCES *Sources, f64 Time, u32 PreviousSamplesWritten, u32 SamplesToWrite);

} SYS_PLAY;


typedef struct SYS_FADE
{
    //Data
    size_t          SystemCount;    
    ID_SOURCE       *SystemSources;
    
    //Functions
    void Create     (MEMORY_ARENA *Arena, size_t Size);
    void Destroy    (MEMORY_ARENA *Arena);
    void Add        (ID_SOURCE ID);
    bool Remove     (ID_SOURCE ID);
    bool Start      (ENTITY_SOURCES *Sources, ID_SOURCE ID, f64 Time, f64 Amplitude, f64 Duration);
    void Update     (ENTITY_SOURCES *Sources, f64 Time);

} SYS_FADE;



typedef struct SYS_OSCILLATOR_NOISE
{    
    //Data
    size_t                  SystemCount;
    ID_SOURCE               *SystemSources;
    
    //Functions
    void Create             (MEMORY_ARENA *Arena, size_t Size);
    void Destroy            (MEMORY_ARENA *Arena);
    void Add                (ID_SOURCE ID);
    bool Remove             (ID_SOURCE ID);
    void RenderToBuffer     (CMP_OSCILLATOR &Oscillator, CMP_BUFFER &Buffer, size_t BufferCount);
    void Update             (ENTITY_SOURCES *Sources, size_t BufferCount);

} SYS_OSCILLATOR_NOISE;

typedef struct SYS_OSCILLATOR_SINE
{    
    //Data
    size_t                  SystemCount;
    ID_SOURCE               *SystemSources;
    
    //Functions
    void Create             (MEMORY_ARENA *Arena, size_t Size);
    void Destroy            (MEMORY_ARENA *Arena);
    void Add                (ID_SOURCE ID);
    bool Remove             (ID_SOURCE ID);
    void RenderToBufferWithModulation(CMP_OSCILLATOR &Oscillator, CMP_MODULATOR &Modulator, CMP_BUFFER &Buffer, size_t BufferCount);
    void RenderToBuffer     (CMP_OSCILLATOR &Oscillator, CMP_BUFFER &Buffer, size_t BufferCount);
    void Update             (ENTITY_SOURCES *Sources, size_t BufferCount);

} SYS_OSCILLATOR_SINE;

typedef struct SYS_OSCILLATOR_SQUARE
{    
    //Data
    size_t                  SystemCount;
    ID_SOURCE               *SystemSources;
    
    //Functions
    void Create             (MEMORY_ARENA *Arena, size_t Size);
    void Destroy            (MEMORY_ARENA *Arena);
    void Add                (ID_SOURCE ID);
    bool Remove             (ID_SOURCE ID);
    void RenderToBuffer     (CMP_OSCILLATOR &Oscillator, CMP_BUFFER &Buffer, size_t BufferCount);
    void Update             (ENTITY_SOURCES *Sources, size_t BufferCount);

} SYS_OSCILLATOR_SQUARE;


//Module - collection of systems
typedef struct MDL_OSCILLATOR
{
    SYS_OSCILLATOR_NOISE    Noise;
    SYS_OSCILLATOR_SINE     Sine;
    SYS_OSCILLATOR_SQUARE   Square;
} MDL_OSCILLATOR;


typedef struct SYS_WAV
{
    //Data
    size_t                  SystemCount;    
    ID_SOURCE               *SystemSources;
    
    //Functions
    void Create             (MEMORY_ARENA *Arena, size_t Size);
    void Destroy            (MEMORY_ARENA *Arena);
    void Add                (ID_SOURCE ID);
    bool Remove             (ID_SOURCE ID);
    void RenderToBuffer     (CMP_WAV &WAV, CMP_BUFFER &Buffer, i32 Rate, size_t BufferCount);
    void Update             (ENTITY_SOURCES *Sources, f64 Pitch, size_t BufferCount);

} SYS_WAV;

typedef struct SYS_MIX
{
    //Data
    size_t                  SystemCount;    
    ID_SOURCE               *SystemSources;

    //Functions
    void Create             (MEMORY_ARENA *Arena, size_t Size);
    void Destroy            (MEMORY_ARENA *Arena);
    void Add                (ID_SOURCE ID);
    bool Remove             (ID_SOURCE ID);
    void RenderToBuffer     (f32 *MixBuffer, size_t SamplesToWrite, u32 Channels, CMP_BUFFER &SourceBuffer, CMP_FADE &SourceAmplitude, f64 TargetAmplitude); 
    void Update             (ENTITY_SOURCES *Sources, f32 *MixBuffer, size_t SamplesToWrite); 
} SYS_MIX;


typedef struct SYS_ENVELOPE_ADSR
{
    //Data
    size_t                  SystemCount;    
    ID_SOURCE               *SystemSources;
    
    //Functions
    void Create(MEMORY_ARENA *Arena, size_t Size)
    {
        SystemSources = (ID_SOURCE *) Arena->Alloc((sizeof(ID_SOURCE) * Size), MEMORY_ARENA_ALIGNMENT);
        SystemCount = 0;
    }

    void Destroy(MEMORY_ARENA *Arena)
    {
        Arena->FreeAll();
    }

    void Add(ID_SOURCE ID)
    {
        SystemSources[SystemCount] = ID;
        ++SystemCount;
    }

    bool Remove(ID_SOURCE ID)
    {
        for(size_t i = 0; i <= SystemCount; ++i)
        {
            if(SystemSources[i] == ID)
            {
                SystemSources[i] = 0;
                --SystemCount;
                return true;
            }
        }
        //!Log
        return false;
    }

    void Edit(ENTITY_SOURCES *Sources, ID_SOURCE ID, f64 ControlRate, f64 MaxAmplitude, f64 Attack, f64 Decay, f64 SustainAmplitude, f64 Release, bool IsAligned = true)
    {
        //Loop through every source that was added to the system
        for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
        {
            //Find active sources in the system
            ID_SOURCE Source = SystemSources[SystemIndex];
            if(Source == ID)
            {
                //Source is valid - get component
                size_t SourceIndex          = Sources->RetrieveIndex(Source);
                CMP_ADSR &ADSR              = Sources->ADSRs[SourceIndex];

                //Amplitudes
                ADSR.MaxAmplitude           = MaxAmplitude;
                ADSR.SustainAmplitude       = SustainAmplitude;

                //Convert to sample counts for linear ramping
                ADSR.Attack                 = Attack   * ControlRate;
                ADSR.Decay                  = Decay    * ControlRate;
                ADSR.Release                = Release  * ControlRate;

                //Durations
                ADSR.Index                  = 0;
                ADSR.DurationInSamples      = ((Attack + Decay + Release) * ControlRate);
                if(IsAligned)
                {
                    ADSR.DurationInSamples  = NearestPowerOf2(ADSR.DurationInSamples);
                }
                ADSR.IsActive               = true;                
            }
        }
    }

    void RenderToBuffer(CMP_ADSR &ADSR, CMP_BUFFER &Buffer, size_t BufferCount)
    {
        if(ADSR.IsActive)
        {
            f64 Sample = 0;
            for(size_t i = 0; i < BufferCount; ++i)
            {
                //ADSR finished
                if(ADSR.Index == ADSR.DurationInSamples)
                {
                    ADSR.IsActive = false;
                    return;
                }

                //Attack
                if(ADSR.Index <= ADSR.Attack)
                {
                    Sample = ADSR.Index * (ADSR.MaxAmplitude / ADSR.Attack);
                }

                //Decay
                else if(ADSR.Index <= (ADSR.Attack + ADSR.Decay))
                {
                    Sample = ((ADSR.SustainAmplitude - ADSR.MaxAmplitude) / ADSR.Decay) * (ADSR.Index - ADSR.Attack) + ADSR.MaxAmplitude;
                }

                //Sustain
                else if(ADSR.Index <= (ADSR.DurationInSamples - ADSR.Release))
                {
                    Sample = ADSR.SustainAmplitude;
                }

                //Release
                else if(ADSR.Index > (ADSR.DurationInSamples - ADSR.Release))
                {
                    Sample = -(ADSR.SustainAmplitude / ADSR.Release) * (ADSR.Index - (ADSR.DurationInSamples - ADSR.Release)) + ADSR.SustainAmplitude;
                }

                ADSR.Index++;

                Buffer.Data[i] *= Sample;
            }
        }
    }

    void Update(ENTITY_SOURCES *Sources, size_t BufferCount)
    {
        //Loop through every source that was added to the system
        for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
        {
            //Find active sources in the system
            ID_SOURCE Source = SystemSources[SystemIndex];
            if(Source != 0)
            {
                //Source is valid - get component
                size_t SourceIndex = Sources->RetrieveIndex(Source);
                RenderToBuffer(Sources->ADSRs[SourceIndex], Sources->Playbacks[SourceIndex].Buffer, BufferCount);
            }
        }
    }


} SYS_ENVELOPE_ADSR;

typedef struct SYS_ENVELOPE_BREAKPOINT
{
    //Data
    size_t                  SystemCount;    
    ID_SOURCE               *SystemSources;
    
    //Functions
    void Create(MEMORY_ARENA *Arena, size_t Size)
    {
        SystemSources = (ID_SOURCE *) Arena->Alloc((sizeof(ID_SOURCE) * Size), MEMORY_ARENA_ALIGNMENT);
        SystemCount = 0;
    }

    void Destroy(MEMORY_ARENA *Arena)
    {
        Arena->FreeAll();
    }

    void Add(ID_SOURCE ID)
    {
        SystemSources[SystemCount] = ID;
        ++SystemCount;
    }

    bool Remove(ID_SOURCE ID)
    {
        for(size_t i = 0; i <= SystemCount; ++i)
        {
            if(SystemSources[i] == ID)
            {
                SystemSources[i] = 0;
                --SystemCount;
                return true;
            }
        }
        //!Log
        return false;
    }

    void CreateFromFile(ENTITY_SOURCES *Sources, ID_SOURCE ID, char const *File)
    {
        //Loop through every source that was added to the system
        for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
        {
            //Find active sources in the system
            ID_SOURCE Source = SystemSources[SystemIndex];
            if(Source == ID)
            {
                //Source is valid - get component
                size_t SourceIndex          = Sources->RetrieveIndex(Source);
                CMP_BREAKPOINT &Breakpoint  = Sources->Breakpoints[SourceIndex];

                FILE *InputFile = 0;
                fopen_s(&InputFile, File, "r");
                int done = 0;
                int err = 0;

                for(u32 i = 0; i < MAX_BREAKPOINTS && done != 1; ++i)
                {
                    char *Line = fread_csv_line(InputFile, MAX_STRING_LENGTH, &done, &err);
                    if(done != 1)
                    {
                        char **Values = split_on_unescaped_newlines(Line);

                        if(!err)
                        {
                            sscanf(*Values, "%lf,%lf", &Breakpoint.Points[i].Time, &Breakpoint.Points[i].Value);
                            ++Breakpoint.Count;
                        }
                    }
                }
            }
        }
    }




    void EditPoint(ENTITY_SOURCES *Sources, ID_SOURCE ID, size_t PointIndex, f64 Value, f64 Time)
    {
        //Loop through every source that was added to the system
        for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
        {
            //Find active sources in the system
            ID_SOURCE Source = SystemSources[SystemIndex];
            if(Source == ID)
            {
                //Source is valid - get component
                size_t SourceIndex          = Sources->RetrieveIndex(Source);
                CMP_BREAKPOINT &Breakpoint  = Sources->Breakpoints[SourceIndex];
                
                Breakpoint.Init(PointIndex, Value, Time);
            }
        }
    }

    void Update(ENTITY_SOURCES *Sources, SYS_FADE *FadeSystem, f64 Time)
    {
        //Loop through every source that was added to the system
        for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
        {
            //Find active sources in the system
            ID_SOURCE Source = SystemSources[SystemIndex];
            if(Source != 0)
            {
                //Source is valid - get component
                size_t SourceIndex = Sources->RetrieveIndex(Source);
                CMP_BREAKPOINT &Breakpoint = Sources->Breakpoints[SourceIndex];
                CMP_FADE &Amplitude = Sources->Amplitudes[SourceIndex];

                if(!Amplitude.IsFading)
                {
                    bool PointSet = false;
                    while(Breakpoint.Index < Breakpoint.Count && !PointSet)
                    {
                        FadeSystem->Start(Sources, Source, Time, Breakpoint.Points[Breakpoint.Index].Value, Breakpoint.Points[Breakpoint.Index].Time);
                        ++Breakpoint.Index;
                        PointSet = true;
                    }
                }
            }
        }
    }


} SYS_ENVELOPE_BREAKPOINT;

typedef struct POLAR_MIXER
{
    size_t  Count;
    SYS_MIX **Mixes;
} POLAR_MIXER;


//Utility code
#include "polar_dsp.cpp"
#include "polar_render.cpp"
#include "polar_source.cpp"


//Components
#include "component/fade.cpp"

//Entities
#include "entity/sources.cpp"

//Systems
#include "system/play.cpp"
#include "system/fade.cpp"


#include "system/oscillator/noise.cpp"
#include "system/oscillator/sine.cpp"
#include "system/oscillator/square.cpp"

#include "system/wav.cpp"
#include "system/mix.cpp"

